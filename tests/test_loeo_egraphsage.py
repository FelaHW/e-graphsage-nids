"""Verify target exclusion, graph corner cases, gradients, and holdout use."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from egraphsage_model import EGraphSAGE
from loeo_egraphsage_model import LOEOEGraphSAGE, aggregate_without_target
import run_loeo_experiment


def graph_from_edges(edges, features, number_of_nodes):
    features = torch.tensor(features, dtype=torch.float32)
    return Data(
        x=torch.ones(number_of_nodes, features.shape[1]),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=features,
        edge_y=torch.arange(len(edges)) % 2,
    )


def multiple_edge_graph():
    return graph_from_edges(
        [(0, 1), (0, 2), (3, 0), (1, 2)],
        [[1, 2], [3, 4], [5, 6], [7, 8]], 4,
    )


class LOEOTests(unittest.TestCase):
    def test_multiple_edges_give_target_specific_contexts(self):
        source, destination = aggregate_without_target(multiple_edge_graph())
        torch.testing.assert_close(source[0], torch.tensor([4., 5.]))
        torch.testing.assert_close(destination[0], torch.tensor([7., 8.]))
        # Same source node, different excluded target -> different mean.
        torch.testing.assert_close(source[1], torch.tensor([3., 4.]))

    def test_target_only_endpoints_have_zero_context(self):
        graph = graph_from_edges([(0, 1)], [[12, 24]], 3)
        source, destination = aggregate_without_target(graph)
        torch.testing.assert_close(source, torch.zeros(1, 2))
        torch.testing.assert_close(destination, torch.zeros(1, 2))

    def test_parallel_edges_remain_neighbors(self):
        graph = graph_from_edges([(0, 1), (0, 1)], [[1, 2], [10, 20]], 2)
        source, destination = aggregate_without_target(graph)
        expected = torch.tensor([[10., 20.], [1., 2.]])
        torch.testing.assert_close(source, expected)
        torch.testing.assert_close(destination, expected)

    def test_reverse_edges_remain_neighbors(self):
        graph = graph_from_edges([(0, 1), (1, 0)], [[1, 2], [10, 20]], 2)
        source, destination = aggregate_without_target(graph)
        expected = torch.tensor([[10., 20.], [1., 2.]])
        torch.testing.assert_close(source, expected)
        torch.testing.assert_close(destination, expected)

    def test_self_loop_excludes_both_incidences(self):
        graph = graph_from_edges([(0, 0), (0, 1)], [[10, 20], [2, 4]], 2)
        source, destination = aggregate_without_target(graph)
        torch.testing.assert_close(source, torch.tensor([[2., 4.], [10., 20.]]))
        torch.testing.assert_close(destination, torch.tensor([[2., 4.], [0., 0.]]))
        lone_loop = graph_from_edges([(0, 0)], [[10, 20]], 1)
        for context in aggregate_without_target(lone_loop):
            torch.testing.assert_close(context, torch.zeros(1, 2))

    def test_target_features_cannot_change_their_own_context_or_prediction(self):
        graph = multiple_edge_graph()
        changed = graph.clone()
        changed.edge_attr[0] = torch.tensor([999.5, -321.25])
        original_contexts = aggregate_without_target(graph)
        changed_contexts = aggregate_without_target(changed)
        for original, new in zip(original_contexts, changed_contexts):
            torch.testing.assert_close(original[0], new[0])
        # Those features are still legitimate context for a different edge.
        self.assertFalse(torch.equal(original_contexts[0][1], changed_contexts[0][1]))
        torch.manual_seed(42)
        model = LOEOEGraphSAGE(2, 2, hidden_size=4)
        torch.testing.assert_close(model(graph)[0], model(changed)[0])
        graph.edge_attr.requires_grad_()
        own_logits = model(graph)[0].sum()
        feature_gradients = torch.autograd.grad(own_logits, graph.edge_attr)[0]
        torch.testing.assert_close(feature_gradients[0], torch.zeros(2), rtol=0, atol=0)

    def test_labels_do_not_influence_predictions(self):
        graph = multiple_edge_graph()
        model = LOEOEGraphSAGE(2, 2, hidden_size=4)
        changed = graph.clone()
        changed.edge_y = 1 - changed.edge_y
        unlabeled = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
        torch.testing.assert_close(model(graph), model(changed))
        torch.testing.assert_close(model(graph), model(unlabeled))

    def test_finite_forward_backward_and_output_shape(self):
        graph = graph_from_edges(
            [(0, 0), (0, 1), (0, 1), (1, 0), (2, 3)],
            [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], 4,
        )
        graph.edge_attr.requires_grad_()
        model = LOEOEGraphSAGE(2, 2, hidden_size=4)
        logits = model(graph)
        self.assertEqual(tuple(logits.shape), (5, 2))
        self.assertTrue(torch.isfinite(logits).all().item())
        loss = torch.nn.CrossEntropyLoss()(logits, graph.edge_y)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
        self.assertTrue(torch.isfinite(graph.edge_attr.grad).all().item())

    def test_control_dimensions_and_seed_initialization_are_preserved(self):
        torch.manual_seed(42)
        control = EGraphSAGE(8, 8, hidden_size=32)
        torch.manual_seed(42)
        loeo = LOEOEGraphSAGE(8, 8, hidden_size=32)
        self.assertEqual(sum(parameter.numel() for parameter in loeo.parameters()), 674)
        for name, value in control.state_dict().items():
            torch.testing.assert_close(value, loeo.state_dict()[name], rtol=0, atol=0)

    def test_training_restores_validation_checkpoint_before_one_test_call(self):
        base = graph_from_edges(
            [(0, 1), (1, 0), (0, 2), (2, 1)],
            [[value] * 8 for value in (1, 2, 3, 4)], 3,
        )
        train_graph, validation_graph, test_graph = base, base.clone(), base.clone()
        validation_calls = 0
        test_calls = 0
        best_weights = None

        def fake_load(path):
            if path == run_loeo_experiment.TRAIN_PATH:
                return train_graph
            if path == run_loeo_experiment.VALIDATION_PATH:
                return validation_graph
            self.assertEqual(path, run_loeo_experiment.TEST_PATH)
            self.assertEqual(validation_calls, 2)
            return test_graph

        def fake_evaluate(model, graph, loss_function):
            nonlocal validation_calls, test_calls, best_weights
            if graph is validation_graph:
                validation_calls += 1
                score = 0.9 if validation_calls == 1 else 0.8
                if validation_calls == 1:
                    best_weights = {name: value.clone() for name, value in model.state_dict().items()}
            else:
                self.assertIs(graph, test_graph)
                test_calls += 1
                self.assertEqual(validation_calls, 2)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, best_weights[name])
                score = 0.7
            return {
                "loss": 0.5, "accuracy": score, "precision": score,
                "recall": score, "f1": score, "confusion_matrix": [[1, 1], [1, 1]],
            }

        with tempfile.TemporaryDirectory() as temporary_directory:
            with mock.patch.object(run_loeo_experiment, "NUMBER_OF_EPOCHS", 2), \
                    mock.patch.object(run_loeo_experiment, "load_graph", side_effect=fake_load), \
                    mock.patch.object(run_loeo_experiment, "evaluate_model", side_effect=fake_evaluate), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = run_loeo_experiment.run_seed(42, Path(temporary_directory) / "seed_42")
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(result["best_validation_f1"], 0.9)
            self.assertEqual(result["test_f1"], 0.7)
            self.assertEqual(test_calls, 1)
            checkpoint = torch.load(
                Path(temporary_directory) / "seed_42/checkpoint.pt", weights_only=True
            )
            self.assertEqual(checkpoint["model_class"], "LOEOEGraphSAGE")
            self.assertFalse(checkpoint["target_edge_features_in_aggregation"])


if __name__ == "__main__":
    unittest.main()
