"""Verify assignment-only changes and reuse of the original hybrid trainer."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hybrid_egraphsage_model import HybridEGraphSAGE
from shuffled_hybrid_model import ShuffledHybridEGraphSAGE, make_context_derangement, permutation_seed
import run_hybrid_experiment as real_hybrid
import run_shuffled_hybrid_experiment as shuffled_runner


def example_graph():
    return Data(
        x=torch.ones(6, 8),
        edge_index=torch.tensor([[0, 2, 2, 5], [1, 3, 4, 5]]),
        edge_attr=torch.arange(32, dtype=torch.float32).reshape(4, 8) / 10,
        edge_y=torch.tensor([0, 1, 0, 1]),
    )


class ShuffledHybridTests(unittest.TestCase):
    def test_permutation_is_deterministic_seeded_split_specific_and_a_derangement(self):
        first = make_context_derangement(101, 42, "train")
        torch.testing.assert_close(first, make_context_derangement(101, 42, "train"))
        self.assertFalse(torch.equal(first, make_context_derangement(101, 7, "train")))
        self.assertFalse(torch.equal(first, make_context_derangement(101, 42, "validation")))
        self.assertNotEqual(permutation_seed(42, "validation"), permutation_seed(42, "test"))
        for size in (0, 2, 3, 4, 101):
            permutation = make_context_derangement(size, 42, "test")
            torch.testing.assert_close(permutation.sort().values, torch.arange(size))
            self.assertFalse((permutation == torch.arange(size)).any().item())
        with self.assertRaisesRegex(ValueError, "one-edge"):
            make_context_derangement(1, 42, "train")

    def test_permutation_does_not_consume_model_rng_or_change_initialization(self):
        torch.manual_seed(42)
        expected = HybridEGraphSAGE(8, 8)
        torch.manual_seed(42)
        before = torch.get_rng_state().clone()
        make_context_derangement(101, 42, "train")
        torch.testing.assert_close(torch.get_rng_state(), before)
        actual = ShuffledHybridEGraphSAGE(8, 8)
        self.assertEqual(sum(parameter.numel() for parameter in actual.parameters()), 3442)
        for name, value in expected.state_dict().items():
            torch.testing.assert_close(value, actual.state_dict()[name], rtol=0, atol=0)

    def test_direct_features_labels_and_graph_are_unchanged_and_context_pairs_move(self):
        graph = example_graph()
        originals = {name: getattr(graph, name).clone() for name in ("x", "edge_index", "edge_attr", "edge_y")}
        graph.context_permutation = torch.tensor([1, 2, 3, 0])
        model = ShuffledHybridEGraphSAGE(8, 8)
        expected_source, expected_destination = HybridEGraphSAGE.context_embeddings(model, graph)
        actual_source, actual_destination = model.context_embeddings(graph)
        torch.testing.assert_close(actual_source, expected_source[graph.context_permutation])
        torch.testing.assert_close(actual_destination, expected_destination[graph.context_permutation])
        classifier_inputs = []
        hook = model.edge_classifier[0].register_forward_pre_hook(
            lambda module, arguments: classifier_inputs.append(arguments[0].detach().clone())
        )
        try:
            model(graph)
        finally:
            hook.remove()
        torch.testing.assert_close(classifier_inputs[0][:, :8], originals["edge_attr"])
        torch.testing.assert_close(classifier_inputs[0][:, 8:40], actual_source)
        torch.testing.assert_close(classifier_inputs[0][:, 40:], actual_destination)
        for name, before in originals.items():
            torch.testing.assert_close(getattr(graph, name), before, rtol=0, atol=0)

    def test_distinct_tagged_context_pairs_are_reassigned_together(self):
        graph = example_graph()
        graph.context_permutation = torch.tensor([1, 2, 3, 0])
        source_tags = torch.arange(4 * 32, dtype=torch.float32).reshape(4, 32)
        destination_tags = source_tags + 1000
        model = ShuffledHybridEGraphSAGE(8, 8)
        with mock.patch.object(HybridEGraphSAGE, "context_embeddings", return_value=(source_tags, destination_tags)):
            source, destination = model.context_embeddings(graph)
        self.assertFalse(torch.equal(source, source_tags))
        torch.testing.assert_close(source, source_tags[graph.context_permutation])
        torch.testing.assert_close(destination, destination_tags[graph.context_permutation])
        torch.testing.assert_close(destination - source, torch.full_like(source, 1000))

    def test_labels_do_not_control_assignments_or_predictions(self):
        graph = example_graph()
        changed = graph.clone()
        changed.edge_y = 1 - graph.edge_y
        graph.context_permutation = make_context_derangement(graph.num_edges, 21, "train")
        changed.context_permutation = make_context_derangement(changed.num_edges, 21, "train")
        torch.testing.assert_close(graph.context_permutation, changed.context_permutation)
        unlabeled = Data(
            x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr,
            context_permutation=graph.context_permutation,
        )
        model = ShuffledHybridEGraphSAGE(8, 8)
        torch.testing.assert_close(model(graph), model(changed))
        torch.testing.assert_close(model(graph), model(unlabeled))

    def test_forward_backward_is_finite_and_assignments_stay_fixed(self):
        graph = example_graph()
        graph.context_permutation = make_context_derangement(graph.num_edges, 84, "train")
        before = graph.context_permutation.clone()
        model = ShuffledHybridEGraphSAGE(8, 8)
        logits = model(graph)
        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertTrue(torch.isfinite(logits).all().item())
        loss = torch.nn.CrossEntropyLoss()(logits, graph.edge_y)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
        torch.testing.assert_close(model(graph), logits)
        torch.testing.assert_close(graph.context_permutation, before)

    def test_diagnostics_report_availability_migration_and_shared_endpoints(self):
        graph = example_graph()
        graph.context_permutation = torch.tensor([1, 2, 3, 0])
        result = shuffled_runner.permutation_diagnostics(graph, 42, "test")
        self.assertEqual(result["fixed_points"], 0)
        self.assertEqual(result["availability_transitions"], {
            "empty_to_empty": 1, "empty_to_nonempty": 1,
            "nonempty_to_empty": 1, "nonempty_to_nonempty": 1,
        })
        self.assertEqual(result["recipient_shares_endpoint_with_donor"], 1)

    def test_validation_selection_delayed_test_loading_and_one_test_forward(self):
        base = example_graph()
        train_graph, validation_graph, test_graph = base, base.clone(), base.clone()
        validation_calls = 0
        test_calls = 0
        best_weights = None
        original_evaluate = real_hybrid.evaluate_model

        def fake_load(path):
            if path == real_hybrid.TRAIN_PATH:
                return train_graph
            if path == real_hybrid.VALIDATION_PATH:
                return validation_graph
            self.assertEqual(path, real_hybrid.TEST_PATH)
            self.assertEqual(validation_calls, 2)
            return test_graph

        def fake_evaluate(model, graph, loss_function):
            nonlocal validation_calls, test_calls, best_weights
            self.assertIsInstance(model, ShuffledHybridEGraphSAGE)
            if graph is validation_graph:
                validation_calls += 1
                score = 0.9 if validation_calls == 1 else 0.8
                if validation_calls == 1:
                    best_weights = {name: value.clone() for name, value in model.state_dict().items()}
                return {
                    "loss": 0.5, "accuracy": score, "precision": score,
                    "recall": score, "f1": score, "confusion_matrix": [[1, 1], [1, 1]],
                }
            self.assertIs(graph, test_graph)
            test_calls += 1
            self.assertEqual(validation_calls, 2)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, best_weights[name])
            return original_evaluate(model, graph, loss_function)

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(real_hybrid, "NUMBER_OF_EPOCHS", 2), \
                    mock.patch.object(real_hybrid, "load_graph", side_effect=fake_load), \
                    mock.patch.object(real_hybrid, "evaluate_model", side_effect=fake_evaluate), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = shuffled_runner.run_seed(42, Path(directory) / "seed_42")
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(result["best_validation_f1"], 0.9)
            self.assertEqual(test_calls, 1)
            self.assertEqual(result["test_forward_passes"], 1)
            self.assertEqual(result["fixed_points"], {"train": 0, "validation": 0, "test": 0})
            checkpoint = torch.load(Path(directory) / "seed_42/checkpoint.pt", weights_only=True)
            self.assertEqual(checkpoint["model_class"], "ShuffledHybridEGraphSAGE")
        self.assertIs(real_hybrid.HybridEGraphSAGE, HybridEGraphSAGE)


if __name__ == "__main__":
    unittest.main()
