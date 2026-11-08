"""Check direct/context separation and subgroup metrics from one evaluation."""

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
import run_hybrid_experiment


def example_graph():
    # Two empty-context targets (including a self-loop), two with neighbors.
    return Data(
        x=torch.ones(6, 8),
        edge_index=torch.tensor([[0, 2, 2, 5], [1, 3, 4, 5]]),
        edge_attr=torch.arange(32, dtype=torch.float32).reshape(4, 8) / 10,
        edge_y=torch.tensor([0, 1, 0, 1]),
    )


class HybridTests(unittest.TestCase):
    def test_dimensions_and_finite_gradients(self):
        model = HybridEGraphSAGE(8, 8)
        self.assertEqual(model.edge_classifier[0].in_features, 72)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 3442)
        graph = example_graph()
        logits = model(graph)
        self.assertEqual(tuple(logits.shape), (4, 2))
        loss = torch.nn.CrossEntropyLoss()(logits, graph.edge_y)
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_own_context_excludes_target_but_direct_path_can_use_it(self):
        model = HybridEGraphSAGE(8, 8)
        graph = example_graph()
        changed = graph.clone()
        changed.edge_attr[1] += 10
        for before, after in zip(model.context_embeddings(graph), model.context_embeddings(changed)):
            torch.testing.assert_close(before[1], after[1])
        # Set a simple positive direct-feature path, independent of context.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.edge_classifier[0].weight[0, 0] = 1
            model.edge_classifier[2].weight[0, 0] = 1
            model.edge_classifier[4].weight[0, 0] = 1
        self.assertGreater(model(changed)[1, 0].item(), model(graph)[1, 0].item())
        # Direct features still work for a target with both contexts empty.
        changed.edge_attr[0, 0] = 3
        self.assertGreater(model(changed)[0, 0].item(), model(graph)[0, 0].item())

    def test_labels_do_not_influence_predictions(self):
        model = HybridEGraphSAGE(8, 8)
        graph = example_graph()
        changed = graph.clone()
        changed.edge_y = 1 - graph.edge_y
        unlabeled = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
        torch.testing.assert_close(model(graph), model(changed))
        torch.testing.assert_close(model(graph), model(unlabeled))

    def test_groups_use_topology_and_remove_both_self_loop_incidences(self):
        graph = example_graph()
        # Zero-valued features do not mean that a neighborhood is empty.
        graph.edge_attr.zero_()
        masks = run_hybrid_experiment.neighborhood_group_masks(graph)
        torch.testing.assert_close(masks["both_endpoints_empty"], torch.tensor([True, False, False, True]))
        torch.testing.assert_close(masks["at_least_one_nonempty"], torch.tensor([False, True, True, False]))

    def test_subgroups_reuse_exactly_one_test_forward(self):
        class CountingModel(HybridEGraphSAGE):
            calls = 0

            def forward(self, graph):
                self.calls += 1
                return super().forward(graph)

        model = CountingModel(8, 8)
        result = run_hybrid_experiment.evaluate_test_with_subgroups(
            model, example_graph(), torch.nn.CrossEntropyLoss()
        )
        self.assertEqual(model.calls, 1)
        groups = result["subgroups"]
        self.assertEqual(groups["both_endpoints_empty"]["edges"], 2)
        self.assertEqual(groups["at_least_one_nonempty"]["edges"], 2)
        combined = sum(torch.tensor(group["confusion_matrix"]) for group in groups.values())
        torch.testing.assert_close(combined, torch.tensor(result["confusion_matrix"]))
        self.assertEqual(len(result["prediction_artifact"]["predictions"]), 4)

    def test_training_selects_validation_checkpoint_before_one_test_forward(self):
        base = example_graph()
        train_graph, validation_graph, test_graph = base, base.clone(), base.clone()
        validation_calls = 0
        test_calls = 0
        best_weights = None
        original_evaluate = run_hybrid_experiment.evaluate_model

        def fake_load(path):
            if path == run_hybrid_experiment.TRAIN_PATH:
                return train_graph
            if path == run_hybrid_experiment.VALIDATION_PATH:
                return validation_graph
            self.assertEqual(path, run_hybrid_experiment.TEST_PATH)
            self.assertEqual(validation_calls, 2)
            return test_graph

        def fake_evaluate(model, graph, loss_function):
            nonlocal validation_calls, test_calls, best_weights
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
            with mock.patch.object(run_hybrid_experiment, "NUMBER_OF_EPOCHS", 2), \
                    mock.patch.object(run_hybrid_experiment, "load_graph", side_effect=fake_load), \
                    mock.patch.object(run_hybrid_experiment, "evaluate_model", side_effect=fake_evaluate), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = run_hybrid_experiment.run_seed(42, Path(directory) / "seed_42")
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(result["best_validation_f1"], 0.9)
            self.assertEqual(test_calls, 1)
            self.assertEqual(result["test_forward_passes"], 1)
            self.assertEqual(sum(group["edges"] for group in result["subgroups"].values()), 4)


if __name__ == "__main__":
    unittest.main()
