"""Check aggregation semantics, label exclusion, and holdout isolation."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egraphsage_model import EGraphSAGE, aggregate_connected_edges
from toy_egraphsage import aggregate_connected_edges as toy_aggregation
import train_egraphsage


def example_graph() -> Data:
    # Includes parallel flows, both directions, a self-loop, and an
    # isolated node, so averaging only outgoing flows would fail the test.
    return Data(
        x=torch.ones((4, 2)),
        edge_index=torch.tensor([[0, 0, 1, 2, 2], [1, 1, 0, 2, 0]]),
        edge_attr=torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.]]),
        edge_y=torch.tensor([0, 0, 1, 1, 0]),
    )


class EGraphSAGETests(unittest.TestCase):
    def test_connected_means_match_expected_values_and_toy(self):
        graph = example_graph()
        expected = torch.tensor([
            [4.5, 5.5], [3., 4.], [23. / 3., 26. / 3.], [0., 0.]
        ])
        actual = aggregate_connected_edges(graph)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, toy_aggregation(graph))

    def test_predictions_ignore_labels_and_support_finite_gradients(self):
        torch.manual_seed(42)
        graph = example_graph()
        model = EGraphSAGE(node_feature_size=2, edge_feature_size=2, hidden_size=4)
        logits = model(graph)
        self.assertEqual(tuple(logits.shape), (5, 2))
        unlabeled_graph = Data(
            x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr
        )
        torch.testing.assert_close(logits, model(unlabeled_graph))
        changed_labels = graph.clone()
        changed_labels.edge_y = 1 - changed_labels.edge_y
        torch.testing.assert_close(logits, model(changed_labels))

        loss = nn.CrossEntropyLoss()(logits, graph.edge_y)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_selection_uses_validation_and_loads_test_only_after_training(self):
        # Use tiny synthetic graphs and deliberately make epoch 2 best.
        # No real dataset training or test evaluation happens in this test.
        train_graph = example_graph()
        train_graph.edge_y = torch.tensor([0, 0, 0, 1, 1])
        # Ten balanced edges satisfy the training balance check.
        train_graph.edge_index = train_graph.edge_index.repeat(1, 2)
        train_graph.edge_attr = train_graph.edge_attr.repeat(2, 1)
        train_graph.edge_y = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
        validation_graph = example_graph()
        test_graph = example_graph()
        validation_scores = [0.7, 0.9, 0.8]
        validation_calls = 0
        test_calls = 0
        best_weights = None

        def fake_load(path):
            if path == train_egraphsage.TRAIN_PATH:
                return train_graph
            if path == train_egraphsage.VALIDATION_PATH:
                return validation_graph
            self.assertEqual(path, train_egraphsage.TEST_PATH)
            self.assertEqual(validation_calls, 3)
            return test_graph

        def fake_evaluate(model, graph, loss_function):
            nonlocal validation_calls, test_calls, best_weights
            if graph is validation_graph:
                score = validation_scores[validation_calls]
                validation_calls += 1
                if validation_calls == 2:
                    best_weights = {
                        name: value.clone() for name, value in model.state_dict().items()
                    }
            else:
                self.assertIs(graph, test_graph)
                test_calls += 1
                self.assertEqual(validation_calls, 3)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, best_weights[name])
                score = 0.6
            return {
                "loss": 0.5, "accuracy": score, "precision": score,
                "recall": score, "f1": score, "confusion_matrix": [[1, 1], [1, 2]],
            }

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            overrides = {
                "NUMBER_OF_EPOCHS": 3,
                "EDGE_FEATURE_COLUMNS": ["feature_a", "feature_b"],
                "RESULTS_DIRECTORY": directory,
                "MODEL_PATH": directory / "model.pt",
                "METRICS_PATH": directory / "metrics.json",
                "HISTORY_PATH": directory / "history.json",
            }
            with mock.patch.multiple(train_egraphsage, **overrides), \
                    mock.patch.object(train_egraphsage, "load_graph", side_effect=fake_load), \
                    mock.patch.object(train_egraphsage, "evaluate_model", side_effect=fake_evaluate), \
                    contextlib.redirect_stdout(io.StringIO()):
                train_egraphsage.main()
            report = json.loads((directory / "metrics.json").read_text())
            self.assertEqual(report["best_epoch"], 2)
            self.assertEqual(report["best_validation_f1"], 0.9)
            self.assertEqual(report["f1"], 0.6)
            self.assertEqual(test_calls, 1)


if __name__ == "__main__":
    unittest.main()
