"""Test holdout guards and aggregate statistics using synthetic values."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_multiseed_experiment import EvaluationTracker, summarize, temporary_settings


def metrics(f1):
    return {"loss": 0.5, "accuracy": f1, "precision": f1, "recall": f1, "f1": f1}


class MultiSeedTests(unittest.TestCase):
    def test_test_evaluation_requires_finished_training_and_restored_best_weights(self):
        model = torch.nn.Linear(1, 2)
        tracker = EvaluationTracker(expected_epochs=2)
        with self.assertRaisesRegex(RuntimeError, "before training"):
            tracker.evaluate(model, "test", lambda: metrics(0.99))
        tracker.evaluate(model, "validation", lambda: metrics(0.9))
        best_weights = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.no_grad():
            model.weight.add_(1)
        tracker.evaluate(model, "validation", lambda: metrics(0.8))
        with self.assertRaisesRegex(RuntimeError, "not restored"):
            tracker.evaluate(model, "test", lambda: metrics(0.99))
        model.load_state_dict(best_weights)
        tracker.evaluate(model, "test", lambda: metrics(0.99))
        self.assertEqual(tracker.best_epoch, 1)
        self.assertEqual(tracker.best_validation_f1, 0.9)
        self.assertEqual(tracker.test_evaluations, 1)
        with self.assertRaisesRegex(RuntimeError, "second test"):
            tracker.evaluate(model, "test", lambda: metrics(0.99))

    def test_sample_std_and_paired_differences(self):
        runs = []
        for seed, mlp_f1, gnn_f1 in [(1, 0.8, 0.9), (2, 1.0, 0.8)]:
            for name, f1 in [("mlp", mlp_f1), ("egraphsage", gnn_f1)]:
                runs.append({
                    "model": name, "seed": seed,
                    **{f"test_{metric}": f1 for metric in ("f1", "accuracy", "precision", "recall")},
                })
        summary = summarize(runs)
        self.assertAlmostEqual(summary["mlp"]["mean_test_f1"], 0.9)
        self.assertAlmostEqual(summary["mlp"]["std_test_f1"], math.sqrt(0.02))
        self.assertAlmostEqual(summary["comparison"]["mean_f1_difference"], -0.05)
        self.assertEqual(summary["comparison"]["egraphsage_wins"], 1)
        self.assertEqual(summary["comparison"]["mlp_wins"], 1)

    def test_temporary_paths_are_restored_after_failure(self):
        module = SimpleNamespace(MODEL_PATH="historical.pt", RANDOM_SEED=42)
        with self.assertRaisesRegex(ValueError, "synthetic failure"):
            with temporary_settings(module, {"MODEL_PATH": "separate.pt", "RANDOM_SEED": 7}):
                self.assertEqual(module.MODEL_PATH, "separate.pt")
                raise ValueError("synthetic failure")
        self.assertEqual(module.MODEL_PATH, "historical.pt")
        self.assertEqual(module.RANDOM_SEED, 42)


if __name__ == "__main__":
    unittest.main()
