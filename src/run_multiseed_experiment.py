"""Repeat the existing MLP and E-GraphSAGE training recipes on fixed inputs.

The original trainers, architectures, and historical artifacts are unchanged.
Temporary module settings only redirect outputs and change the random seed.
Evaluation wrappers record validation selection and check that each restored
best checkpoint reaches the test evaluator exactly once.
"""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import json
from pathlib import Path
import random
import statistics

import numpy as np
import torch

import train_egraphsage as gnn
import train_mlpbaseline as mlp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "multiseed"
SEEDS = [42, 7, 21, 84, 123]
METRIC_NAMES = ["accuracy", "precision", "recall", "f1"]


def model_state_hash(model: torch.nn.Module) -> str:
    """Compare weights without another forward pass or consuming RNG state."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class EvaluationTracker:
    """Observe existing evaluators without changing their returned metrics."""

    def __init__(self, expected_epochs: int) -> None:
        self.expected_epochs = expected_epochs
        self.validation_history = []
        self.best_epoch = 0
        self.best_validation_f1 = -1.0
        self.best_weights_hash = None
        self.test_evaluations = 0

    def evaluate(self, model, role: str, original_evaluation):
        if role == "validation":
            if self.test_evaluations or len(self.validation_history) >= self.expected_epochs:
                raise RuntimeError("Unexpected validation evaluation order.")
        elif role == "test":
            if len(self.validation_history) != self.expected_epochs:
                raise RuntimeError("Test evaluation requested before training finished.")
            if self.test_evaluations:
                raise RuntimeError("A second test evaluation was requested.")
            if model_state_hash(model) != self.best_weights_hash:
                raise RuntimeError("The best-validation weights were not restored.")
            self.test_evaluations += 1
        else:
            raise RuntimeError(f"Unexpected evaluator input: {role}")

        metrics = original_evaluation()
        if not all(np.isfinite(float(metrics[name])) for name in ["loss", *METRIC_NAMES]):
            raise ValueError("Non-finite evaluation metrics encountered.")
        if role == "validation":
            epoch = len(self.validation_history) + 1
            self.validation_history.append({
                "epoch": epoch,
                "validation_f1": float(metrics["f1"]),
                "validation_loss": float(metrics["loss"]),
            })
            # This is the same strict-improvement/earliest-tie rule used
            # inside both original trainers. It does not alter selection.
            if metrics["f1"] > self.best_validation_f1:
                self.best_epoch = epoch
                self.best_validation_f1 = float(metrics["f1"])
                self.best_weights_hash = model_state_hash(model)
        return metrics


@contextmanager
def temporary_settings(module, overrides: dict):
    """Restore every module setting even if a run fails."""
    originals = {name: getattr(module, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


def run_one(model_name: str, seed: int, output_directory: Path) -> dict:
    """Call an existing trainer with isolated paths and evaluation tracking."""
    output_directory.mkdir(parents=True, exist_ok=False)
    trainer = mlp if model_name == "mlp" else gnn
    tracker = EvaluationTracker(trainer.NUMBER_OF_EPOCHS)
    roles = {
        trainer.TRAIN_PATH: "train",
        trainer.VALIDATION_PATH: "validation",
        trainer.TEST_PATH: "test",
    }
    input_roles = {}
    overrides = {
        "RANDOM_SEED": seed,
        "RESULTS_DIRECTORY": output_directory,
        "MODEL_PATH": output_directory / "checkpoint.pt",
        "METRICS_PATH": output_directory / "trainer_metrics.json",
    }

    if model_name == "mlp":
        original_load = mlp.load_dataset
        original_evaluate = mlp.evaluate_model

        def load_dataset(path):
            dataset = original_load(path)
            input_roles[id(dataset)] = roles[path]
            return dataset

        def evaluate_model(model, data_loader):
            role = input_roles[id(data_loader.dataset)]
            return tracker.evaluate(
                model, role, lambda: original_evaluate(model, data_loader)
            )

        overrides.update(load_dataset=load_dataset, evaluate_model=evaluate_model)
    elif model_name == "egraphsage":
        original_load = gnn.load_graph
        original_evaluate = gnn.evaluate_model

        def load_graph(path):
            # Preserve the GNN trainer's stronger load-after-selection rule.
            if roles[path] == "test" and len(tracker.validation_history) != tracker.expected_epochs:
                raise RuntimeError("Test graph loaded before training finished.")
            graph = original_load(path)
            input_roles[id(graph)] = roles[path]
            return graph

        def evaluate_model(model, graph, loss_function):
            return tracker.evaluate(
                model, input_roles[id(graph)],
                lambda: original_evaluate(model, graph, loss_function),
            )

        overrides.update(
            load_graph=load_graph,
            evaluate_model=evaluate_model,
            HISTORY_PATH=output_directory / "training_history.json",
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Extra Python/NumPy seeds do not change either architecture or recipe.
    # The native main() functions still set their own PyTorch seed normally.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with (output_directory / "training.log").open("x", encoding="utf-8") as log:
        with temporary_settings(trainer, overrides), redirect_stdout(log), redirect_stderr(log):
            trainer.main()

    if tracker.test_evaluations != 1:
        raise RuntimeError("Expected exactly one test evaluation.")
    native_metrics = json.loads((output_directory / "trainer_metrics.json").read_text())
    if model_name == "egraphsage":
        if native_metrics["best_epoch"] != tracker.best_epoch:
            raise RuntimeError("GNN checkpoint epoch disagrees with validation history.")
        if native_metrics["best_validation_f1"] != tracker.best_validation_f1:
            raise RuntimeError("GNN checkpoint F1 disagrees with validation history.")

    result = {
        "model": model_name,
        "seed": seed,
        "best_epoch": tracker.best_epoch,
        "best_validation_f1": tracker.best_validation_f1,
        **{f"test_{name}": float(native_metrics[name]) for name in METRIC_NAMES},
        "confusion_matrix": native_metrics["confusion_matrix"],
        "test_evaluations": tracker.test_evaluations,
        "restored_best_checkpoint_verified": True,
        "best_weights_sha256": tracker.best_weights_hash,
    }
    (output_directory / "run_summary.json").write_text(
        json.dumps(result, indent=4) + "\n", encoding="utf-8"
    )
    (output_directory / "validation_history.json").write_text(
        json.dumps(tracker.validation_history, indent=4) + "\n", encoding="utf-8"
    )
    return result


def summarize(results: list[dict]) -> dict:
    """Use sample standard deviations (ddof=1) across the five seeds."""
    summary = {}
    for model_name in ("mlp", "egraphsage"):
        runs = [result for result in results if result["model"] == model_name]
        f1_values = [run["test_f1"] for run in runs]
        summary[model_name] = {
            "runs": len(runs),
            "mean_test_f1": statistics.mean(f1_values),
            "std_test_f1": statistics.stdev(f1_values),
            "std_ddof": 1,
            **{
                f"mean_test_{name}": statistics.mean(run[f"test_{name}"] for run in runs)
                for name in ("accuracy", "precision", "recall")
            },
        }
    paired_differences = []
    for seed in sorted({run["seed"] for run in results}):
        paired = {run["model"]: run["test_f1"] for run in results if run["seed"] == seed}
        paired_differences.append({"seed": seed, "f1_difference": paired["egraphsage"] - paired["mlp"]})
    differences = [pair["f1_difference"] for pair in paired_differences]
    summary["comparison"] = {
        "mean_f1_difference": summary["egraphsage"]["mean_test_f1"] - summary["mlp"]["mean_test_f1"],
        "paired_differences": paired_differences,
        "std_paired_f1_difference": statistics.stdev(differences),
        "egraphsage_wins": sum(value > 0 for value in differences),
        "mlp_wins": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
    }
    return summary


def file_hashes(paths: list[Path]) -> dict:
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def main() -> None:
    # Refuse to replace a previous multi-seed run or any historical output.
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Output folder already exists: {OUTPUT_ROOT}")
    protected_paths = list((PROJECT_ROOT / "src").glob("*.py"))
    protected_paths += list((PROJECT_ROOT / "tests").glob("*.py"))
    protected_paths += list((PROJECT_ROOT / "examples").rglob("*.py"))
    protected_paths += [path for path in (PROJECT_ROOT / "results").rglob("*") if path.is_file()]
    protected_paths += [path for path in (PROJECT_ROOT / "data/processed").rglob("*") if path.is_file()]
    protected_hashes = file_hashes(protected_paths)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    protocol = {
        "seeds": SEEDS,
        "device": "cpu",
        "cpu_threads": 1,
        "deterministic_algorithms": True,
        "torch_version": str(torch.__version__),
        "fixed_inputs_and_historical_files_sha256": protected_hashes,
        "training_procedures": {
            "mlp": {
                "epochs": mlp.NUMBER_OF_EPOCHS,
                "optimizer": "Adam",
                "learning_rate": mlp.LEARNING_RATE,
                "batch_size": mlp.BATCH_SIZE,
                "shuffle_training": True,
                "loss": "CrossEntropyLoss",
            },
            "egraphsage": {
                "epochs": gnn.NUMBER_OF_EPOCHS,
                "optimizer": "Adam",
                "learning_rate": gnn.LEARNING_RATE,
                "batch_size": "full graph",
                "shuffle_training": False,
                "loss": "CrossEntropyLoss",
            },
        },
        "selection_rule": "maximum validation F1 each epoch; earliest tied epoch",
        "test_rule": "evaluate once per model/seed after restoring the selected checkpoint",
        "split_regeneration": False,
        "preprocessing_refit": False,
        "sample_standard_deviation_ddof": 1,
        "comparison_scope": "existing training recipes, not an architecture-only ablation",
    }
    (OUTPUT_ROOT / "protocol.json").write_text(
        json.dumps(protocol, indent=4) + "\n", encoding="utf-8"
    )
    results = []
    for seed in SEEDS:
        for model_name in ("mlp", "egraphsage"):
            print(f"Starting {model_name}, seed {seed}", flush=True)
            result = run_one(model_name, seed, OUTPUT_ROOT / model_name / f"seed_{seed}")
            results.append(result)
            if file_hashes(protected_paths) != protected_hashes:
                raise RuntimeError("A protected source, dataset, or historical artifact changed.")
            print(
                f"Finished {model_name}, seed {seed}: best epoch={result['best_epoch']}, "
                f"validation F1={result['best_validation_f1']:.6f}, test F1={result['test_f1']:.6f}",
                flush=True,
            )

    report = {"seeds": SEEDS, "runs": results, "statistics": summarize(results)}
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(report, indent=4) + "\n", encoding="utf-8"
    )
    print("\n" + json.dumps(report["statistics"], indent=4), flush=True)
    print(f"\nSaved separate results to {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
