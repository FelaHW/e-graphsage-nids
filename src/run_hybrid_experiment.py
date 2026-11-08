"""Five-seed direct-edge + LOEO-context experiment with one-pass subgroups.

No previous model, graph, preprocessing, split, or result is modified. Test
subgroup metrics reuse the logits from the single selected-model evaluation.
"""

from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import random
import statistics

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn

from build_nf_unsw_graphs import EDGE_FEATURE_COLUMNS, TEST_PATH, TRAIN_PATH, VALIDATION_PATH
from hybrid_egraphsage_model import HybridEGraphSAGE
from run_multiseed_experiment import EvaluationTracker, file_hashes
from train_egraphsage import check_predictions, evaluate_model, graph_summary, load_graph


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "hybrid_multiseed"
SEEDS = [42, 7, 21, 84, 123]
NUMBER_OF_EPOCHS = 500
LEARNING_RATE = 0.01
HIDDEN_SIZE = 32
CLASSIFIER_HIDDEN_SIZE_1 = 32
CLASSIFIER_HIDDEN_SIZE_2 = 16
DEVICE = torch.device("cpu")
METRIC_NAMES = ["accuracy", "precision", "recall", "f1"]
GROUP_NAMES = ["both_endpoints_empty", "at_least_one_nonempty"]


def neighborhood_group_masks(graph) -> dict:
    """Use remaining incidence counts, not whether a feature mean is zero."""
    source_ids, destination_ids = graph.edge_index
    counts = torch.bincount(graph.edge_index.reshape(-1), minlength=graph.num_nodes)
    multiplicity = 1 + (source_ids == destination_ids).long()
    both_empty = (counts[source_ids] - multiplicity == 0) & (
        counts[destination_ids] - multiplicity == 0
    )
    return {GROUP_NAMES[0]: both_empty, GROUP_NAMES[1]: ~both_empty}


def subgroup_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    """Binary metrics with attack as positive; record class counts as context."""
    if len(labels) == 0:
        return {
            "edges": 0, "label_counts": [0, 0],
            **{name: None for name in METRIC_NAMES},
            "confusion_matrix": [[0, 0], [0, 0]],
        }
    return {
        "edges": len(labels),
        "label_counts": np.bincount(labels, minlength=2).tolist(),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def evaluate_test_with_subgroups(model, graph, loss_function) -> dict:
    """Capture the existing evaluator's logits instead of forwarding again."""
    captured_logits = []

    def capture_output(module, arguments, output):
        captured_logits.append(output.detach().cpu())

    hook = model.register_forward_hook(capture_output)
    try:
        metrics = evaluate_model(model, graph, loss_function)
    finally:
        hook.remove()
    if len(captured_logits) != 1:
        raise RuntimeError("Subgroup analysis requires exactly one test forward pass.")

    predictions = captured_logits[0].argmax(dim=1).numpy()
    labels = graph.edge_y.cpu().numpy()
    masks = neighborhood_group_masks(graph)
    metrics["subgroups"] = {}
    for name, mask in masks.items():
        selected = mask.cpu().numpy()
        metrics["subgroups"][name] = subgroup_metrics(labels[selected], predictions[selected])
    metrics["prediction_artifact"] = {
        "labels": labels.tolist(),
        "predictions": predictions.tolist(),
        "both_endpoints_empty": masks[GROUP_NAMES[0]].cpu().tolist(),
    }
    return metrics


def run_seed(seed: int, output_directory: Path) -> dict:
    output_directory.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    train_graph = load_graph(TRAIN_PATH)
    validation_graph = load_graph(VALIDATION_PATH)
    attack_fraction = (train_graph.edge_y == 1).float().mean().item()
    if not 0.45 <= attack_fraction <= 0.55:
        raise ValueError("Training labels are not approximately balanced.")
    configuration = {
        "node_feature_size": train_graph.x.shape[1],
        "edge_feature_size": len(EDGE_FEATURE_COLUMNS),
        "hidden_size": HIDDEN_SIZE,
        "classifier_hidden_size_1": CLASSIFIER_HIDDEN_SIZE_1,
        "classifier_hidden_size_2": CLASSIFIER_HIDDEN_SIZE_2,
        "number_of_classes": 2,
    }
    model = HybridEGraphSAGE(**configuration).to(DEVICE)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    tracker = EvaluationTracker(expected_epochs=NUMBER_OF_EPOCHS)
    checkpoint_path = output_directory / "checkpoint.pt"
    history = []

    for epoch in range(1, NUMBER_OF_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_graph)
        check_predictions(logits, train_graph)
        loss = loss_function(logits, train_graph.edge_y)
        if not torch.isfinite(loss).item():
            raise ValueError("Non-finite hybrid training loss.")
        loss.backward()
        optimizer.step()
        validation = tracker.evaluate(
            model, "validation", lambda: evaluate_model(model, validation_graph, loss_function)
        )
        history.append({
            "epoch": epoch, "training_loss": float(loss.item()),
            "validation_loss": validation["loss"], "validation_f1": validation["f1"],
        })
        if tracker.best_epoch == epoch:
            torch.save({
                "model_class": "HybridEGraphSAGE",
                "model_state_dict": model.state_dict(),
                "model_configuration": configuration,
                "best_epoch": tracker.best_epoch,
                "best_validation_f1": tracker.best_validation_f1,
                "seed": seed,
                "target_features_directly_available": True,
                "target_edge_features_in_aggregation": False,
            }, checkpoint_path)
        if epoch == 1 or epoch % 25 == 0 or epoch == NUMBER_OF_EPOCHS:
            print(
                f"Epoch {epoch:03d} | train loss={loss.item():.6f} | "
                f"val F1={validation['f1']:.6f} | best epoch={tracker.best_epoch}"
            )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Test information is unavailable until training and selection finish.
    test_graph = load_graph(TEST_PATH)
    test_metrics = tracker.evaluate(
        model, "test", lambda: evaluate_test_with_subgroups(model, test_graph, loss_function)
    )
    if tracker.test_evaluations != 1:
        raise RuntimeError("Expected exactly one hybrid test evaluation.")
    result = {
        "model": "hybrid_egraphsage", "seed": seed,
        "best_epoch": tracker.best_epoch,
        "best_validation_f1": tracker.best_validation_f1,
        **{f"test_{name}": float(test_metrics[name]) for name in METRIC_NAMES},
        "test_loss": test_metrics["loss"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "subgroups": test_metrics["subgroups"],
        "parameter_count": parameter_count,
        "model_configuration": configuration,
        "test_evaluations": tracker.test_evaluations,
        "test_forward_passes": 1,
        "restored_best_checkpoint_verified": True,
        "best_weights_sha256": tracker.best_weights_hash,
        "graphs": {
            name: graph_summary(graph)
            for name, graph in (("train", train_graph), ("validation", validation_graph), ("test", test_graph))
        },
        "evaluation_protocol": {
            "target_features_directly_available": True,
            "target_edge_features_in_aggregation": False,
            "target_specific_endpoint_embeddings": True,
            "labels_used_as_model_inputs": False,
            "other_same_split_edges_in_aggregation": True,
            "cross_split_aggregation": False,
            "subgroup_metrics_reuse_test_logits": True,
            "style": "batch/transductive-style, not real-time causal IDS",
        },
    }
    for filename, value in (
        ("run_summary.json", result),
        ("training_history.json", history),
        ("test_predictions.json", test_metrics["prediction_artifact"]),
    ):
        (output_directory / filename).write_text(
            json.dumps(value, indent=4) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=4))
    return result


def summarize(control_runs: list[dict], hybrid_runs: list[dict]) -> dict:
    models = {}
    for name in ("mlp", "egraphsage", "loeo_egraphsage", "hybrid_egraphsage"):
        runs = [run for run in [*control_runs, *hybrid_runs] if run["model"] == name]
        models[name] = {
            "runs": len(runs), "std_ddof": 1,
            **{
                f"{statistic}_test_{metric}": function(run[f"test_{metric}"] for run in runs)
                for metric in METRIC_NAMES
                for statistic, function in (("mean", statistics.mean), ("std", statistics.stdev))
            },
        }
    subgroups = {}
    for name in GROUP_NAMES:
        groups = [run["subgroups"][name] for run in hybrid_runs]
        if len({group["edges"] for group in groups}) != 1:
            raise ValueError("Subgroup sizes changed across seeds.")
        subgroups[name] = {
            "edges": groups[0]["edges"], "label_counts": groups[0]["label_counts"],
            "std_ddof": 1,
            **{
                f"{statistic}_{metric}": function(group[metric] for group in groups)
                for metric in METRIC_NAMES
                for statistic, function in (("mean", statistics.mean), ("std", statistics.stdev))
            },
        }
    paired = []
    for seed in SEEDS:
        hybrid = next(run for run in hybrid_runs if run["seed"] == seed)
        mlp = next(run for run in control_runs if run["model"] == "mlp" and run["seed"] == seed)
        gnn = next(run for run in control_runs if run["model"] == "egraphsage" and run["seed"] == seed)
        paired.append({
            "seed": seed,
            "hybrid_minus_mlp": hybrid["test_f1"] - mlp["test_f1"],
            "hybrid_minus_original_gnn": hybrid["test_f1"] - gnn["test_f1"],
        })
    return {
        "models": models, "hybrid_subgroups": subgroups,
        "mean_hybrid_minus_mlp": models["hybrid_egraphsage"]["mean_test_f1"] - models["mlp"]["mean_test_f1"],
        "mean_hybrid_minus_original_gnn": models["hybrid_egraphsage"]["mean_test_f1"] - models["egraphsage"]["mean_test_f1"],
        "paired_f1_differences": paired,
        "hybrid_wins_over_mlp": sum(pair["hybrid_minus_mlp"] > 0 for pair in paired),
        "hybrid_wins_over_original_gnn": sum(pair["hybrid_minus_original_gnn"] > 0 for pair in paired),
    }


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {OUTPUT_ROOT}")
    controls = json.loads((PROJECT_ROOT / "results/multiseed/summary.json").read_text())
    loeo = json.loads((PROJECT_ROOT / "results/loeo_multiseed/summary.json").read_text())
    if controls["seeds"] != SEEDS or loeo["seeds"] != SEEDS:
        raise ValueError("Historical seeds do not match the hybrid seeds.")
    protected_paths = list((PROJECT_ROOT / "src").glob("*.py"))
    protected_paths += list((PROJECT_ROOT / "tests").glob("*.py"))
    protected_paths += list((PROJECT_ROOT / "examples").rglob("*.py"))
    protected_paths += [path for path in (PROJECT_ROOT / "results").rglob("*") if path.is_file()]
    protected_paths += [path for path in (PROJECT_ROOT / "data/processed").rglob("*") if path.is_file()]
    protected_hashes = file_hashes(protected_paths)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    protocol = {
        "seeds": SEEDS, "epochs": NUMBER_OF_EPOCHS,
        "optimizer": "Adam", "learning_rate": LEARNING_RATE,
        "loss": "CrossEntropyLoss", "batching": "full graph",
        "device": str(DEVICE), "cpu_threads": 1, "deterministic_algorithms": True,
        "torch_version": str(torch.__version__),
        "architecture": {"node_layer": [16, 32], "edge_classifier": [72, 32, 16, 2]},
        "classifier_activation": "ReLU after each hidden layer",
        "aggregation_implementation": "loeo_egraphsage_model.aggregate_without_target (unchanged)",
        "selection_rule": "maximum validation F1; earliest tied epoch",
        "test_rule": "once after restoring selected weights; subgroup metrics reuse those logits",
        "fixed_inputs_and_historical_files_sha256": protected_hashes,
    }
    (OUTPUT_ROOT / "protocol.json").write_text(
        json.dumps(protocol, indent=4) + "\n", encoding="utf-8"
    )
    results = []
    for seed in SEEDS:
        print(f"Starting hybrid seed {seed}", flush=True)
        with (OUTPUT_ROOT / f"seed_{seed}.log").open("x", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                result = run_seed(seed, OUTPUT_ROOT / f"seed_{seed}")
        results.append(result)
        if file_hashes(protected_paths) != protected_hashes:
            raise RuntimeError("A protected source, dataset, or historical result changed.")
        print(
            f"Finished hybrid seed {seed}: best epoch={result['best_epoch']}, "
            f"validation F1={result['best_validation_f1']:.6f}, test F1={result['test_f1']:.6f}",
            flush=True,
        )
    comparison = summarize([*controls["runs"], *loeo["runs"]], results)
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps({"seeds": SEEDS, "runs": results, "comparison": comparison}, indent=4) + "\n",
        encoding="utf-8",
    )
    print("\n" + json.dumps(comparison, indent=4), flush=True)
    print(f"\nSaved separate results to {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
