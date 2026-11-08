"""Run the five-seed LOEO ablation without replacing control artifacts.

Reuse graph loading, evaluation, safety checks, and the original GNN recipe.
The only model change is target exclusion before its endpoint embeddings.
Test graphs load after validation selection and are evaluated once per seed.
"""

from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import random
import statistics

import numpy as np
import torch
from torch import nn

from build_nf_unsw_graphs import EDGE_FEATURE_COLUMNS, TEST_PATH, TRAIN_PATH, VALIDATION_PATH
from loeo_egraphsage_model import LOEOEGraphSAGE
from run_multiseed_experiment import EvaluationTracker, file_hashes
from train_egraphsage import (
    DEVICE,
    HIDDEN_SIZE,
    LEARNING_RATE,
    NUMBER_OF_CLASSES,
    NUMBER_OF_EPOCHS,
    check_predictions,
    evaluate_model,
    graph_summary,
    load_graph,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "loeo_multiseed"
CONTROL_SUMMARY_PATH = PROJECT_ROOT / "results" / "multiseed" / "summary.json"
SEEDS = [42, 7, 21, 84, 123]
METRIC_NAMES = ["accuracy", "precision", "recall", "f1"]


def context_coverage(graph) -> dict:
    """Describe empty neighborhoods without another prediction or label use."""
    source_ids, destination_ids = graph.edge_index
    node_counts = torch.bincount(graph.edge_index.reshape(-1), minlength=graph.num_nodes)
    multiplicity = 1 + (source_ids == destination_ids).long()
    source_empty = node_counts[source_ids] - multiplicity == 0
    destination_empty = node_counts[destination_ids] - multiplicity == 0
    return {
        "edges": graph.num_edges,
        "source_empty": int(source_empty.sum().item()),
        "destination_empty": int(destination_empty.sum().item()),
        "either_endpoint_empty": int((source_empty | destination_empty).sum().item()),
        "both_endpoints_empty": int((source_empty & destination_empty).sum().item()),
        "fraction_both_endpoints_empty": float((source_empty & destination_empty).float().mean().item()),
    }


def run_seed(seed: int, output_directory: Path) -> dict:
    output_directory.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    train_graph = load_graph(TRAIN_PATH)
    validation_graph = load_graph(VALIDATION_PATH)
    label_counts = torch.bincount(train_graph.edge_y, minlength=2)
    if not 0.45 <= label_counts[1].item() / train_graph.num_edges <= 0.55:
        raise ValueError("Training labels are not approximately balanced.")

    configuration = {
        "node_feature_size": train_graph.x.shape[1],
        "edge_feature_size": len(EDGE_FEATURE_COLUMNS),
        "hidden_size": HIDDEN_SIZE,
        "number_of_classes": NUMBER_OF_CLASSES,
    }
    model = LOEOEGraphSAGE(**configuration).to(DEVICE)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
            raise ValueError("Non-finite LOEO training loss.")
        loss.backward()
        optimizer.step()

        validation_metrics = tracker.evaluate(
            model, "validation", lambda: evaluate_model(model, validation_graph, loss_function)
        )
        history.append({
            "epoch": epoch,
            "training_loss": float(loss.item()),
            "validation_loss": validation_metrics["loss"],
            "validation_f1": validation_metrics["f1"],
        })
        # The shared tracker uses strict improvement and keeps the earliest
        # tied epoch, exactly as the target-inclusive control did.
        if tracker.best_epoch == epoch:
            torch.save({
                "model_class": "LOEOEGraphSAGE",
                "model_state_dict": model.state_dict(),
                "model_configuration": configuration,
                "target_edge_features_in_aggregation": False,
                "best_epoch": tracker.best_epoch,
                "best_validation_f1": tracker.best_validation_f1,
                "seed": seed,
            }, checkpoint_path)
        if epoch == 1 or epoch % 25 == 0 or epoch == NUMBER_OF_EPOCHS:
            print(
                f"Epoch {epoch:03d} | train loss={loss.item():.6f} | "
                f"val F1={validation_metrics['f1']:.6f} | best epoch={tracker.best_epoch}"
            )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    # No test graph or test labels have been loaded before this point.
    test_graph = load_graph(TEST_PATH)
    test_metrics = tracker.evaluate(
        model, "test", lambda: evaluate_model(model, test_graph, loss_function)
    )
    if tracker.test_evaluations != 1:
        raise RuntimeError("Expected exactly one LOEO test evaluation.")
    result = {
        "model": "loeo_egraphsage",
        "seed": seed,
        "best_epoch": tracker.best_epoch,
        "best_validation_f1": tracker.best_validation_f1,
        **{f"test_{name}": float(test_metrics[name]) for name in METRIC_NAMES},
        "test_loss": test_metrics["loss"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "parameter_count": parameter_count,
        "test_evaluations": tracker.test_evaluations,
        "restored_best_checkpoint_verified": True,
        "best_weights_sha256": tracker.best_weights_hash,
        "model_configuration": configuration,
        "graphs": {
            name: graph_summary(graph)
            for name, graph in (("train", train_graph), ("validation", validation_graph), ("test", test_graph))
        },
        "context_coverage": {
            name: context_coverage(graph)
            for name, graph in (("train", train_graph), ("validation", validation_graph), ("test", test_graph))
        },
        "evaluation_protocol": {
            "target_edge_features_in_aggregation": False,
            "target_specific_endpoint_embeddings": True,
            "other_same_split_edges_in_aggregation": True,
            "cross_split_aggregation": False,
            "labels_used_as_model_inputs": False,
            "style": "batch/transductive-style, not real-time causal IDS",
        },
    }
    (output_directory / "run_summary.json").write_text(
        json.dumps(result, indent=4) + "\n", encoding="utf-8"
    )
    (output_directory / "training_history.json").write_text(
        json.dumps(history, indent=4) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=4))
    return result


def summarize_three_models(control_runs: list[dict], loeo_runs: list[dict]) -> dict:
    statistics_by_model = {}
    for model_name in ("mlp", "egraphsage", "loeo_egraphsage"):
        runs = [run for run in [*control_runs, *loeo_runs] if run["model"] == model_name]
        statistics_by_model[model_name] = {
            "runs": len(runs),
            "std_ddof": 1,
            **{
                f"{statistic}_test_{metric}": function(run[f"test_{metric}"] for run in runs)
                for metric in METRIC_NAMES
                for statistic, function in (("mean", statistics.mean), ("std", statistics.stdev))
            },
        }
    differences = []
    for seed in SEEDS:
        mlp_run = next(run for run in control_runs if run["model"] == "mlp" and run["seed"] == seed)
        gnn_run = next(run for run in control_runs if run["model"] == "egraphsage" and run["seed"] == seed)
        loeo_run = next(run for run in loeo_runs if run["seed"] == seed)
        differences.append({
            "seed": seed,
            "loeo_minus_mlp": loeo_run["test_f1"] - mlp_run["test_f1"],
            "loeo_minus_original_gnn": loeo_run["test_f1"] - gnn_run["test_f1"],
        })
    return {
        "models": statistics_by_model,
        "mean_loeo_minus_mlp": statistics_by_model["loeo_egraphsage"]["mean_test_f1"] - statistics_by_model["mlp"]["mean_test_f1"],
        "mean_loeo_minus_original_gnn": statistics_by_model["loeo_egraphsage"]["mean_test_f1"] - statistics_by_model["egraphsage"]["mean_test_f1"],
        "paired_f1_differences": differences,
        "loeo_wins_over_mlp": sum(pair["loeo_minus_mlp"] > 0 for pair in differences),
        "loeo_wins_over_original_gnn": sum(pair["loeo_minus_original_gnn"] > 0 for pair in differences),
    }


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {OUTPUT_ROOT}")
    controls = json.loads(CONTROL_SUMMARY_PATH.read_text())
    if controls["seeds"] != SEEDS:
        raise ValueError("Historical comparison seeds do not match the ablation seeds.")

    # Historical results, source files, splits, and normalization must retain
    # their existing bytes. Only the new LOEO output tree may be written.
    protected_paths = list((PROJECT_ROOT / "src").glob("*.py"))
    protected_paths += list((PROJECT_ROOT / "tests").glob("*.py"))
    protected_paths += list((PROJECT_ROOT / "examples").rglob("*.py"))
    protected_paths += [path for path in (PROJECT_ROOT / "results").rglob("*") if path.is_file()]
    protected_paths += [path for path in (PROJECT_ROOT / "data/processed").rglob("*") if path.is_file()]
    protected_hashes = file_hashes(protected_paths)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    protocol = {
        "seeds": SEEDS,
        "epochs": NUMBER_OF_EPOCHS,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "CrossEntropyLoss",
        "batching": "one full training graph per optimizer step",
        "device": str(DEVICE),
        "cpu_threads": 1,
        "deterministic_algorithms": True,
        "torch_version": str(torch.__version__),
        "selection_rule": "maximum validation F1; earliest tied epoch",
        "test_rule": "load after selection; evaluate once after restoring best weights",
        "aggregation": "target-specific incident means, target excluded at both endpoints",
        "self_loops": "remove both source and destination incidences",
        "accumulation_dtype": "float64 sums/subtraction; means cast back to float32",
        "fixed_inputs_and_historical_files_sha256": protected_hashes,
    }
    (OUTPUT_ROOT / "protocol.json").write_text(
        json.dumps(protocol, indent=4) + "\n", encoding="utf-8"
    )
    results = []
    for seed in SEEDS:
        print(f"Starting LOEO seed {seed}", flush=True)
        output_directory = OUTPUT_ROOT / f"seed_{seed}"
        # run_seed creates the directory, so keep its log at the new root.
        with (OUTPUT_ROOT / f"seed_{seed}.log").open("x", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                result = run_seed(seed, output_directory)
        results.append(result)
        if file_hashes(protected_paths) != protected_hashes:
            raise RuntimeError("A protected dataset, source, or historical result changed.")
        print(
            f"Finished LOEO seed {seed}: best epoch={result['best_epoch']}, "
            f"validation F1={result['best_validation_f1']:.6f}, test F1={result['test_f1']:.6f}",
            flush=True,
        )
    comparison = summarize_three_models(controls["runs"], results)
    report = {"seeds": SEEDS, "runs": results, "comparison": comparison}
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(report, indent=4) + "\n", encoding="utf-8"
    )
    print("\n" + json.dumps(comparison, indent=4), flush=True)
    print(f"\nSaved separate results to {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
