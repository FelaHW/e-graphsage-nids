"""Run the fixed-recipe shuffled-context control with historical files intact.

Reuse the real hybrid trainer itself. Temporary runtime settings substitute
only the context-permuting subclass and a graph loader that attaches fixed
donor indices. Checkpoint selection and one-pass subgroup evaluation are the
same code paths as the real-hybrid experiment.
"""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
from pathlib import Path
import statistics

import torch

import run_hybrid_experiment as real_hybrid
from run_multiseed_experiment import file_hashes, temporary_settings
from shuffled_hybrid_model import (
    ShuffledHybridEGraphSAGE,
    make_context_derangement,
    permutation_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "shuffled_hybrid_multiseed"
SEEDS = [42, 7, 21, 84, 123]
METRIC_NAMES = real_hybrid.METRIC_NAMES
GROUP_NAMES = real_hybrid.GROUP_NAMES


def permutation_diagnostics(graph, experiment_seed: int, split: str) -> dict:
    """Describe assignment changes using topology only, without labels."""
    permutation = graph.context_permutation
    original_empty = real_hybrid.neighborhood_group_masks(graph)[GROUP_NAMES[0]]
    assigned_empty = original_empty[permutation]
    source_ids, destination_ids = graph.edge_index
    donor_sources = source_ids[permutation]
    donor_destinations = destination_ids[permutation]
    shares_endpoint = (
        (source_ids == donor_sources) | (source_ids == donor_destinations)
        | (destination_ids == donor_sources) | (destination_ids == donor_destinations)
    )
    return {
        "edges": graph.num_edges,
        "permutation_seed": permutation_seed(experiment_seed, split),
        "fixed_points": int((permutation == torch.arange(graph.num_edges, device=permutation.device)).sum().item()),
        "permutation_sha256": hashlib.sha256(permutation.cpu().numpy().tobytes()).hexdigest(),
        "original_both_empty": int(original_empty.sum().item()),
        "assigned_both_empty": int(assigned_empty.sum().item()),
        "availability_transitions": {
            "empty_to_empty": int((original_empty & assigned_empty).sum().item()),
            "empty_to_nonempty": int((original_empty & ~assigned_empty).sum().item()),
            "nonempty_to_empty": int((~original_empty & assigned_empty).sum().item()),
            "nonempty_to_nonempty": int((~original_empty & ~assigned_empty).sum().item()),
        },
        # Derangement excludes donor==recipient, but the recipient flow can
        # still occur inside the donor's valid LOEO neighborhood by chance.
        "recipient_shares_endpoint_with_donor": int(shares_endpoint.sum().item()),
    }


def run_seed(seed: int, output_directory: Path) -> dict:
    original_load = real_hybrid.load_graph
    split_names = {
        real_hybrid.TRAIN_PATH: "train",
        real_hybrid.VALIDATION_PATH: "validation",
        real_hybrid.TEST_PATH: "test",
    }
    loaded_graphs = {}

    def load_with_permutation(path):
        graph = original_load(path)
        split = split_names[path]
        graph.context_permutation = make_context_derangement(
            graph.num_edges, seed, split
        ).to(graph.edge_index.device)
        loaded_graphs[split] = graph
        return graph

    # This preserves the real trainer's exact RNG initialization, optimizer,
    # schedule, selection, restoration, and single test forward behavior.
    with temporary_settings(real_hybrid, {
        "HybridEGraphSAGE": ShuffledHybridEGraphSAGE,
        "load_graph": load_with_permutation,
    }):
        result = real_hybrid.run_seed(seed, output_directory)

    diagnostics = {
        split: permutation_diagnostics(graph, seed, split)
        for split, graph in loaded_graphs.items()
    }
    if any(info["fixed_points"] != 0 for info in diagnostics.values()):
        raise RuntimeError("A context permutation has a fixed point.")
    result["model"] = "shuffled_hybrid_egraphsage"
    result["permutations"] = diagnostics
    # The donor context excludes its own target. After reassignment, that
    # statement does not imply exclusion of the recipient from donor neighbors.
    result["evaluation_protocol"].pop("target_edge_features_in_aggregation", None)
    result["evaluation_protocol"].update({
        "context_source_and_destination_shuffled_together": True,
        "direct_target_features_shuffled": False,
        "labels_shuffled": False,
        "donor_context_excludes_its_original_target": True,
        "recipient_explicitly_removed_from_donor_context": False,
        "fixed_context_assignment_across_epochs": True,
        "subgroups_use_original_recipient_context_availability": True,
    })
    result["fixed_points"] = {split: info["fixed_points"] for split, info in diagnostics.items()}

    # These are freshly created control files only; historical outputs are
    # never opened for writing. Tag the unchanged weights with the right class.
    checkpoint_path = output_directory / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["model_class"] = "ShuffledHybridEGraphSAGE"
    checkpoint.pop("target_edge_features_in_aggregation", None)
    checkpoint["donor_context_excludes_its_original_target"] = True
    checkpoint["permutation_seeds"] = {
        split: info["permutation_seed"] for split, info in diagnostics.items()
    }
    checkpoint["context_permutation_file"] = "permutations.json"
    torch.save(checkpoint, checkpoint_path)

    permutations = {
        split: graph.context_permutation.cpu().tolist()
        for split, graph in loaded_graphs.items()
    }
    (output_directory / "permutations.json").write_text(
        json.dumps(permutations, indent=4) + "\n", encoding="utf-8"
    )
    prediction_path = output_directory / "test_predictions.json"
    predictions = json.loads(prediction_path.read_text())
    original_empty = torch.tensor(predictions["both_endpoints_empty"], dtype=torch.bool)
    predictions["assigned_context_both_endpoints_empty"] = original_empty[
        loaded_graphs["test"].context_permutation.cpu()
    ].tolist()
    prediction_path.write_text(json.dumps(predictions, indent=4) + "\n", encoding="utf-8")
    (output_directory / "run_summary.json").write_text(
        json.dumps(result, indent=4) + "\n", encoding="utf-8"
    )
    print(f"Final control variant: {result['model']}; fixed points: {result['fixed_points']}")
    return result


def compare_saved_predictions(shuffled_runs: list[dict]) -> list[dict]:
    """Compare paired subgroup errors without evaluating historical models."""
    comparisons = []
    for run in shuffled_runs:
        seed = run["seed"]
        real = json.loads((PROJECT_ROOT / f"results/hybrid_multiseed/seed_{seed}/test_predictions.json").read_text())
        shuffled = json.loads((OUTPUT_ROOT / f"seed_{seed}/test_predictions.json").read_text())
        if real["labels"] != shuffled["labels"] or real["both_endpoints_empty"] != shuffled["both_endpoints_empty"]:
            raise ValueError("Saved real/shuffled predictions do not describe the same test rows.")
        groups = {}
        for name, want_empty in zip(GROUP_NAMES, (True, False)):
            rows = [
                (label, real_prediction, shuffled_prediction)
                for label, real_prediction, shuffled_prediction, empty in zip(
                    real["labels"], real["predictions"], shuffled["predictions"], real["both_endpoints_empty"]
                ) if empty == want_empty
            ]
            groups[name] = {
                "edges": len(rows),
                "prediction_disagreements": sum(a != b for label, a, b in rows),
                "real_correct_shuffled_wrong": sum(a == label and b != label for label, a, b in rows),
                "shuffled_correct_real_wrong": sum(b == label and a != label for label, a, b in rows),
                "both_correct": sum(a == label and b == label for label, a, b in rows),
                "both_wrong": sum(a != label and b != label for label, a, b in rows),
            }
        comparisons.append({"seed": seed, "groups": groups})
    return comparisons


def summarize(shuffled_runs: list[dict], real_report: dict, mlp_runs: list[dict]) -> dict:
    # The four existing model statistics are reused, not recomputed by model
    # evaluation. Only the new control is trained in this experiment.
    models = dict(real_report["comparison"]["models"])
    models["shuffled_hybrid_egraphsage"] = {
        "runs": len(shuffled_runs), "std_ddof": 1,
        **{
            f"{statistic}_test_{metric}": function(run[f"test_{metric}"] for run in shuffled_runs)
            for metric in METRIC_NAMES
            for statistic, function in (("mean", statistics.mean), ("std", statistics.stdev))
        },
    }
    pairs = []
    for shuffled in shuffled_runs:
        seed = shuffled["seed"]
        real = next(run for run in real_report["runs"] if run["seed"] == seed)
        mlp = next(run for run in mlp_runs if run["seed"] == seed)
        pairs.append({
            "seed": seed,
            "real_minus_shuffled": real["test_f1"] - shuffled["test_f1"],
            "shuffled_minus_mlp": shuffled["test_f1"] - mlp["test_f1"],
            "real_minus_mlp": real["test_f1"] - mlp["test_f1"],
        })
    subgroup_comparison = {}
    for name in GROUP_NAMES:
        groups = [run["subgroups"][name] for run in shuffled_runs]
        real_group = real_report["comparison"]["hybrid_subgroups"][name]
        shuffled_group = {
            "edges": groups[0]["edges"], "label_counts": groups[0]["label_counts"],
            **{
                f"{statistic}_{metric}": function(group[metric] for group in groups)
                for metric in METRIC_NAMES
                for statistic, function in (("mean", statistics.mean), ("std", statistics.stdev))
            },
        }
        subgroup_comparison[name] = {
            "real": real_group, "shuffled": shuffled_group,
            "mean_real_minus_shuffled_f1": real_group["mean_f1"] - shuffled_group["mean_f1"],
        }
    return {
        "models": models,
        "mean_real_minus_shuffled_f1": models["hybrid_egraphsage"]["mean_test_f1"] - models["shuffled_hybrid_egraphsage"]["mean_test_f1"],
        "mean_shuffled_minus_mlp_f1": models["shuffled_hybrid_egraphsage"]["mean_test_f1"] - models["mlp"]["mean_test_f1"],
        "mean_real_minus_mlp_f1": models["hybrid_egraphsage"]["mean_test_f1"] - models["mlp"]["mean_test_f1"],
        "paired_f1_differences": pairs,
        "std_real_minus_shuffled_f1": statistics.stdev(pair["real_minus_shuffled"] for pair in pairs),
        "real_wins_over_shuffled": sum(pair["real_minus_shuffled"] > 0 for pair in pairs),
        "shuffled_wins_over_real": sum(pair["real_minus_shuffled"] < 0 for pair in pairs),
        "ties": sum(pair["real_minus_shuffled"] == 0 for pair in pairs),
        "subgroup_comparison_by_original_recipient_availability": subgroup_comparison,
    }


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_ROOT}")
    real_report = json.loads((PROJECT_ROOT / "results/hybrid_multiseed/summary.json").read_text())
    controls = json.loads((PROJECT_ROOT / "results/multiseed/summary.json").read_text())
    if real_report["seeds"] != SEEDS or controls["seeds"] != SEEDS:
        raise ValueError("Historical seeds do not match shuffled-control seeds.")
    protected_paths = list((PROJECT_ROOT / "src").glob("*.py"))
    protected_paths += list((PROJECT_ROOT / "tests").glob("*.py"))
    protected_paths += [path for path in (PROJECT_ROOT / "results").rglob("*") if path.is_file()]
    protected_paths += [path for path in (PROJECT_ROOT / "data/processed").rglob("*") if path.is_file()]
    protected_hashes = file_hashes(protected_paths)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    protocol = {
        "seeds": SEEDS,
        "architecture": "inherit HybridEGraphSAGE unchanged; 3442 parameters",
        "epochs": real_hybrid.NUMBER_OF_EPOCHS,
        "optimizer": "Adam", "learning_rate": real_hybrid.LEARNING_RATE,
        "loss": "CrossEntropyLoss", "batching": "full graph", "device": "cpu",
        "cpu_threads": 1, "deterministic_algorithms": True,
        "torch_version": str(torch.__version__),
        "training_implementation": "run_hybrid_experiment.run_seed, reused unchanged",
        "permutation_seed_derivation": "UTF-8 shuffled-context-v1|{experiment_seed}|{split}; SHA256; first 8 bytes big-endian modulo 2^63",
        "split_identities": ["train", "validation", "test"],
        "derangement": "private CPU torch.Generator; reject torch.randperm draws with any fixed points",
        "assignment": "fixed per split/seed; source and destination contexts move together",
        "model_global_rng_advanced_by_permutation": False,
        "selection_rule": "maximum validation F1; earliest tied epoch",
        "test_rule": "one evaluation/forward after restoration; reuse predictions for subgroup analysis",
        "fixed_inputs_and_historical_files_sha256": protected_hashes,
    }
    (OUTPUT_ROOT / "protocol.json").write_text(
        json.dumps(protocol, indent=4) + "\n", encoding="utf-8"
    )
    results = []
    for seed in SEEDS:
        print(f"Starting shuffled hybrid seed {seed}", flush=True)
        with (OUTPUT_ROOT / f"seed_{seed}.log").open("x", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                print("Shuffled paired-context control; real-hybrid trainer reused unchanged.")
                result = run_seed(seed, OUTPUT_ROOT / f"seed_{seed}")
        results.append(result)
        if file_hashes(protected_paths) != protected_hashes:
            raise RuntimeError("A protected source, input, checkpoint, or historical result changed.")
        print(
            f"Finished shuffled seed {seed}: best epoch={result['best_epoch']}, "
            f"validation F1={result['best_validation_f1']:.6f}, test F1={result['test_f1']:.6f}, "
            f"fixed points={result['fixed_points']}", flush=True,
        )
    comparison = summarize(
        results, real_report, [run for run in controls["runs"] if run["model"] == "mlp"]
    )
    comparison["saved_prediction_comparison_by_original_recipient_availability"] = compare_saved_predictions(results)
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps({"seeds": SEEDS, "runs": results, "comparison": comparison}, indent=4) + "\n",
        encoding="utf-8",
    )
    print("\n" + json.dumps(comparison, indent=4), flush=True)
    print(f"\nSaved separate results to {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
