"""Train the first batch E-GraphSAGE-style NF-UNSW-NB15 experiment.

The existing clean splits and training-only normalization are read unchanged.
Each graph aggregates its own unlabeled edge features, including target-edge
features. Validation/test edges can therefore use context from other flows in
the same split. This is batch/transductive-style evaluation, not a causal,
real-time IDS experiment. Graphs are never combined across splits.

Only training labels contribute to gradients. Validation F1 selects the
checkpoint. The test graph is loaded after selection and evaluated once.
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch_geometric.data import Data

from build_nf_unsw_graphs import (
    EDGE_FEATURE_COLUMNS,
    TEST_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
    dataframe_to_graph,
)
from egraphsage_model import EGraphSAGE


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
MODEL_PATH = RESULTS_DIRECTORY / "egraphsage.pt"
METRICS_PATH = RESULTS_DIRECTORY / "egraphsage_metrics.json"
HISTORY_PATH = RESULTS_DIRECTORY / "egraphsage_training_history.json"

HIDDEN_SIZE = 32
NUMBER_OF_CLASSES = 2
NUMBER_OF_EPOCHS = 500
LEARNING_RATE = 0.01
RANDOM_SEED = 42
MLP_BASELINE_F1 = 0.9767
DEVICE = torch.device("cpu")


def load_graph(path: Path) -> Data:
    """Reuse the verified graph builder; do not refit preprocessing here."""
    return dataframe_to_graph(pd.read_csv(path)).to(DEVICE)


def check_predictions(logits: torch.Tensor, graph: Data) -> None:
    """Fail immediately if edge predictions or labels are inconsistent."""
    if logits.shape != (graph.num_edges, NUMBER_OF_CLASSES):
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if graph.edge_y.shape != (graph.num_edges,):
        raise ValueError(f"Unexpected label shape: {tuple(graph.edge_y.shape)}")
    if not torch.isfinite(logits).all().item():
        raise ValueError("NaN or infinite logits encountered.")


def evaluate_model(
    model: EGraphSAGE, graph: Data, loss_function: nn.Module
) -> dict:
    """Evaluate one graph with a single forward pass and no gradients."""
    model.eval()
    with torch.no_grad():
        logits = model(graph)
        check_predictions(logits, graph)
        loss = loss_function(logits, graph.edge_y)
        if not torch.isfinite(loss).item():
            raise ValueError("NaN or infinite evaluation loss encountered.")
        predictions = logits.argmax(dim=1).cpu().numpy()
        labels = graph.edge_y.cpu().numpy()

    # Attack (label 1) is the positive class, matching the MLP baseline.
    return {
        "loss": float(loss.item()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def graph_summary(graph: Data) -> dict:
    """Record the exact graphs used for this run."""
    return {
        "nodes": graph.num_nodes,
        "edges": graph.num_edges,
        "edge_index_shape": list(graph.edge_index.shape),
        "edge_attr_shape": list(graph.edge_attr.shape),
        "edge_y_shape": list(graph.edge_y.shape),
        "label_counts": torch.bincount(graph.edge_y, minlength=2).tolist(),
    }


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    # CPU accumulation is deterministic; one thread also avoids excessive
    # threading overhead for this small full-graph experiment.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Deliberately do not load the test graph before checkpoint selection.
    train_graph = load_graph(TRAIN_PATH)
    validation_graph = load_graph(VALIDATION_PATH)
    label_counts = torch.bincount(train_graph.edge_y, minlength=2)
    attack_fraction = label_counts[1].item() / train_graph.num_edges
    if not 0.45 <= attack_fraction <= 0.55:
        raise ValueError(f"Training labels are not approximately balanced: {label_counts}")
    print(f"Device: {DEVICE}")
    print(f"Training label counts [benign, attack]: {label_counts.tolist()}")
    print("Batch aggregation includes the target edge and other same-split flows.")

    model_configuration = {
        "node_feature_size": train_graph.x.shape[1],
        "edge_feature_size": len(EDGE_FEATURE_COLUMNS),
        "hidden_size": HIDDEN_SIZE,
        "number_of_classes": NUMBER_OF_CLASSES,
    }
    model = EGraphSAGE(**model_configuration).to(DEVICE)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {parameter_count:,}")
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_validation_f1 = -1.0
    best_epoch = 0
    history = []

    # One optimizer step uses every training edge. Validation is a separate
    # forward pass with no gradients and never enters the optimizer loss.
    for epoch in range(1, NUMBER_OF_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_graph)
        check_predictions(logits, train_graph)
        training_loss = loss_function(logits, train_graph.edge_y)
        if not torch.isfinite(training_loss).item():
            raise ValueError("NaN or infinite training loss encountered.")
        training_loss.backward()
        optimizer.step()

        validation_metrics = evaluate_model(model, validation_graph, loss_function)
        history.append({
            "epoch": epoch,
            "training_loss": float(training_loss.item()),
            "validation_loss": validation_metrics["loss"],
            "validation_f1": validation_metrics["f1"],
        })
        # Strict improvement keeps the earliest checkpoint when F1 ties.
        if validation_metrics["f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["f1"]
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_configuration": model_configuration,
                "best_epoch": best_epoch,
                "best_validation_f1": best_validation_f1,
            }, MODEL_PATH)

        if epoch == 1 or epoch % 25 == 0 or epoch == NUMBER_OF_EPOCHS:
            print(
                f"Epoch {epoch:03d} | train loss={training_loss.item():.4f} | "
                f"val loss={validation_metrics['loss']:.4f} | "
                f"val F1={validation_metrics['f1']:.4f} | "
                f"best epoch={best_epoch}"
            )

    # Selection is now finished. Restore that checkpoint before the test
    # graph (including its labels) is even loaded.
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_graph = load_graph(TEST_PATH)
    test_metrics = evaluate_model(model, test_graph, loss_function)
    f1_difference = test_metrics["f1"] - MLP_BASELINE_F1

    report = {
        **test_metrics,
        "best_epoch": checkpoint["best_epoch"],
        "best_validation_f1": checkpoint["best_validation_f1"],
        "parameter_count": parameter_count,
        "mlp_baseline_f1": MLP_BASELINE_F1,
        "f1_difference_vs_mlp": f1_difference,
        "configuration": {
            **model_configuration,
            "epochs": NUMBER_OF_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seed": RANDOM_SEED,
            "device": str(DEVICE),
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "batching": "one full training graph per optimizer step",
            "cpu_threads": torch.get_num_threads(),
            "deterministic_algorithms": True,
            "edge_feature_columns": EDGE_FEATURE_COLUMNS,
            "torch_version": str(torch.__version__),
        },
        "graphs": {
            "train": graph_summary(train_graph),
            "validation": graph_summary(validation_graph),
            "test": graph_summary(test_graph),
        },
        # Hashes document which existing inputs were used; they do not
        # influence training, model selection, or feature preprocessing.
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (TRAIN_PATH, VALIDATION_PATH, TEST_PATH)
        },
        "evaluation_protocol": {
            "selection_metric": "validation F1",
            "test_evaluations": 1,
            "labels_used_as_model_inputs": False,
            "target_edge_features_in_aggregation": True,
            "other_same_split_edges_in_aggregation": True,
            "cross_split_aggregation": False,
            "style": "batch/transductive-style, not real-time causal IDS",
        },
    }
    METRICS_PATH.write_text(json.dumps(report, indent=4) + "\n", encoding="utf-8")
    HISTORY_PATH.write_text(json.dumps(history, indent=4) + "\n", encoding="utf-8")

    print(f"\nBest epoch: {checkpoint['best_epoch']}")
    print(f"Best validation F1: {checkpoint['best_validation_f1']:.6f}")
    print("\nTest results (best-validation checkpoint, evaluated once)")
    for metric in ("accuracy", "precision", "recall", "f1"):
        print(f"{metric.capitalize()}: {test_metrics[metric]:.6f}")
    print(f"Confusion matrix: {test_metrics['confusion_matrix']}")
    print(f"\nMLP baseline: F1 = {MLP_BASELINE_F1:.4f}")
    print(f"E-GraphSAGE: F1 = {test_metrics['f1']:.6f}")
    print(f"Difference (E-GraphSAGE - MLP): {f1_difference:+.6f}")
    if f1_difference > 0:
        print("This run exceeds the MLP F1; an ablation is needed to isolate graph benefit.")
    else:
        print("This run does not show an F1 improvement over the MLP baseline.")
    print(f"\nSaved checkpoint: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved training history: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
