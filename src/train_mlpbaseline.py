"""Train a simple MLP baseline on NF-UNSW-NB15.

The MLP does NOT use graph structure.

It sees only the 8 features belonging to each individual flow.

This gives us a baseline that we can later compare against
our E-GraphSAGE model.
"""

import json
from pathlib import Path
import torch
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT   = Path(__file__).resolve().parent.parent

DATA_DIRECTORY = (
    PROJECT_ROOT 
    / "data" 
    / "processed"
    / "nf_unsw_nb15_splits"
)
TRAIN_PATH = DATA_DIRECTORY / "train_normalized.csv"
VALIDATION_PATH = DATA_DIRECTORY / "validation_normalized.csv"
TEST_PATH = DATA_DIRECTORY / "test_normalized.csv"

RESULTS_DIRECTORY = PROJECT_ROOT / "results" 

MODEL_PATH = RESULTS_DIRECTORY / "mlp_baseline.pt"
METRICS_PATH = RESULTS_DIRECTORY / "mlp_baseline_metrics.json"

EDGE_FEATURE_COLUMNS = [
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS"
]

INPUT_SIZE = len(EDGE_FEATURE_COLUMNS)

HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16

NUMBER_OF_CLASSES = 2

BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUMBER_OF_EPOCHS = 30

RANDOM_SEED = 42

DEVICE = torch.device("cpu")

#---------------------------------------------------------
#Model
#--------------------------------------------------------

class FlowMLP(nn.Module):
    """Simple natural networck for classifying individual flows. """
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            # 8 flow features -> 32 learned values
            nn.Linear(INPUT_SIZE,HIDDEN_SIZE_1,),
            nn.ReLU(),
            # 32 -> 16
            nn.Linear(HIDDEN_SIZE_1,HIDDEN_SIZE_2,),
            nn.ReLU(),
            # 16 -> 2 scores:
            # score for Benign and score for Attack
            nn.Linear(HIDDEN_SIZE_2,NUMBER_OF_CLASSES),
        )
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
    
#---------------------------------------------------------
#Data Loading
#---------------------------------------------------------
def load_dataset( path: Path,) -> TensorDataset:
     """Load one normalized CSV split into PyTorch tensors."""
     flows = pd.read_csv(path)
     features = torch.tensor(flows[EDGE_FEATURE_COLUMNS].values, dtype=torch.float32)
     labels = torch.tensor(flows["Label"].values, dtype=torch.long)
     return TensorDataset(features, labels)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_model(model: nn.Module, data_loader: DataLoader) -> dict:
    """Evaluate a model and calculate classification metrics."""
    model.eval()
    all_predictions = []
    all_labels = []

    tottal_loss = 0.0
    loss_function = nn.CrossEntropyLoss()

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(features)
            loss = loss_function(logits, labels)
            tottal_loss += (loss.item()*len(labels))
            predictions= torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        average_loss = tottal_loss / len(data_loader.dataset)
        return {
        "loss": average_loss,

        "accuracy": accuracy_score(
            all_labels,
            all_predictions,
        ),

        "precision": precision_score(
            all_labels,
            all_predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            all_labels,
            all_predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            all_labels,
            all_predictions,
            zero_division=0,
        ),

        "confusion_matrix": confusion_matrix(
            all_labels,
            all_predictions,
        ).tolist(),
    }
# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
def main() -> None:

    # Make our experiment reproducible.
    torch.manual_seed(RANDOM_SEED)

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Device: {DEVICE}")

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------

    train_dataset = load_dataset(
        TRAIN_PATH
    )

    validation_dataset = load_dataset(
        VALIDATION_PATH
    )

    test_dataset = load_dataset(
        TEST_PATH
    )

    print("\nDataset sizes:")
    print(f"Train:      {len(train_dataset):,}")
    print(f"Validation: {len(validation_dataset):,}")
    print(f"Test:       {len(test_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # -----------------------------------------------------
    # Create model
    # -----------------------------------------------------

    model = FlowMLP().to(DEVICE)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_f1 = -1.0

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    for epoch in range(
        1,
        NUMBER_OF_EPOCHS + 1,
    ):

        model.train()

        total_training_loss = 0.0

        for features, labels in train_loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            # Remove gradients from the previous batch.
            optimizer.zero_grad()

            # Forward pass.
            logits = model(features)

            # Calculate error.
            loss = loss_function(
                logits,
                labels,
            )

            # Calculate gradients.
            loss.backward()

            # Update model weights.
            optimizer.step()

            total_training_loss += (
                loss.item()
                * len(labels)
            )

        average_training_loss = (
            total_training_loss
            / len(train_dataset)
        )

        # Evaluate after every epoch.
        validation_metrics = evaluate_model(
            model,
            validation_loader,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss={average_training_loss:.4f} | "
            f"val loss={validation_metrics['loss']:.4f} | "
            f"val acc={validation_metrics['accuracy']:.4f} | "
            f"val F1={validation_metrics['f1']:.4f}"
        )

        # Save only the model that performs best
        # on validation F1.
        if (
            validation_metrics["f1"]
            > best_validation_f1
        ):
            best_validation_f1 = (
                validation_metrics["f1"]
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

    # -----------------------------------------------------
    # Test the BEST validation model
    # -----------------------------------------------------

    print("\nLoading best model...")

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    test_metrics = evaluate_model(
        model,
        test_loader,
    )

    print("\nTest results")
    print("------------")

    print(
        f"Accuracy:  "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1-score:  "
        f"{test_metrics['f1']:.4f}"
    )

    print("\nConfusion matrix:")
    print(test_metrics["confusion_matrix"])

    # Save results so we can compare them
    # against E-GraphSAGE later.
    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            test_metrics,
            file,
            indent=4,
        )

    print("\nSaved:")
    print(MODEL_PATH)
    print(METRICS_PATH)


if __name__ == "__main__":
    main()
