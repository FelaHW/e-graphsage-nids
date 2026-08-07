"""Preprocess the NF-UNSW-NB15 train/validation/test splits.

Important rule:
Normalization statistics are calculated ONLY from the training set.

Steps:
1. Apply log1p to strongly skewed flow features.
2. Calculate mean and standard deviation from the training set.
3. Normalize train, validation and test using the SAME statistics.
4. Save the processed datasets and normalization statistics.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLITS_DIRECTORY = ( 
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nf_unsw_nb15_splits"
)

TRAIN_PATH = SPLITS_DIRECTORY / "train.csv"
VALIDATION_PATH = SPLITS_DIRECTORY / "validation.csv"
TEST_PATH = SPLITS_DIRECTORY / "test.csv"

EDGE_FEATURE_COLUMNS = [
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
]

LOG_TRANSFORM_COLUMNS = [
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
]

def apply_log_transform(flows: pd.DataFrame) -> pd.DataFrame:
    flows = flows.copy()

    for column in LOG_TRANSFORM_COLUMNS:
        if (flows[column] < 0).any():
            raise ValueError(
                f"Negative values found in column '{column}'"
            )

        flows[column] = np.log1p(flows[column])

    return flows

def  calculate_training_statistics(train_flows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """"calculate normalization statistics (mean and std) from the training data only."""

    mean = train_flows[EDGE_FEATURE_COLUMNS].mean()
    standard_deviation = (train_flows[EDGE_FEATURE_COLUMNS].std(ddof=0))

    standard_deviation=standard_deviation.replace(0,1.0)

    return mean, standard_deviation

def normalize_features(flows: pd.DataFrame, mean: pd.Series, standard_deviation: pd.Series) -> pd.DataFrame:
    """Normalize the features using training-set statistics."""
    flows = flows.copy()

    flows[EDGE_FEATURE_COLUMNS] = (flows[EDGE_FEATURE_COLUMNS] - mean) / standard_deviation

    return flows

def verify_dataset(name: str , flows: pd.DataFrame) -> None:
    """Verify that the dataset is valid after preprocessing."""
    values = flows[EDGE_FEATURE_COLUMNS].to_numpy()

    if np.isnan(values).any():
        raise ValueError(f"NaN values found in {name} dataset after preprocessing.")
    if np.isinf(values).any():
        raise ValueError(f"Infinite values found in {name} dataset after preprocessing.")

def main() -> None:
     # -------------------------
    # Load original splits
    # -------------------------

    train_flows = pd.read_csv(TRAIN_PATH)
    validation_flows = pd.read_csv(VALIDATION_PATH)
    test_flows = pd.read_csv(TEST_PATH)

    print("Original split sizes:")
    print(f"Train:      {len(train_flows):,}")
    print(f"Validation: {len(validation_flows):,}")
    print(f"Test:       {len(test_flows):,}")

    # -------------------------
    # Step 1: log transformation
    # -------------------------

    train_flows = apply_log_transform(train_flows)
    validation_flows = apply_log_transform(validation_flows)
    test_flows = apply_log_transform(test_flows)

    # -------------------------
    # Step 2: training statistics
    # -------------------------

    means, standard_deviations = (
        calculate_training_statistics(train_flows)
    )

    print("\nTraining means after log transformation:")
    print(means)

    print("\nTraining standard deviations:")
    print(standard_deviations)

    # -------------------------
    # Step 3: normalize
    # -------------------------

    train_normalized = normalize_features(
        train_flows,
        means,
        standard_deviations,
    )

    validation_normalized = normalize_features(
        validation_flows,
        means,
        standard_deviations,
    )

    test_normalized = normalize_features(
        test_flows,
        means,
        standard_deviations,
    )

    # -------------------------
    # Step 4: validate results
    # -------------------------

    verify_dataset(
        "Training set",
        train_normalized,
    )

    verify_dataset(
        "Validation set",
        validation_normalized,
    )

    verify_dataset(
        "Test set",
        test_normalized,
    )

    print("\nNormalized TRAIN feature means:")
    print(
        train_normalized[
            EDGE_FEATURE_COLUMNS
        ].mean()
    )

    print("\nNormalized TRAIN feature standard deviations:")
    print(
        train_normalized[
            EDGE_FEATURE_COLUMNS
        ].std(ddof=0)
    )

    # Labels must remain unchanged.
    print("\nTraining labels:")
    print(
        train_normalized["Label"]
        .value_counts()
        .sort_index()
    )

    # -------------------------
    # Step 5: save datasets
    # -------------------------

    train_output = (
        SPLITS_DIRECTORY / "train_normalized.csv"
    )

    validation_output = (
        SPLITS_DIRECTORY / "validation_normalized.csv"
    )

    test_output = (
        SPLITS_DIRECTORY / "test_normalized.csv"
    )

    train_normalized.to_csv(
        train_output,
        index=False,
    )

    validation_normalized.to_csv(
        validation_output,
        index=False,
    )

    test_normalized.to_csv(
        test_output,
        index=False,
    )

    # Save the normalization values.
    # Later we can use exactly the same values on new data.
    statistics = {
        "log_transform_columns": LOG_TRANSFORM_COLUMNS,
        "means": means.to_dict(),
        "standard_deviations": (
            standard_deviations.to_dict()
        ),
    }

    statistics_path = (
        SPLITS_DIRECTORY
        / "normalization_stats.json"
    )

    with open(
        statistics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            statistics,
            file,
            indent=4,
        )

    print("\nSaved:")
    print(train_output)
    print(validation_output)
    print(test_output)
    print(statistics_path)


if __name__ == "__main__":
    main()
    

