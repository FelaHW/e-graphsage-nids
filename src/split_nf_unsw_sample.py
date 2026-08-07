"""Split the NF-UNSW-NB15 working sample.

The split is stratified by the binary Label column, so each subset
keeps the same benign/attack ratio.

Result:

- 70% training
- 15% validation
- 15% testing
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nf_unsw_nb15_balanced_20k.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nf_unsw_nb15_splits"
)

TRAIN_PATH = OUTPUT_DIRECTORY / "train.csv"
VALIDATION_PATH = OUTPUT_DIRECTORY / "validation.csv"
TEST_PATH = OUTPUT_DIRECTORY / "test.csv"

RANDOM_SEED = 42


def print_split_summary(
    split_name: str,
    split: pd.DataFrame,
) -> None:
    """Print the size and label distribution of one split."""

    print(f"\n{split_name}")
    print("-" * len(split_name))
    print(f"Number of flows: {len(split):,}")

    print("\nBinary labels:")
    print(
        split["Label"]
        .value_counts()
        .sort_index()
    )

    print("\nAttack categories:")
    print(
        split["Attack"]
        .value_counts()
    )


def main() -> None:
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Working sample was not found:\n{SAMPLE_PATH}"
        )

    flows = pd.read_csv(SAMPLE_PATH)

    # First split:
    # 70% training and 30% temporary data.
    train_flows, temporary_flows = train_test_split(
        flows,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=flows["Label"],
    )

    # Second split:
    # Divide the temporary 30% equally into validation and test.
    validation_flows, test_flows = train_test_split(
        temporary_flows,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temporary_flows["Label"],
    )

    # Reset the row numbers after splitting.
    train_flows = train_flows.reset_index(drop=True)
    validation_flows = validation_flows.reset_index(drop=True)
    test_flows = test_flows.reset_index(drop=True)

    # Confirm that every original row appears in exactly one split.
    assert (
        len(train_flows)
        + len(validation_flows)
        + len(test_flows)
        == len(flows)
    )

    # Expected sizes for a dataset containing 20,000 rows.
    assert len(train_flows) == 14_000
    assert len(validation_flows) == 3_000
    assert len(test_flows) == 3_000

    # Confirm that each split remains balanced.
    assert train_flows["Label"].value_counts().to_dict() == {
        0: 7_000,
        1: 7_000,
    }

    assert validation_flows["Label"].value_counts().to_dict() == {
        0: 1_500,
        1: 1_500,
    }

    assert test_flows["Label"].value_counts().to_dict() == {
        0: 1_500,
        1: 1_500,
    }

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_flows.to_csv(
        TRAIN_PATH,
        index=False,
    )

    validation_flows.to_csv(
        VALIDATION_PATH,
        index=False,
    )

    test_flows.to_csv(
        TEST_PATH,
        index=False,
    )

    print("Dataset split completed")
    print("=======================")

    print(f"\nTraining file:\n{TRAIN_PATH}")
    print(f"\nValidation file:\n{VALIDATION_PATH}")
    print(f"\nTest file:\n{TEST_PATH}")

    print_split_summary(
        "Training split",
        train_flows,
    )

    print_split_summary(
        "Validation split",
        validation_flows,
    )

    print_split_summary(
        "Test split",
        test_flows,
    )


if __name__ == "__main__":
    main()