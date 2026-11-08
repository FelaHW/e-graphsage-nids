"""Inspect the NF-ToN-IoT dataset without loading it completely.

The dataset is large, so this script reads only a small sample.

We want to discover:
- available CSV files
- column names
- sample values
- data types
- possible label columns
"""

from pathlib import Path

import pandas as pd


# Project root:
# e-graphsage-nids/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder containing the extracted real dataset.
DATASET_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nf_ton_iot"
    / "extracted"
)


def find_dataset_files(directory: Path) -> list[Path]:
    """Find normal CSV files and compressed CSV files."""

    csv_files = list(directory.rglob("*.csv"))
    compressed_csv_files = list(directory.rglob("*.csv.gz"))

    return sorted(csv_files + compressed_csv_files)


def inspect_csv(csv_path: Path, sample_size: int = 5) -> None:
    """Read and display only a small sample from one CSV file."""

    print("=" * 70)
    print(f"File: {csv_path}")
    print(
        "Size: "
        f"{csv_path.stat().st_size / (1024 * 1024):.2f} MB"
    )

    # Read only a few rows.
    # This avoids loading the complete dataset into memory.
    sample = pd.read_csv(
        csv_path,
        nrows=sample_size,
    )

    print("\nSample shape:")
    print(sample.shape)

    print("\nColumn names:")
    for column_number, column_name in enumerate(sample.columns):
        print(f"{column_number:2d}: {column_name}")

    print("\nFirst rows:")
    print(sample.to_string())

    print("\nDetected data types:")
    print(sample.dtypes)

    # Look for likely label-related column names.
    possible_label_columns = [
        column
        for column in sample.columns
        if any(
            word in column.lower()
            for word in ["label", "attack", "class", "category"]
        )
    ]

    print("\nPossible label columns:")
    if possible_label_columns:
        print(possible_label_columns)

        for column in possible_label_columns:
            print(f"\nValues found in '{column}':")
            print(sample[column].value_counts(dropna=False))
    else:
        print("No obvious label column was found.")


def main() -> None:
    if not DATASET_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Extracted dataset folder was not found:\n"
            f"{DATASET_DIRECTORY}"
        )

    dataset_files = find_dataset_files(DATASET_DIRECTORY)

    if not dataset_files:
        raise FileNotFoundError(
            "No CSV or CSV.GZ files were found inside:\n"
            f"{DATASET_DIRECTORY}"
        )

    print(f"Found {len(dataset_files)} dataset file(s).\n")

    for csv_path in dataset_files:
        inspect_csv(csv_path)


if __name__ == "__main__":
    main()