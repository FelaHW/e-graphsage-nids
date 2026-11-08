"""Profile NF-UNSW-NB15 without loading the whole dataset at once.

We read only the label columns in chunks and count:

- total flows
- benign flows
- attack flows
- individual attack categories
"""

from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nf_unsw_nb15"
    / "extracted"
    / "88695f0f620eb568_MOHANAD_A4706"
    / "data"
    / "NF-UNSW-NB15.csv"
)

CHUNK_SIZE = 100_000


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset was not found:\n{DATASET_PATH}"
        )

    total_flows = 0

    # Counter works like a dictionary that counts repeated values.
    binary_label_counts: Counter[int] = Counter()
    attack_category_counts: Counter[str] = Counter()

    # chunksize tells pandas to read only part of the CSV at a time.
    chunks = pd.read_csv(
        DATASET_PATH,
        usecols=["Label", "Attack"],
        chunksize=CHUNK_SIZE,
    )

    for chunk_number, chunk in enumerate(chunks, start=1):
        total_flows += len(chunk)

        binary_label_counts.update(
            chunk["Label"].astype(int).tolist()
        )

        attack_category_counts.update(
            chunk["Attack"].astype(str).tolist()
        )

        print(
            f"Processed chunk {chunk_number}: "
            f"{total_flows:,} flows"
        )

    print("\nDataset summary")
    print("----------------")

    print(f"Total flows: {total_flows:,}")

    print("\nBinary label counts:")

    for label, count in sorted(binary_label_counts.items()):
        class_name = "Benign" if label == 0 else "Attack"
        percentage = count / total_flows * 100

        print(
            f"{label} ({class_name}): "
            f"{count:,} flows "
            f"({percentage:.2f}%)"
        )

    print("\nAttack-category counts:")

    for attack_name, count in attack_category_counts.most_common():
        percentage = count / total_flows * 100

        print(
            f"{attack_name}: "
            f"{count:,} flows "
            f"({percentage:.2f}%)"
        )


if __name__ == "__main__":
    main()