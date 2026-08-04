"""Create a balanced working sample from NF-UNSW-NB15.

The complete dataset contains more than 1.6 million flows and is
highly imbalanced:

- approximately 95.5% benign
- approximately 4.5% attack

For our first real experiment, we create a smaller balanced sample:

- 10,000 benign flows
- 10,000 attack flows

The script reads the source CSV in chunks so it does not load the
entire dataset into memory.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nf_unsw_nb15"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nf_unsw_nb15_balanced_20k.csv"
)

TARGET_PER_CLASS = 10_000
CHUNK_SIZE = 100_000
RANDOM_SEED = 42


def find_dataset_path() -> Path:
    """Find the original NF-UNSW-NB15 CSV file."""

    matches = list(
        RAW_DATA_DIRECTORY.rglob("NF-UNSW-NB15.csv")
    )

    if not matches:
        raise FileNotFoundError(
            "NF-UNSW-NB15.csv was not found inside:\n"
            f"{RAW_DATA_DIRECTORY}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            "Multiple matching dataset files were found:\n"
            + "\n".join(str(path) for path in matches)
        )

    return matches[0]


def keep_best_random_candidates(
    current_candidates: pd.DataFrame,
    new_candidates: pd.DataFrame,
    target_size: int,
) -> pd.DataFrame:
    """Keep the rows with the smallest random sampling keys.

    Every row receives a random number.

    After processing the complete dataset, keeping the rows with the
    smallest random numbers gives us a random sample without loading
    the entire CSV into memory.
    """

    combined = pd.concat(
        [current_candidates, new_candidates],
        ignore_index=True,
    )

    if len(combined) <= target_size:
        return combined

    return combined.nsmallest(
        target_size,
        columns="_sample_key",
    )


def main() -> None:
    dataset_path = find_dataset_path()

    print(f"Reading dataset:\n{dataset_path}")
    print()

    # NumPy's random generator creates reproducible random numbers.
    random_generator = np.random.default_rng(RANDOM_SEED)

    benign_candidates = pd.DataFrame()
    attack_candidates = pd.DataFrame()

    chunks = pd.read_csv(
        dataset_path,
        chunksize=CHUNK_SIZE,
    )

    total_processed = 0

    for chunk_number, chunk in enumerate(chunks, start=1):
        total_processed += len(chunk)

        # Give every row an independent random sampling key.
        chunk["_sample_key"] = random_generator.random(
            len(chunk)
        )

        benign_rows = chunk[chunk["Label"] == 0]
        attack_rows = chunk[chunk["Label"] == 1]

        benign_candidates = keep_best_random_candidates(
            benign_candidates,
            benign_rows,
            TARGET_PER_CLASS,
        )

        attack_candidates = keep_best_random_candidates(
            attack_candidates,
            attack_rows,
            TARGET_PER_CLASS,
        )

        print(
            f"Chunk {chunk_number:2d} | "
            f"processed={total_processed:,} | "
            f"benign candidates={len(benign_candidates):,} | "
            f"attack candidates={len(attack_candidates):,}"
        )

    if len(benign_candidates) < TARGET_PER_CLASS:
        raise ValueError(
            "Not enough benign flows were found."
        )

    if len(attack_candidates) < TARGET_PER_CLASS:
        raise ValueError(
            "Not enough attack flows were found."
        )

    # Combine both classes.
    sample = pd.concat(
        [
            benign_candidates,
            attack_candidates,
        ],
        ignore_index=True,
    )

    # Remove the temporary random-number column.
    sample = sample.drop(columns="_sample_key")

    # Shuffle benign and attack rows together.
    sample = sample.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nBalanced sample created")
    print("-----------------------")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Shape: {sample.shape}")

    print("\nBinary label counts:")
    print(
        sample["Label"]
        .value_counts()
        .sort_index()
    )

    print("\nAttack categories:")
    print(
        sample["Attack"]
        .value_counts()
    )


if __name__ == "__main__":
    main()