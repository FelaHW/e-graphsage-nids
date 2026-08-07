"""Inspect the numerical feature scales in the working dataset.

Before normalizing the features, we inspect:

- minimum
- maximum
- mean
- standard deviation
- percentiles
- missing and infinite values
- number of zero values
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nf_unsw_nb15_balanced_20k.csv"
)


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


def main() -> None:
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Working sample was not found:\n{SAMPLE_PATH}"
        )

    flows = pd.read_csv(
        SAMPLE_PATH,
        usecols=EDGE_FEATURE_COLUMNS,
    )

    print("Dataset shape:")
    print(flows.shape)

    print("\nMissing values:")
    print(flows.isna().sum())

    # Convert the DataFrame to NumPy so we can check for infinity.
    feature_values = flows.to_numpy(dtype=np.float64)

    print("\nNumber of infinite values:")
    print(np.isinf(feature_values).sum())

    print("\nFeature statistics:")
    print(
        flows.describe(
            percentiles=[
                0.01,
                0.25,
                0.50,
                0.75,
                0.99,
            ]
        ).T.to_string()
    )

    print("\nZero-value counts:")

    for feature_name in EDGE_FEATURE_COLUMNS:
        zero_count = int((flows[feature_name] == 0).sum())
        zero_percentage = zero_count / len(flows) * 100

        print(
            f"{feature_name}: "
            f"{zero_count:,} "
            f"({zero_percentage:.2f}%)"
        )


if __name__ == "__main__":
    main()