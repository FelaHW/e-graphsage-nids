"""Load network-flow records from a CSV file.

Real intrusion-detection datasets are usually stored in CSV files.
Each row represents one network flow.
"""
from pathlib import Path

import pandas as pd 

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOY_DATASET_PATH = PROJECT_ROOT /"data"/ "toy_flows.csv"

def load_flow_csv(csv_path: Path) -> pd.DataFrame :
    if not csv_path.exists():
        raise FileExistsError(f"Dataset was not founs {csv_path}")
    flows = pd.read_csv(csv_path)
    return flows

def main() -> None:

    flows = load_flow_csv(TOY_DATASET_PATH)

    print("Loaded flow table:")
    print(flows)

    # shape returns:
    # (number of rows, number of columns)
    print("\nTable shape:")
    print(flows.shape)

    print("\nColumn names:")
    print(flows.columns.tolist())

    # Count how many examples exist for every label.
    print("\nLabel counts:")
    print(
        flows["label"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()
