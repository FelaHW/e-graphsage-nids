"""Convert the NF-UNSW-NB15 working sample into a PyG graph.

Graph representation:

- Node:
    (IP address, port)

- Edge:
    One network flow

- Edge features:
    Protocol, bytes, packets, TCP flags and duration

- Edge label:
    0 = benign
    1 = attack
"""

from pathlib import Path
from typing import TypeAlias

import pandas as pd
import torch
from torch_geometric.data import Data

Endpoint: TypeAlias = tuple[str, int]  # (IP address, port)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_PATH =(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nf_unsw_nb15_balanced_20k.csv"
)
# The order here is important.
# This will also be the order of values inside edge_attr.
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

REQUIRED_COLUMNS = {
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "Label",
    *EDGE_FEATURE_COLUMNS,
}

def validate_dataset(flows: pd.DataFrame) -> None:
    """Check that the DataFrame contains all required columns."""
    if flows.empty:
        raise ValueError("The DataFrame is empty.")

    missing_columns = REQUIRED_COLUMNS - set(flows.columns)

    if missing_columns:
        raise ValueError(
            f"missing required columns: {sorted(missing_columns)}"
        )
    
    invalid_labels = set(flows["Label"].unique()) - {0, 1}

    if invalid_labels:
        raise ValueError(f"unexpected binary label: {sorted(invalid_labels)}")

    # check for missing values in columns used by the model
    column_to_check = [
        "IPV4_SRC_ADDR",
        "L4_SRC_PORT",
        "IPV4_DST_ADDR",
        "L4_DST_PORT",
        "Label",
        *EDGE_FEATURE_COLUMNS,
    ]
    missing_values_counts = flows[column_to_check].isna().sum()
    missing_values_counts = missing_values_counts[missing_values_counts > 0]

    if not missing_values_counts.empty:
        raise ValueError(
            f"missing values were found:\n{missing_values_counts}"
        )

def create_endpoint_mapping(flows: pd.DataFrame) -> dict[Endpoint, int]:
    """assign one numeric node id to each endpoint."""   
    endpoint_to_id: dict[Endpoint,int] = {}

    for flow in flows.itertuples(index = False):
        source_endpoint: Endpoint = (str(flow.IPV4_SRC_ADDR), int(flow.L4_SRC_PORT))

        destination_endpoint: Endpoint = (str(flow.IPV4_DST_ADDR), int(flow.L4_DST_PORT))
        
        if source_endpoint not in endpoint_to_id:
            endpoint_to_id[source_endpoint] = len(endpoint_to_id)
        if destination_endpoint not in endpoint_to_id:
            endpoint_to_id[destination_endpoint] = len(endpoint_to_id)

    return endpoint_to_id

def dataframe_to_nf_unsw_graph(flows: pd.DataFrame) -> tuple[Data, dict[Endpoint,int]]:
    "convert NF-UNSW-NB15 rows into graph tensors."

    validate_dataset(flows)

    endpoint_to_id = create_endpoint_mapping(flows)

    source_node_ids : list[int]= []
    destination_node_ids : list[int]= []

    for flow in flows.itertuples(index =False):
        source_endpoint = (str(flow.IPV4_SRC_ADDR), int(flow.L4_SRC_PORT))
        destination_endpoint = (str(flow.IPV4_DST_ADDR), int(flow.L4_DST_PORT))

        source_node_ids.append(endpoint_to_id[source_endpoint])
        destination_node_ids.append(endpoint_to_id[destination_endpoint])

    edge_index = torch.tensor(
        [source_node_ids, destination_node_ids], dtype=torch.long
    )
    edge_attr = torch.tensor(
        flows[EDGE_FEATURE_COLUMNS].to_numpy(), dtype=torch.float32
    )
    edge_y = torch.tensor(flows["Label"].to_numpy(), dtype=torch.long
    )
    number_of_nodes = len(endpoint_to_id)
    number_of_features = len(EDGE_FEATURE_COLUMNS)
    node_features = torch.ones((number_of_nodes, number_of_features), dtype=torch.float32)
    graph = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_y=edge_y,
    )

    return graph, endpoint_to_id

def main() -> None:
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Working sample was not found:\n{SAMPLE_PATH}"
        )

    print(f"Loading sample:\n{SAMPLE_PATH}\n")

    flows = pd.read_csv(SAMPLE_PATH)

    graph, endpoint_to_id = dataframe_to_nf_unsw_graph(flows)

    print("Real graph summary")
    print("------------------")
    print(f"Number of flows: {len(flows):,}")
    print(f"Number of nodes: {graph.num_nodes:,}")
    print(f"Number of edges: {graph.num_edges:,}")
    print(f"Node features: {graph.x.shape}")
    print(f"Edge index: {graph.edge_index.shape}")
    print(f"Edge features: {graph.edge_attr.shape}")
    print(f"Edge labels: {graph.edge_y.shape}")

    print("\nFeature order:")
    for index, feature_name in enumerate(EDGE_FEATURE_COLUMNS):
        print(f"{index}: {feature_name}")

    print("\nLabel counts:")
    print(torch.bincount(graph.edge_y))

    print("\nFirst five edges:")

    id_to_endpoint = {
        node_id: endpoint
        for endpoint, node_id in endpoint_to_id.items()
    }

    for edge_id in range(5):
        source_id = int(graph.edge_index[0, edge_id])
        destination_id = int(graph.edge_index[1, edge_id])

        print(
            f"\nEdge {edge_id}: "
            f"{id_to_endpoint[source_id]} -> "
            f"{id_to_endpoint[destination_id]}"
        )
        print(f"Features: {graph.edge_attr[edge_id]}")
        print(f"Label: {int(graph.edge_y[edge_id])}")

    # Basic consistency checks.
    assert graph.num_edges == len(flows)
    assert graph.edge_attr.shape == (
        len(flows),
        len(EDGE_FEATURE_COLUMNS),
    )
    assert graph.edge_y.shape[0] == len(flows)


if __name__ == "__main__":
    main()
    
     

