"""Build separate NF-UNSW-NB15 graphs for training,
validation and testing.

Graph representation:

Node:
    (IP address, port)

Edge:
    One network flow

Edge features:
    8 normalized NetFlow features

Edge label:
    0 = Benign
    1 = Attack

This file ONLY builds the graphs.
It does not train the GNN.
"""
from pathlib import Path
from typing import TypeAlias

import pandas as pd
import torch
from torch_geometric.data import Data

# ---------------------------------------------------------
# Types
# ---------------------------------------------------------

Endpoint: TypeAlias = tuple[str, int]

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nf_unsw_nb15_splits"
)

TRAIN_PATH = DATA_DIRECTORY / "train_normalized.csv"
VALIDATION_PATH = DATA_DIRECTORY / "validation_normalized.csv"
TEST_PATH = DATA_DIRECTORY / "test_normalized.csv"
# ---------------------------------------------------------
# Features
# ---------------------------------------------------------

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
# ---------------------------------------------------------
# DataFrame -> Graph
# ---------------------------------------------------------

def dataframe_to_graph(flows: pd.DataFrame) -> Data:
    """Convert one dataset split into a PyG graph."""
    required_columns = {
        "IPV4_SRC_ADDR",
        "L4_SRC_PORT",
        "IPV4_DST_ADDR",
        "L4_DST_PORT",
        "Label",
        *EDGE_FEATURE_COLUMNS,
    }
    missing_columns = required_columns - set(flows.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if flows.empty:
        raise ValueError("Cannot build a graph from an empty split.")
    if flows[list(required_columns)].isna().any().any():
        raise ValueError("Missing values found in graph columns.")
    if not flows["Label"].isin([0, 1]).all():
        raise ValueError("Edge labels must be 0 (benign) or 1 (attack).")

    # Each split gets its own mapping. Node IDs need not match across
    # graphs because they index endpoints locally, not learned identities.
    endpoint_to_id: dict[Endpoint, int] = {}
    source_node_ids: list[int] = []
    destination_node_ids: list[int] = []

    # One DataFrame loop both assigns unseen endpoints and records edges.
    for flow in flows.itertuples(index=False):
        source_endpoint: Endpoint = (str(flow.IPV4_SRC_ADDR), int(flow.L4_SRC_PORT))
        destination_endpoint: Endpoint = (str(flow.IPV4_DST_ADDR), int(flow.L4_DST_PORT))
        if source_endpoint not in endpoint_to_id:
            endpoint_to_id[source_endpoint] = len(endpoint_to_id)
        if destination_endpoint not in endpoint_to_id:
            endpoint_to_id[destination_endpoint] = len(endpoint_to_id)
        source_id = endpoint_to_id[source_endpoint]
        destination_id = endpoint_to_id[destination_endpoint]
        source_node_ids.append(source_id)
        destination_node_ids.append(destination_id)

    # Row 0 contains source IDs; row 1 contains destination IDs.
    # Repeated endpoint pairs remain separate directed flow edges.
    edge_index = torch.tensor(
        [source_node_ids, destination_node_ids], dtype=torch.long
    )
    edge_attr = torch.tensor(
        flows[EDGE_FEATURE_COLUMNS].to_numpy(), dtype=torch.float32
    )
    edge_y = torch.tensor(flows["Label"].to_numpy(), dtype=torch.long)

    number_of_nodes = len(endpoint_to_id)
    number_of_features = len(EDGE_FEATURE_COLUMNS)
    # Ones are placeholders; future aggregation will supply flow context.
    node_features = torch.ones(
        (number_of_nodes, number_of_features), dtype=torch.float32
    )
    graph = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_y=edge_y,
    )

    # Check alignment, tensor types, finite features, and node-ID bounds.
    assert graph.num_nodes == number_of_nodes
    assert graph.num_edges == len(flows)
    assert graph.x.shape == (number_of_nodes, number_of_features)
    assert graph.edge_index.shape == (2, len(flows))
    assert graph.edge_attr.shape == (len(flows), number_of_features)
    assert graph.edge_y.shape == (len(flows),)
    assert graph.edge_index.dtype == torch.long
    assert graph.edge_attr.dtype == torch.float32
    assert graph.edge_y.dtype == torch.long
    assert torch.isfinite(graph.edge_attr).all().item()
    assert torch.equal(
        torch.bincount(graph.edge_y, minlength=2),
        torch.tensor(
            flows["Label"].value_counts().reindex([0, 1], fill_value=0).to_numpy(),
            dtype=torch.long,
        ),
    )
    graph.validate(raise_on_error=True)
    return graph


def main() -> None:
    split_paths = {
        "Training": TRAIN_PATH,
        "Validation": VALIDATION_PATH,
        "Test": TEST_PATH,
    }
    split_flows = {
        name: pd.read_csv(path) for name, path in split_paths.items()
    }

    # This is a consistency check only: no statistics or features are
    # fitted here. In particular, graphs never merge across splits.
    combined_flows = pd.concat(list(split_flows.values()), ignore_index=True)
    assert not combined_flows.duplicated().any(), "Duplicate flows across splits."

    graphs = {
        name: dataframe_to_graph(flows) for name, flows in split_flows.items()
    }
    for split_name, graph in graphs.items():
        print(f"\n{split_name} graph")
        print(f"Number of nodes: {graph.num_nodes:,}")
        print(f"Number of edges: {graph.num_edges:,}")
        print(f"Node feature shape: {graph.x.shape}")
        print(f"edge_index.shape: {graph.edge_index.shape}")
        print(f"edge_attr.shape: {graph.edge_attr.shape}")
        print(f"edge_y.shape: {graph.edge_y.shape}")
        print(f"Label counts [benign, attack]: {torch.bincount(graph.edge_y, minlength=2)}")


if __name__ == "__main__":
    main()
