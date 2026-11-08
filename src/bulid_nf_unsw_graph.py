"""Build separate NF-UNSW-NB15 graphs for
training, validation and testing.

Node:
    (IP address, port)

Edge:
    Network flow

Edge features:
    8 normalized NetFlow features

Edge label:
    0 = Benign
    1 = Attack
"""

from pathlib import Path
from typing import TypeAlias

import pandas as pd
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------
# Type
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

VALIDATION_PATH = (
    DATA_DIRECTORY
    / "validation_normalized.csv"
)

TEST_PATH = DATA_DIRECTORY / "test_normalized.csv"


# ---------------------------------------------------------
# Edge features
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
# Convert DataFrame -> graph
# ---------------------------------------------------------

def dataframe_to_graph(
    flows: pd.DataFrame,
) -> Data:
    """Convert one dataset split into a PyG graph."""

    # Maps:
    #
    # (IP, port) -> integer node ID
    #
    # Example:
    #
    # ("59.166.0.7", 36008) -> 0
    endpoint_to_id: dict[Endpoint, int] = {}

    source_node_ids: list[int] = []
    destination_node_ids: list[int] = []

    # -----------------------------------------------------
    # Process every network flow
    # -----------------------------------------------------

    for flow in flows.itertuples(index=False):

        source_endpoint: Endpoint = (
            str(flow.IPV4_SRC_ADDR),
            int(flow.L4_SRC_PORT),
        )

        destination_endpoint: Endpoint = (
            str(flow.IPV4_DST_ADDR),
            int(flow.L4_DST_PORT),
        )

        # Create source node if we have not seen it before.
        if source_endpoint not in endpoint_to_id:
            endpoint_to_id[source_endpoint] = len(
                endpoint_to_id
            )

        # Create destination node if it is new.
        if destination_endpoint not in endpoint_to_id:
            endpoint_to_id[destination_endpoint] = len(
                endpoint_to_id
            )

        # Both endpoints now have node IDs.
        source_id = endpoint_to_id[
            source_endpoint
        ]

        destination_id = endpoint_to_id[
            destination_endpoint
        ]

        # Save this flow as an edge.
        source_node_ids.append(source_id)

        destination_node_ids.append(
            destination_id
        )

    # -----------------------------------------------------
    # Edge structure
    # -----------------------------------------------------

    edge_index = torch.tensor(
        [
            source_node_ids,
            destination_node_ids,
        ],
        dtype=torch.long,
    )

    # -----------------------------------------------------
    # Edge features
    # -----------------------------------------------------

    edge_attr = torch.tensor(
        flows[
            EDGE_FEATURE_COLUMNS
        ].to_numpy(),
        dtype=torch.float32,
    )

    # -----------------------------------------------------
    # Edge labels
    # -----------------------------------------------------

    edge_y = torch.tensor(
        flows["Label"].to_numpy(),
        dtype=torch.long,
    )

    # -----------------------------------------------------
    # Initial node features
    # -----------------------------------------------------

    number_of_nodes = len(endpoint_to_id)

    # We don't currently have direct information
    # describing each node.
    #
    # Later the GNN will create useful node information
    # by aggregating connected edge features.
    node_features = torch.ones(
        (
            number_of_nodes,
            len(EDGE_FEATURE_COLUMNS),
        ),
        dtype=torch.float32,
    )

    # -----------------------------------------------------
    # PyTorch Geometric graph
    # -----------------------------------------------------

    graph = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_y=edge_y,
    )

    # -----------------------------------------------------
    # Safety checks
    # -----------------------------------------------------

    number_of_flows = len(flows)

    assert graph.num_edges == number_of_flows

    assert graph.edge_index.shape == (
        2,
        number_of_flows,
    )

    assert graph.edge_attr.shape == (
        number_of_flows,
        len(EDGE_FEATURE_COLUMNS),
    )

    assert graph.edge_y.shape == (
        number_of_flows,
    )

    return graph


# ---------------------------------------------------------
# Load one CSV
# ---------------------------------------------------------

def load_graph(
    path: Path,
) -> Data:
    """Load a normalized CSV and convert it to a graph."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset split was not found:\n{path}"
        )

    flows = pd.read_csv(path)

    return dataframe_to_graph(flows)


# ---------------------------------------------------------
# Print graph information
# ---------------------------------------------------------

def print_graph_summary(
    name: str,
    graph: Data,
) -> None:
    """Print important information about one graph."""

    print(f"\n{name}")
    print("-" * len(name))

    print(
        f"Nodes: {graph.num_nodes:,}"
    )

    print(
        f"Edges: {graph.num_edges:,}"
    )

    print(
        f"Node features: {graph.x.shape}"
    )

    print(
        f"edge_index: {graph.edge_index.shape}"
    )

    print(
        f"edge_attr: {graph.edge_attr.shape}"
    )

    print(
        f"edge_y: {graph.edge_y.shape}"
    )

    print(
        f"Labels: {torch.bincount(graph.edge_y)}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    print("Building graphs...")

    train_graph = load_graph(
        TRAIN_PATH
    )

    validation_graph = load_graph(
        VALIDATION_PATH
    )

    test_graph = load_graph(
        TEST_PATH
    )

    print_graph_summary(
        "Training graph",
        train_graph,
    )

    print_graph_summary(
        "Validation graph",
        validation_graph,
    )

    print_graph_summary(
        "Test graph",
        test_graph,
    )

    # Expected numbers of edges.
    assert train_graph.num_edges == 14_000

    assert validation_graph.num_edges == 3_000

    assert test_graph.num_edges == 3_000


if __name__ == "__main__":
    main()