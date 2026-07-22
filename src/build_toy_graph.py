"""Build a small network-flow graph.

This is not the neural network yet.

The purpose of this program is to understand how network traffic
is represented as a graph:

- Node: an endpoint represented by (IP address, port)
- Edge: a network flow between two endpoints
- Edge features: packets, bytes, and duration
- Edge label: benign or attack
"""
from typing import Any

import torch 
from torch_geometric.data import Data

# Each dictionary represents one network flow.
#
# label:
# 0 = benign
# 1 = attack
FLOWS: list[dict[str,Any]]=[
    {
        "source_ip": "10.0.0.1",
        "source_port": 5000,
        "destination_ip": "10.0.0.2",
        "destination_port": 80,
        "packets": 12,
        "bytes": 2400,
        "duration": 1.2,
        "label": 0,
    },
    {
        "source_ip": "10.0.0.2",
        "source_port": 80,
        "destination_ip": "10.0.0.1",
        "destination_port": 5000,
        "packets": 10,
        "bytes": 1800,
        "duration": 1.0,
        "label": 0,
    },
    {
        "source_ip": "10.0.0.9",
        "source_port": 44000,
        "destination_ip": "10.0.0.3",
        "destination_port": 21,
        "packets": 2,
        "bytes": 120,
        "duration": 0.1,
        "label": 1,
    },
    {
        "source_ip": "10.0.0.9",
        "source_port": 44000,
        "destination_ip": "10.0.0.3",
        "destination_port": 22,
        "packets": 2,
        "bytes": 115,
        "duration": 0.1,
        "label": 1,
    },
    {
        "source_ip": "10.0.0.9",
        "source_port": 44000,
        "destination_ip": "10.0.0.3",
        "destination_port": 80,
        "packets": 2,
        "bytes": 125,
        "duration": 0.1,
        "label": 1,
    },
]

def get_endpoint(flow: dict[str, Any], side: str) -> tuple[str, int]:
    """return the source or destination endpoint of a flow"""
    if side == "source":
        return flow["source_ip"], flow["source_port"]
    if side == "destination":
        return flow["destination_ip"], flow["destination_port"]
    raise ValueError("side must be 'source' or 'destination'") 

def create_endpoint_mapping(flows: list[dict[str,Any]],) -> dict[tuple[str,int],int]:
    """Give each unique endpoint a unique integer ID.
     PyTorch graphs use integer node IDs, so endpoints such as
    ('10.0.0.1', 5000) must be converted into numbers.
    """
    endpoint_to_id: dict[tuple[str,int],int] = {}
    for flow in flows:
        source = get_endpoint(flow, "source")
        destination = get_endpoint(flow, "destination")

        if source not in endpoint_to_id:
            endpoint_to_id[source] = len(endpoint_to_id)
        if destination not in endpoint_to_id:
            endpoint_to_id[destination] = len(endpoint_to_id)
    return endpoint_to_id

def build_graph(flows: list[dict[str,Any]]) -> tuple[Data, dict[tuple[str,int],int]]:
    """convert a list of flows into a PyTorch Geometric graph"""

    endpoint_to_id = create_endpoint_mapping(flows)

    source_nodes: list[int] = []
    destination_nodes: list[int] = []
    edge_features: list[list[float]] = []
    edge_labels: list[int] = []

    for flow in flows: 
        source = get_endpoint(flow, "source")
        destination = get_endpoint(flow, "destination")

        source_nodes.append(endpoint_to_id[source])
        destination_nodes.append(endpoint_to_id[destination])

        # These values describe a communication flow,
        # so they are stored as edge features.
        edge_features.append([flow["packets"], flow["bytes"], flow["duration"]])

        edge_labels.append(flow["label"])
   # edge_index has two rows:
    #
    # Row 0: source node IDs
    # Row 1: destination node IDs
    edge_index = torch.tensor([source_nodes, destination_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    edge_y = torch.tensor(edge_labels, dtype=torch.long)

    number_of_nides = len(endpoint_to_id)
    number_of_features = edge_attr.shape[1]

    # Our dataset currently has edge features but no meaningful
    # node features. We therefore initialize each node with ones.

    node_features = torch.ones((number_of_nides, number_of_features), dtype=torch.float32)
    graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, edge_y=edge_y)
    return graph, endpoint_to_id

def print_graph(graph: Data, endpoint_to_id: dict[tuple[str,int],int]) -> None:
    """print a graph in a human-readable format"""

    id_to_endpoint = {node_id: endpoint for endpoint, node_id in endpoint_to_id.items()}
    
    print("Graph:")
    print(f"  Number of nodes: {graph.num_nodes}")
    print(f"  Number of edges: {graph.num_edges}")
    print(f"  Node features shape: {graph.x.shape}")
    print(f"  Edge features shape: {graph.edge_attr.shape}")
    print(f"  Edge labels shape: {graph.edge_y.shape}")

    for edge_id in range(graph.num_edges):
        source_id = int(graph.edge_index[0, edge_id])
        destination_id = int(graph.edge_index[1, edge_id])

        source_endpoint = id_to_endpoint[source_id]
        destination_endpoint = id_to_endpoint[destination_id]

        features = graph.edge_attr[edge_id].tolist()
        label = int(graph.edge_y[edge_id])

        class_name = "Attack" if label == 1 else "Benign"

        print(
             f"Edge {edge_id}: "
            f"{source_endpoint} -> {destination_endpoint} | "
            f"features={features} | "
            f"class={class_name}"
        )

def main() -> None:
    graph, endpoint_to_id = build_graph(FLOWS)

    # These checks help us detect graph-construction mistakes.
    assert graph.num_edges == len(FLOWS)
    assert graph.edge_attr.shape[0] == graph.num_edges
    assert graph.edge_y.shape[0] == graph.num_edges

    print_graph(graph, endpoint_to_id)

if __name__ == "__main__":
    main()
