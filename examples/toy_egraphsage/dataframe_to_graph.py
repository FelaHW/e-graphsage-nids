"""Convert a pandas network-flow table into a PyTorch Geometric graph.

Input:
    A pandas DataFrame where every row is one network flow.

Output:
    A PyTorch Geometric Data object where:

    - node = (IP address, port)
    - edge = network flow
    - edge_attr = packets, bytes, duration
    - edge_y = benign or attack label
"""
from typing import TypeAlias

import pandas as pd 
import torch
from torch_geometric.data import Data

from load_flows_csv import TOY_DATASET_PATH, load_flow_csv

Endpoint:TypeAlias = tuple [str,int]

EDGE_FEATURES_COLUMNS = [
    "packets",
    "bytes",
    "duration"
]

REQUIRED_COLUMNS = {
    "source_ip",
    "source_port",
    "destination_ip",
    "destination_port",
    "packets",
    "bytes",
    "duration",
    "label"
}

def validate_flow_table(flows:pd.DataFrame)->None:
    missing_columns = REQUIRED_COLUMNS- set(flows.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    if flows.empty:
        raise ValueError("The flow dataset is empty.")

def create_endpoint_mapping(flows:pd.DataFrame)-> dict[Endpoint, int]:

    endpoint_to_id : dict[Endpoint,int] = {}

    for flow in flows.itertuples(index=False):
        source_endpoint : Endpoint = (str(flow.source_ip),int(flow.source_port))
        destination_endpoint : Endpoint = (str(flow.destination_ip),int(flow.destination_port))

        if source_endpoint not in endpoint_to_id:
            endpoint_to_id[source_endpoint] = len(endpoint_to_id)
        if destination_endpoint not in endpoint_to_id:
            endpoint_to_id[destination_endpoint] = len(endpoint_to_id)

    return endpoint_to_id

def dataframe_to_graph(flows: pd.DataFrame,) -> tuple[Data,dict[Endpoint,int]]:

    validate_flow_table(flows)
    endpoint_to_id = create_endpoint_mapping(flows)

    source_node_ids: list[int] = []
    destination_node_ids:list[int] = []
    edge_features : list[list[float]] = []
    edge_labels : list[int] = []

    for flow in flows.itertuples(index=False):
        source_endpoint:Endpoint = (str(flow.source_ip),int(flow.source_port))
        destination_endpoint:Endpoint = (str(flow.destination_ip),int(flow.destination_port))

        source_node_ids.append(endpoint_to_id[source_endpoint])
        destination_node_ids.append(endpoint_to_id[destination_endpoint])
        edge_features.append(
            [
            float(flow.packets),
            float(flow.bytes),
            float(flow.duration)
            ]
        )
        edge_labels.append(int(flow.label))

    edge_index = torch.tensor([source_node_ids,destination_node_ids,],dtype = torch.long,)
    edge_attr = torch.tensor(edge_features,dtype = torch.float32)
    edge_y = torch.tensor(edge_labels,dtype = torch.long)

    number_of_nodes = len(endpoint_to_id)
    number_of_node_features = edge_attr.shape[1]

    node_features = torch.ones((number_of_nodes,number_of_node_features,),dtype=torch.float32,)
    graph = Data(
        x=node_features,
        edge_index = edge_index,
        edge_attr = edge_attr,
        edge_y = edge_y
    )

    return graph, endpoint_to_id

def main() -> None:
    flows = load_flow_csv(TOY_DATASET_PATH)
    graph,endpoint_to_id = dataframe_to_graph(flows)

    print("Graph created from DataFrame:")
    print("-----------------------------")

    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Number of edges: {graph.num_edges}")
    print(f"Node feature shape: {graph.x.shape}")
    print(f"Edge index shape: {graph.edge_index.shape}")
    print(f"Edge feature shape: {graph.edge_attr.shape}")
    print(f"Edge label shape: {graph.edge_y.shape}")

    print("\nEndpoint mapping:")

    for endpoint, node_id in sorted(
        endpoint_to_id.items(),
        key=lambda item: item[1],
    ):
        print(f"Node {node_id}: {endpoint}")

    print("\nedge_index:")
    print(graph.edge_index)

    print("\nedge_attr:")
    print(graph.edge_attr)

    print("\nedge_y:")
    print(graph.edge_y)

    assert graph.num_edges == len(flows)
    assert graph.edge_attr.shape[0] == len(flows)
    assert graph.edge_y.shape[0] == len(flows)

if __name__=="__main__":
    main()

