"""Calculate mean flow features for each source node.

This is a learning step before implementing the real GNN layer.

For each node, we:
1. Find all outgoing flows.
2. Collect their edge features.
3. Calculate the mean of those features.
"""
from build_toy_graph import FLOWS, build_graph

def main() -> None:
    graph,endpoint_to_id = build_graph(FLOWS)

    id_to_endpoint = {
        node_id: endpoint
        for endpoint,node_id in endpoint_to_id.items()
        }

    print("mean outgoing flow features for each source node:")
    print("------------------------------------------------------")

    # visit each node in the graph
    for node_id in range(graph.num_nodes):
        outgoing_mask = graph.edge_index[0] == node_id
        outgoing_features = graph.edge_attr[outgoing_mask]
        endpoint = id_to_endpoint[node_id]

        if outgoing_features.shape[0] == 0:
            print(f"Node {node_id} ({endpoint})")
            print("No outgoing flows")      
            continue
        mean_features = outgoing_features.mean(dim=0)
        print(f"\nNode{node_id}{endpoint}")
        print("outgoing flow features:")
        print(outgoing_features)
        print("mean features:")
        print(mean_features)

if __name__ == "__main__":
    main()