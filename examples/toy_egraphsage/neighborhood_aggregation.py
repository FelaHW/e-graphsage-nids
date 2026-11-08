"""Aggregate all flow features connected to each node.

Unlike the previous exercise, this version includes:

- outgoing flows: the node is the source
- incoming flows: the node is the destination

Together, these flows form the node's local neighborhood.
"""

from build_toy_graph import FLOWS, build_graph

def main() -> None:
    graph, endpoint_to_id = build_graph(FLOWS)
    id_to_endpoint = {
        node_id: endpoint
        for endpoint,node_id in endpoint_to_id.items()
        }
    print("Mean features of all flows connected to each node:")
    print("--------------------------------------------------")

    for node_id in range(graph.num_nodes):
        outgoing_mask = graph.edge_index[0] == node_id
        incoming_mask = graph.edge_index[1] == node_id
        connected_mask = outgoing_mask | incoming_mask
        connected_features = graph.edge_attr[connected_mask]
        endpoint = id_to_endpoint[node_id]

        if connected_features.shape[0] == 0:
            print(f"Node {node_id} ({endpoint})")
            print("No connected flows")
            continue
        mean_features = connected_features.mean(dim=0)

        print(f"\nNode {node_id} ({endpoint})")
        print("Connected flow features:")
        print(connected_features)
        print("Mean neighborhood features:")
        print(mean_features)
if __name__ == "__main__":
    main()