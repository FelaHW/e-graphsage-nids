"""Inspect how PyTorch Geometric stores the graph"""

from build_toy_graph import FLOWS, build_graph

def main() -> None:
    graph, endpoint_to_id = build_graph(FLOWS)

    print("Endpoint to ID mapping:")
    print("------------------------------")

    # sort by node id so that the graph is easier to read
    for endpoint, node_id in sorted(endpoint_to_id.items(), key=lambda x: x[1]):
        print(f"Node {node_id}: {endpoint}")

    print()
    print("edge_index:")
    print(graph.edge_index)

    print()
    print("Source node IDs:")
    print(graph.edge_index[0])

    print()
    print("Destination node IDs:")
    print(graph.edge_index[1])

    print()
    print("edge features:")
    print(graph.edge_attr)

    print()
    print("edge labels:")
    print(graph.edge_y)

if __name__ == "__main__":
    main()
