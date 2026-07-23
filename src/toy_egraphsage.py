"""A small educational E-GraphSAGE-style model.

This is not yet the complete model from the research paper.

The purpose of this file is to understand the model pipeline:

1. Aggregate edge features around every node.
2. Combine the aggregation with the node's current features.
3. Create a learned embedding for each node.
4. Combine source and destination node embeddings.
5. Produce classification scores for every edge.
"""
import torch
from torch import nn
from torch_geometric.data import Data

from build_toy_graph import FLOWS, build_graph

def aggregate_connected_edges(graph: Data) -> torch.Tensor:
    number_of_nodes = graph.num_nodes
    number_of_features = graph.edge_attr.shape[1]

    #store the sum of connected edge features for each node
    node_feature_sums = torch.zeros(
        (number_of_nodes, number_of_features),
        dtype=torch.float32,
        )
    #store how many edges are connected to each node.
    node_edge_counts= torch.zeros(
        (number_of_nodes, 1),
        dtype=torch.float32,
        )

    for edge_id in range(graph.num_edges):
        source_id = int(graph.edge_index[0, edge_id])
        destination_id = int(graph.edge_index[1, edge_id])
        edge_features = graph.edge_attr[edge_id]

        node_feature_sums[source_id] += edge_features
        node_feature_sums[destination_id] += edge_features

        node_edge_counts[source_id]+=1
        node_edge_counts[destination_id]+=1

    #avoid division by zero
    safe_counts = node_edge_counts.clamp(min=1)

    mean_features = node_feature_sums / safe_counts

    return mean_features
class ToyEGraphSAGE(nn.Module):
    """A simplified edge-classification graph neural network model."""
    def __init__(self, edge_feature_size: int, hidden_size: int, number_of_classes: int) -> None:
        super().__init__()
        self.node_layer = nn.Linear(edge_feature_size*2, hidden_size)
        self.edge_classifier = nn.Linear(hidden_size*2, number_of_classes)

    def forward(self, graph: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """run the graph though the model."""

       # Step 1:
        # Summarize all connected flows for every node.
        neighborhood_features = aggregate_connected_edges(graph)

        # Step 2:
        # Combine original node features and neighborhood features.
        #
        # dim=1 means concatenate the columns.
        combined_node_features = torch.cat(
            [graph.x, neighborhood_features],
            dim=1,
        )

        # Step 3:
        # Transform the combined information into learned node embeddings.
        #
        # ReLU replaces negative values with zero.
        node_embeddings = torch.relu(
            self.node_layer(combined_node_features)
        )

        # Get the source and destination node IDs of all edges.
        source_ids = graph.edge_index[0]
        destination_ids = graph.edge_index[1]

        # Select the embedding belonging to each edge's source node.
        source_embeddings = node_embeddings[source_ids]

        # Select the embedding belonging to each edge's destination node.
        destination_embeddings = node_embeddings[destination_ids]

        # Step 4:
        # Combine the endpoint embeddings to create an edge representation.
        edge_embeddings = torch.cat(
            [source_embeddings, destination_embeddings],
            dim=1,
        )

        # Step 5:
        # Produce two scores for each edge:
        #
        # score 0 = benign
        # score 1 = attack
        #
        # These raw scores are called logits.
        edge_logits = self.edge_classifier(edge_embeddings)

        return edge_logits, node_embeddings, neighborhood_features


def main() -> None:
    # Using a fixed seed makes the random model initialization repeatable.
    torch.manual_seed(42)

    graph, _ = build_graph(FLOWS)

    model = ToyEGraphSAGE(
        edge_feature_size=graph.edge_attr.shape[1],
        hidden_size=8,
        number_of_classes=2,
    )

    edge_logits, node_embeddings, neighborhood_features = model(graph)

    print("Neighborhood feature shape:")
    print(neighborhood_features.shape)

    print("\nNode embedding shape:")
    print(node_embeddings.shape)

    print("\nEdge classification-score shape:")
    print(edge_logits.shape)

    print("\nRaw edge classification scores:")
    print(edge_logits)

    print("\nCorrect edge labels:")
    print(graph.edge_y)

    print(
        "\nImportant: the model has not been trained yet, "
        "so its classification scores are still random."
    )


if __name__ == "__main__":
    main() 
      

