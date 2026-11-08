"""A small E-GraphSAGE-style model for binary flow classification.

This follows the working pipeline in toy_egraphsage.py, using tensor
accumulation instead of a Python loop over every edge. It is an educational
one-layer model, not a complete reproduction of the paper.

Batch evaluation assumptions:
- The target edge contributes its own features to its endpoint means.
- Other edges in the same split also contribute to those means.
- Edge labels are never read by the model or aggregation function.
"""

import torch
from torch import nn
from torch_geometric.data import Data


def aggregate_connected_edges(graph: Data) -> torch.Tensor:
    """Average incoming and outgoing edge features at each endpoint.

    As in the toy implementation, each edge contributes once at its source
    and once at its destination. A self-loop therefore contributes twice.
    Isolated nodes receive zeros rather than an undefined mean.
    """
    source_ids, destination_ids = graph.edge_index
    edge_features = graph.edge_attr
    number_of_nodes = graph.num_nodes
    number_of_features = edge_features.shape[1]

    # new_zeros preserves the feature tensor's dtype and device.
    node_feature_sums = edge_features.new_zeros(
        (number_of_nodes, number_of_features)
    )
    node_edge_counts = edge_features.new_zeros((number_of_nodes, 1))
    one_per_edge = edge_features.new_ones((graph.num_edges, 1))

    # index_add_ adds each row to the node named by its endpoint ID.
    # Repeated endpoint IDs accumulate, including parallel flow edges.
    node_feature_sums.index_add_(0, source_ids, edge_features)
    node_feature_sums.index_add_(0, destination_ids, edge_features)
    node_edge_counts.index_add_(0, source_ids, one_per_edge)
    node_edge_counts.index_add_(0, destination_ids, one_per_edge)

    return node_feature_sums / node_edge_counts.clamp(min=1)


class EGraphSAGE(nn.Module):
    """Learn endpoint embeddings, then classify each directed flow edge."""

    def __init__(
        self,
        node_feature_size: int,
        edge_feature_size: int,
        hidden_size: int = 32,
        number_of_classes: int = 2,
    ) -> None:
        super().__init__()
        self.node_layer = nn.Linear(
            node_feature_size + edge_feature_size, hidden_size
        )
        self.edge_classifier = nn.Linear(hidden_size * 2, number_of_classes)

    def forward(self, graph: Data) -> torch.Tensor:
        """Return [number_of_edges, 2] logits without using edge labels."""
        neighborhood_features = aggregate_connected_edges(graph)

        # Eight initial node values + eight neighborhood means -> 16 values.
        combined_node_features = torch.cat(
            [graph.x, neighborhood_features], dim=1
        )
        node_embeddings = torch.relu(self.node_layer(combined_node_features))

        # Keeping source and destination in separate positions preserves
        # their roles when building the representation of a directed edge.
        source_ids, destination_ids = graph.edge_index
        edge_embeddings = torch.cat(
            [node_embeddings[source_ids], node_embeddings[destination_ids]],
            dim=1,
        )
        return self.edge_classifier(edge_embeddings)
