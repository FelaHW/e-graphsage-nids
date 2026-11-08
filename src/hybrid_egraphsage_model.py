"""Direct target-flow features plus target-excluded endpoint context."""

import torch
from torch import nn
from torch_geometric.data import Data

from loeo_egraphsage_model import aggregate_without_target


class HybridEGraphSAGE(nn.Module):
    """Combine a direct flow path with the existing LOEO context pipeline."""

    def __init__(
        self,
        node_feature_size: int,
        edge_feature_size: int,
        hidden_size: int = 32,
        classifier_hidden_size_1: int = 32,
        classifier_hidden_size_2: int = 16,
        number_of_classes: int = 2,
    ) -> None:
        super().__init__()
        self.node_layer = nn.Linear(node_feature_size + edge_feature_size, hidden_size)
        # Hidden widths match the MLP's classifier, so direct target
        # features have a nonlinear path even when both contexts are empty.
        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_feature_size + 2 * hidden_size, classifier_hidden_size_1),
            nn.ReLU(),
            nn.Linear(classifier_hidden_size_1, classifier_hidden_size_2),
            nn.ReLU(),
            nn.Linear(classifier_hidden_size_2, number_of_classes),
        )

    def context_embeddings(self, graph: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Return target-specific embeddings; own features remain excluded."""
        source_ids, destination_ids = graph.edge_index
        source_means, destination_means = aggregate_without_target(graph)
        source_inputs = torch.cat([graph.x[source_ids], source_means], dim=1)
        destination_inputs = torch.cat(
            [graph.x[destination_ids], destination_means], dim=1
        )
        return (
            torch.relu(self.node_layer(source_inputs)),
            torch.relu(self.node_layer(destination_inputs)),
        )

    def forward(self, graph: Data) -> torch.Tensor:
        source_context, destination_context = self.context_embeddings(graph)
        # Eight direct values + 32 source values + 32 destination values.
        # Only this explicit direct path exposes the target's own features.
        classifier_inputs = torch.cat(
            [graph.edge_attr, source_context, destination_context], dim=1
        )
        return self.edge_classifier(classifier_inputs)
