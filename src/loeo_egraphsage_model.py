"""Leave-one-edge-out (LOEO) ablation of the existing E-GraphSAGE model.

Each prediction uses endpoint neighborhoods with its own edge removed. Other
parallel and reverse edges remain neighbors. Labels are never model inputs.
The control model is inherited without modifying its layers or initialization.
"""

import torch
from torch_geometric.data import Data

from egraphsage_model import EGraphSAGE


def incident_edge_sums_and_counts(graph: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """Count both endpoint incidences, matching the control's self-loop rule."""
    source_ids, destination_ids = graph.edge_index
    edge_features = graph.edge_attr

    # Accumulate/subtract float32 inputs in float64 to reduce cancellation
    # error. Means are cast back before entering the unchanged float32 layers.
    accumulation_dtype = (
        torch.float64
        if edge_features.dtype in (torch.float16, torch.bfloat16, torch.float32)
        else edge_features.dtype
    )
    accumulation_features = edge_features.to(accumulation_dtype)
    node_sums = accumulation_features.new_zeros(
        (graph.num_nodes, edge_features.shape[1])
    )
    node_counts = torch.zeros(
        (graph.num_nodes, 1), dtype=torch.long, device=edge_features.device
    )
    one_per_edge = torch.ones(
        (graph.num_edges, 1), dtype=torch.long, device=edge_features.device
    )
    node_sums.index_add_(0, source_ids, accumulation_features)
    node_sums.index_add_(0, destination_ids, accumulation_features)
    node_counts.index_add_(0, source_ids, one_per_edge)
    node_counts.index_add_(0, destination_ids, one_per_edge)
    return node_sums, node_counts


def aggregate_without_target(graph: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source/destination means for each target edge, shape [E, F]."""
    source_ids, destination_ids = graph.edge_index
    node_sums, node_counts = incident_edge_sums_and_counts(graph)
    edge_features = graph.edge_attr.to(node_sums.dtype)

    # A self-loop contributes twice to the same node, so remove both
    # copies from both views of that endpoint when it is the target.
    target_multiplicity = 1 + (source_ids == destination_ids).long().unsqueeze(1)
    removed_features = target_multiplicity * edge_features
    source_sums = node_sums[source_ids] - removed_features
    destination_sums = node_sums[destination_ids] - removed_features
    source_counts = node_counts[source_ids] - target_multiplicity
    destination_counts = node_counts[destination_ids] - target_multiplicity

    source_means = source_sums / source_counts.clamp(min=1)
    destination_means = destination_sums / destination_counts.clamp(min=1)
    # Explicit zeros also eliminate any subtraction residue for empty context.
    source_means = source_means.masked_fill(source_counts == 0, 0)
    destination_means = destination_means.masked_fill(destination_counts == 0, 0)
    return (
        source_means.to(graph.edge_attr.dtype),
        destination_means.to(graph.edge_attr.dtype),
    )


class LOEOEGraphSAGE(EGraphSAGE):
    """Reuse the control's layers with target-specific endpoint embeddings.

    The inherited constructor preserves parameter names, dimensions, count,
    and seed-based initialization. Only the forward aggregation changes.
    """

    def forward(self, graph: Data) -> torch.Tensor:
        source_ids, destination_ids = graph.edge_index
        source_means, destination_means = aggregate_without_target(graph)

        # Each row now belongs to a target edge, not to a shared node.
        # Apply the same node layer separately to its two endpoint contexts.
        source_inputs = torch.cat([graph.x[source_ids], source_means], dim=1)
        destination_inputs = torch.cat(
            [graph.x[destination_ids], destination_means], dim=1
        )
        source_embeddings = torch.relu(self.node_layer(source_inputs))
        destination_embeddings = torch.relu(self.node_layer(destination_inputs))
        edge_embeddings = torch.cat(
            [source_embeddings, destination_embeddings], dim=1
        )
        # No direct target-feature shortcut is added to this classifier.
        return self.edge_classifier(edge_embeddings)
