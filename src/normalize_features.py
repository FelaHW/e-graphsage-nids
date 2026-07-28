"""Demonstrate feature standardization on the toy graph.

Network-flow features can have very different scales:

- packets may be around 10
- bytes may be around thousands
- duration may be close to zero

Standardization transforms every feature approximately to:

    mean = 0
    standard deviation = 1
"""

import torch

from build_toy_graph import build_graph, FLOWS

def standardize_features(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize every feature column separately.

    Formula:

        normalized_value = (value - mean) / standard_deviation

    Args:
        features:
            Tensor with shape:
            [number_of_edges, number_of_features]

    Returns:
        normalized_features:
            Standardized feature tensor.

        feature_means:
            Mean of every feature column.

        feature_standard_deviations:
            Standard deviation of every feature column.
    """

    feature_means = features.mean(dim=0)

    feature_standard_deviations = features.std(dim=0,unbiased=False)

    safe_standard_deviations = feature_standard_deviations.clamp(min=1e-6)

    normalized_features = (features - feature_means) / safe_standard_deviations

    return (normalized_features, feature_means, feature_standard_deviations)

def main() -> None:
    graph,_ = build_graph(FLOWS)

    normalized_features,means,standard_deviations = (standardize_features(graph.edge_attr))

    print("Original features:")
    print(graph.edge_attr)

    print("\nFeature means:")
    print(means)

    print("\nFeature standard deviations:")
    print(standard_deviations)

    print("\nNormalized edge features:")
    print(normalized_features)

    print("\nMean after normalization")
    print(normalized_features.mean(dim=0))

    print("\nStandard deviation after normalization")
    print(normalized_features.std(dim=0, unbiased=False))

if __name__ == "__main__":
    main()
    