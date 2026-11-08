"""Control model: unchanged hybrid layers with reassigned context pairs."""

import hashlib

import torch
from torch_geometric.data import Data

from hybrid_egraphsage_model import HybridEGraphSAGE


def permutation_seed(experiment_seed: int, split: str) -> int:
    """Derive a label-independent, split-specific 63-bit seed."""
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unknown split identity: {split}")
    payload = f"shuffled-context-v1|{experiment_seed}|{split}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (1 << 63)


def make_context_derangement(
    number_of_edges: int, experiment_seed: int, split: str
) -> torch.Tensor:
    """Draw permutations with a private RNG until there are no fixed points.

    The assignment is fixed across epochs. This function reads no graph,
    features, or labels, and never advances the model's global RNG stream.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(permutation_seed(experiment_seed, split))
    if number_of_edges < 0:
        raise ValueError("The number of edges cannot be negative.")
    if number_of_edges == 1:
        raise ValueError("A one-edge graph cannot have a derangement.")
    identity = torch.arange(number_of_edges)
    while True:
        permutation = torch.randperm(number_of_edges, generator=generator)
        if not (permutation == identity).any().item():
            return permutation


class ShuffledHybridEGraphSAGE(HybridEGraphSAGE):
    """Keep architecture/direct features intact; move each context pair.

    The inherited constructor and forward() are unchanged. Applying the
    shared context transformation before this permutation preserves exactly
    the real model's context vectors and gradients, reassigned as whole pairs.
    """

    def context_embeddings(self, graph: Data) -> tuple[torch.Tensor, torch.Tensor]:
        source_context, destination_context = super().context_embeddings(graph)
        permutation = getattr(graph, "context_permutation", None)
        if permutation is None or permutation.shape != (graph.num_edges,):
            raise ValueError("Attach one context donor index per edge before forwarding.")
        return source_context[permutation], destination_context[permutation]
