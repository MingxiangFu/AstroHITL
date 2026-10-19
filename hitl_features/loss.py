"""Strength-weighted supervised contrastive loss."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from hitl_features.graph import CategoryGraph

TEMPERATURE = 0.1


def build_pair_targets(
    graph: CategoryGraph, sample_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive mask and negative weights, both shape (N, N).

    A negative weight of 0 leaves the pair out of the loss. Ancestor and
    descendant assignments are neither positives nor negatives.
    """
    count = len(sample_ids)
    positive = np.zeros((count, count), dtype=np.float32)
    negative = np.zeros((count, count), dtype=np.float32)
    categories = [graph.category_of(sample_id) for sample_id in sample_ids]
    for i in range(count):
        for j in range(count):
            if i == j or categories[i] is None or categories[j] is None:
                continue
            if categories[i] == categories[j]:
                positive[i, j] = 1.0
                continue
            weight = graph.divergence_weight(categories[i], categories[j])
            if weight is None or weight == 0.0:
                continue
            negative[i, j] = float(weight)
    return positive, negative


def weighted_supcon_loss(
    embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_weight: torch.Tensor,
    temperature: float = TEMPERATURE,
) -> torch.Tensor:
    """Mean over positive pairs of -log(exp(sim) / (positives + weighted negatives))."""
    vectors = F.normalize(embeddings, dim=-1)
    count = vectors.shape[0]
    similarity = vectors @ vectors.T / float(temperature)
    eye = torch.eye(count, device=vectors.device, dtype=torch.bool)
    similarity = similarity.masked_fill(eye, -1e9)
    stable = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(stable).masked_fill(eye, 0.0)
    positive = positive_mask.to(dtype=exp_sim.dtype)
    negative = negative_weight.to(dtype=exp_sim.dtype)
    positive = positive.masked_fill(eye, 0.0)
    negative = negative.masked_fill(eye, 0.0)
    denom = (exp_sim * positive).sum(dim=1, keepdim=True) + (exp_sim * negative).sum(
        dim=1, keepdim=True
    )
    log_prob = stable - torch.log(denom.clamp_min(1e-12))
    weighted = -log_prob * positive
    pair_count = positive.sum()
    if float(pair_count) <= 0:
        return embeddings.sum() * 0.0
    return weighted.sum() / pair_count
