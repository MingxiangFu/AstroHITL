"""Continue one feature mapper on the current graph."""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim

from hitl_features.graph import CategoryGraph
from hitl_features.loss import TEMPERATURE, build_pair_targets, weighted_supcon_loss
from hitl_features.mapper import FeatureMapper


def train_mapper(
    mapper: FeatureMapper,
    features: np.ndarray,
    sample_ids: list[int],
    graph: CategoryGraph,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
    temperature: float = TEMPERATURE,
    on_progress=None,
) -> dict:
    """Train ``mapper`` in place. ``epochs <= 0`` leaves the weights unchanged."""
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.shape[0] != len(sample_ids):
        raise ValueError("features and sample ids must have the same length")
    if int(epochs) <= 0:
        return {"loss": None, "history": [], "steps": 0}
    positive, negative = build_pair_targets(graph, list(sample_ids))
    if float(positive.sum()) <= 0:
        return {"loss": None, "history": [], "steps": 0}

    mapper.to(device)
    mapper.train()
    optimizer = optim.AdamW(mapper.parameters(), lr=lr, weight_decay=1e-5)
    inputs = torch.from_numpy(matrix).to(device)
    positive_t = torch.from_numpy(positive).to(device)
    negative_t = torch.from_numpy(negative).to(device)
    history: list[float] = []
    total = int(epochs)
    if on_progress is not None:
        on_progress(0, total)
    for step in range(1, total + 1):
        optimizer.zero_grad()
        embeddings = mapper(inputs)
        loss = weighted_supcon_loss(
            embeddings, positive_t, negative_t, temperature=temperature
        )
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
        if on_progress is not None:
            on_progress(step, total)
    return {"loss": history[-1], "history": history, "steps": total}
