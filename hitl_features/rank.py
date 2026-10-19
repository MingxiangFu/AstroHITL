"""Scalar ranker with a strength-weighted pairwise margin."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hitl_features.modelsizes import normalize_hidden


class ScoreNet(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, int] = (64, 32)) -> None:
        super().__init__()
        self.hidden = normalize_hidden(hidden)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), self.hidden[0]),
            nn.ReLU(),
            nn.Linear(self.hidden[0], self.hidden[1]),
            nn.ReLU(),
            nn.Linear(self.hidden[1], 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class Ranker:
    def __init__(self, net: ScoreNet) -> None:
        self.net = net.cpu().eval()

    def score(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float32)
        with torch.no_grad():
            values = self.net(torch.from_numpy(matrix))
        return values.cpu().numpy()


def fit_ranker(
    features: np.ndarray,
    groups: list[str],
    order: list[str],
    strength: float,
    *,
    seed: int,
    epochs: int = 40,
    lr: float = 1e-3,
    device: str = "cpu",
    margin: float = 1.0,
    hidden: tuple[int, int] | None = None,
    on_progress=None,
) -> Ranker:
    if float(strength) <= 0:
        raise ValueError("preference strength must be positive")
    matrix = np.asarray(features, dtype=np.float32)
    if len(groups) != len(matrix):
        raise ValueError("groups must align with features")
    rank = {name: index for index, name in enumerate(order)}
    if any(group not in rank for group in groups):
        raise ValueError("every group must appear in the preference order")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    net = ScoreNet(matrix.shape[1], hidden=normalize_hidden(hidden)).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    inputs = torch.from_numpy(matrix).to(device)
    preference = torch.tensor([rank[group] for group in groups], dtype=torch.float32, device=device)
    gap_target = float(strength) * float(margin)
    net.train()
    total = int(epochs)
    if on_progress is not None and total > 0:
        on_progress(0, total)
    for step in range(1, total + 1):
        optimizer.zero_grad()
        scores = net(inputs)
        gap = scores[:, None] - scores[None, :]
        mask = preference[:, None] < preference[None, :]
        if not bool(mask.any()):
            raise ValueError("ranking needs at least two preference groups")
        hinge = torch.relu(gap_target - gap)
        loss = hinge[mask].mean()
        loss.backward()
        optimizer.step()
        if on_progress is not None:
            on_progress(step, total)
    return Ranker(net)


def pairwise_preference_accuracy(
    scores: np.ndarray, groups: list[str], order: list[str]
) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    rank = {name: index for index, name in enumerate(order)}
    correct = 0
    total = 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if groups[i] not in rank or groups[j] not in rank:
                continue
            if rank[groups[i]] == rank[groups[j]]:
                continue
            total += 1
            if rank[groups[i]] < rank[groups[j]]:
                preferred = values[i] > values[j]
            else:
                preferred = values[j] > values[i]
            correct += int(preferred)
    if total == 0:
        return 0.0
    return correct / total
