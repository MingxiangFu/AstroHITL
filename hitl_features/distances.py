"""Within-category and between-category cosine distances."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from hitl_features.space import l2_normalize


def mean_cross_distance(left: np.ndarray, right: np.ndarray) -> float | None:
    """Mean cosine distance between every pair from the two sets."""
    left_n = l2_normalize(left)
    right_n = l2_normalize(right)
    if len(left_n) == 0 or len(right_n) == 0:
        return None
    return float(1.0 - np.dot(left_n.sum(axis=0), right_n.sum(axis=0)) / (len(left_n) * len(right_n)))


def mean_within_distance(vectors: np.ndarray) -> float | None:
    """Mean cosine distance over unordered pairs inside one set."""
    matrix = l2_normalize(vectors)
    count = len(matrix)
    if count < 2:
        return None
    total = matrix.sum(axis=0)
    pair_sum = (float(total @ total) - count) / 2.0
    pairs = count * (count - 1) / 2.0
    return float(1.0 - pair_sum / pairs)


def mean_within_between(
    vectors: np.ndarray, labels: Sequence[str]
) -> tuple[float | None, float | None]:
    """Unweighted mean of per-category within distances, and mean between-category distance.

    Categories with fewer than two samples are skipped in the within term.
    They still take part in the between term. Returns None when a term has no pairs.
    """
    matrix = l2_normalize(np.asarray(vectors, dtype=np.float64))
    names = list(labels)
    count = len(names)
    if count == 0 or matrix.shape[0] != count:
        return None, None
    distance = 1.0 - matrix @ matrix.T
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        groups.setdefault(str(name), []).append(index)
    within_means: list[float] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        pair_values = [
            distance[members[i], members[j]]
            for i in range(len(members))
            for j in range(i + 1, len(members))
        ]
        within_means.append(float(np.mean(pair_values)))
    between: list[float] = []
    for i in range(count):
        for j in range(i + 1, count):
            if names[i] != names[j]:
                between.append(float(distance[i, j]))
    within = float(np.mean(within_means)) if within_means else None
    between_mean = float(np.mean(between)) if between else None
    return within, between_mean
