"""Recent and representative thumbnails for one category card."""

from __future__ import annotations

import numpy as np

from hitl_features.space import cosine_distance

RECENT_N = 2
REPRESENTATIVE_M = 3


def category_preview(
    indices: list[int],
    vectors: np.ndarray,
    orders: list[int],
    *,
    n: int = RECENT_N,
    m: int = REPRESENTATIVE_M,
) -> dict:
    """Return recent and representative sample ids from one class.

    Recent ids are the latest assignments. Representatives are drawn from the
    other members when any remain: high mean cosine similarity inside the
    class, then a greedy farthest-first pick. An empty side is an empty list.
    """
    sample_ids = [int(value) for value in indices]
    if not sample_ids:
        return {"recent": [], "representative": []}
    matrix = np.asarray(vectors, dtype=np.float64)
    times = [int(value) for value in orders]
    if len(sample_ids) != len(matrix) or len(sample_ids) != len(times):
        raise ValueError("preview indices, vectors, and orders must align")
    recent_pos = sorted(
        range(len(sample_ids)),
        key=lambda i: (times[i], sample_ids[i]),
        reverse=True,
    )[: max(int(n), 0)]
    recent = [sample_ids[i] for i in recent_pos]
    recent_set = set(recent)
    rest_pos = [i for i in range(len(sample_ids)) if sample_ids[i] not in recent_set]
    if not rest_pos or int(m) <= 0:
        return {"recent": recent, "representative": []}
    representative = _representatives(
        [sample_ids[i] for i in rest_pos],
        matrix[rest_pos],
        matrix,
        sample_ids,
        int(m),
    )
    return {"recent": recent, "representative": representative}


def _mean_similarity(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) <= 1:
        return np.ones(len(vectors), dtype=np.float64)
    similarity = 1.0 - cosine_distance(vectors, vectors)
    return (similarity.sum(axis=1) - 1.0) / (len(vectors) - 1)


def _representatives(
    rest_ids: list[int],
    rest_vectors: np.ndarray,
    class_vectors: np.ndarray,
    class_ids: list[int],
    m: int,
) -> list[int]:
    mean_all = _mean_similarity(class_vectors)
    mean_of = {class_ids[i]: float(mean_all[i]) for i in range(len(class_ids))}
    ranked = sorted(range(len(rest_ids)), key=lambda i: (-mean_of[rest_ids[i]], rest_ids[i]))
    pool_size = min(len(ranked), max(3 * m, 12))
    pool = ranked[:pool_size]
    if len(pool) == 1 or m == 1:
        return [rest_ids[pool[0]]]
    distances = cosine_distance(rest_vectors, rest_vectors)
    selected = [pool[0]]
    while len(selected) < min(m, len(pool)):
        best = None
        best_key = None
        chosen = set(selected)
        for index in pool:
            if index in chosen:
                continue
            nearest = min(float(distances[index, other]) for other in selected)
            key = (nearest, -rest_ids[index])
            if best_key is None or key > best_key:
                best_key = key
                best = index
        selected.append(int(best))
    return [rest_ids[index] for index in selected]
