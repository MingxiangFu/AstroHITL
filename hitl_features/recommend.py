"""Round recommenders on raw or mapped features."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from hitl_features.space import cosine_distance, l2_normalize

MECHANISMS = ("random", "diversity", "unknown", "certain", "uncertain", "algorithmic")
ROUND_SIZE = 12
KNN_K = 15


def with_subtree_target(
    labels: Sequence[str], target: str, subtree_ids: Sequence[str]
) -> list[str]:
    """Mark every label inside ``subtree_ids`` as the specified target."""
    allowed = {str(value) for value in subtree_ids}
    allowed.add(str(target))
    target_name = str(target)
    return [target_name if str(label) in allowed else str(label) for label in labels]


def recommend(
    candidate_ids: np.ndarray,
    candidate_vectors: np.ndarray,
    graph_ids: np.ndarray,
    graph_vectors: np.ndarray,
    graph_labels: Sequence[str] | None,
    mechanism: str,
    *,
    n: int = ROUND_SIZE,
    rng: np.random.Generator,
    n_categories: int = 0,
    round_index: int = 1,
    target_label: str | None = None,
    knn_k: int = KNN_K,
) -> list[int]:
    """Return up to ``n`` sample ids that are not already in the graph."""
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism: {mechanism}")
    if mechanism == "algorithmic":
        if int(round_index) <= 1 or int(n_categories) <= 0:
            mechanism = "random"
        elif int(n_categories) >= 2:
            mechanism = "unknown"
        else:
            mechanism = "diversity"

    ids = [int(value) for value in np.asarray(candidate_ids).reshape(-1).tolist()]
    vectors = np.asarray(candidate_vectors, dtype=np.float64)
    blocked = {int(value) for value in np.asarray(graph_ids).reshape(-1).tolist()}
    keep = [i for i, sample_id in enumerate(ids) if sample_id not in blocked]
    if not keep or int(n) <= 0:
        return []
    cand_ids = [ids[i] for i in keep]
    cand_vec = vectors[keep]
    take = min(int(n), len(cand_ids))

    if mechanism == "random":
        picked = rng.choice(len(cand_ids), size=take, replace=False)
        return [cand_ids[int(i)] for i in np.asarray(picked).reshape(-1)]
    if mechanism == "diversity":
        return _diversity(cand_ids, cand_vec, take)
    if mechanism == "unknown":
        graph_matrix = np.asarray(graph_vectors, dtype=np.float64)
        if graph_matrix.size == 0 or len(blocked) == 0:
            return _diversity(cand_ids, cand_vec, take)
        return _farthest_from_graph(cand_ids, cand_vec, graph_matrix, take)
    return _by_certainty(
        cand_ids,
        cand_vec,
        np.asarray(graph_vectors, dtype=np.float64),
        list(graph_labels or []),
        mechanism,
        int(knn_k),
        take,
    )


def _diversity(ids: list[int], vectors: np.ndarray, take: int) -> list[int]:
    """Greedy max-min. The first image is the lowest sample id."""
    first = min(range(len(ids)), key=lambda i: ids[i])
    selected = [first]
    if take == 1:
        return [ids[first]]
    distances = cosine_distance(vectors, vectors)
    while len(selected) < take:
        best = None
        best_key = None
        chosen = set(selected)
        for index in range(len(ids)):
            if index in chosen:
                continue
            nearest = min(float(distances[index, other]) for other in selected)
            key = (nearest, -ids[index])
            if best_key is None or key > best_key:
                best_key = key
                best = index
        selected.append(int(best))
    return [ids[index] for index in selected]


def _farthest_from_graph(
    ids: list[int], vectors: np.ndarray, graph_vectors: np.ndarray, take: int
) -> list[int]:
    distances = cosine_distance(vectors, graph_vectors)
    nearest = distances.min(axis=1)
    order = sorted(range(len(ids)), key=lambda i: (-float(nearest[i]), ids[i]))
    return [ids[index] for index in order[:take]]


def knn_class_probabilities(
    query: np.ndarray,
    refs: np.ndarray,
    labels: Sequence[str],
    k: int = KNN_K,
) -> tuple[list[str], np.ndarray]:
    """Class fractions among the ``k`` cosine neighbors.

    Rows of the returned matrix sum to 1 when at least one reference exists.
    Classes follow first-seen label order.
    """
    names = [str(value) for value in labels]
    classes = list(dict.fromkeys(names))
    query_matrix = np.asarray(query, dtype=np.float64)
    ref_matrix = np.asarray(refs, dtype=np.float64)
    if query_matrix.ndim == 1:
        query_matrix = query_matrix.reshape(1, -1)
    if len(classes) == 0 or len(query_matrix) == 0 or len(ref_matrix) == 0:
        return classes, np.zeros((len(query_matrix), len(classes)), dtype=np.float64)
    if len(names) != len(ref_matrix):
        raise ValueError("class labels must align with reference vectors")
    left = l2_normalize(query_matrix)
    right = l2_normalize(ref_matrix)
    similarity = left @ right.T
    width = max(1, min(int(k), similarity.shape[1]))
    if width == similarity.shape[1]:
        nearest = np.argsort(-similarity, axis=1)
    else:
        partial = np.argpartition(-similarity, width - 1, axis=1)[:, :width]
        rows = np.arange(similarity.shape[0])[:, None]
        nearest = partial[rows, np.argsort(-similarity[rows, partial], axis=1)]
    index_of = {name: index for index, name in enumerate(classes)}
    probabilities = np.zeros((len(query_matrix), len(classes)), dtype=np.float64)
    for row in range(len(query_matrix)):
        for column in nearest[row, :width]:
            probabilities[row, index_of[names[int(column)]]] += 1.0
        probabilities[row] /= float(width)
    return classes, probabilities


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Entropy divided by log(n_classes). One class, or an empty row, is 0."""
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] <= 1:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    logged = np.zeros_like(matrix)
    mask = matrix > 0
    logged[mask] = matrix[mask] * np.log(matrix[mask])
    entropy = -logged.sum(axis=1)
    return entropy / np.log(matrix.shape[1])


def certainty_order(sample_ids, probabilities: np.ndarray, mode: str) -> list[int]:
    """Rank ids by kNN class probabilities.

    ``certain`` puts the highest max-class probability first.
    ``uncertain`` puts the highest normalized entropy first, then a lower
    max-class probability, then a lower catalog index.
    """
    ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    matrix = np.asarray(probabilities, dtype=np.float64)
    if len(ids) == 0:
        return []
    if matrix.shape != (len(ids), matrix.shape[1] if matrix.ndim == 2 else 0):
        matrix = np.zeros((len(ids), 1), dtype=np.float64)
    max_probability = matrix.max(axis=1) if matrix.size else np.zeros(len(ids))
    if mode == "uncertain":
        entropy = normalized_entropy(matrix)
        order = np.lexsort((ids, max_probability, -entropy))
    else:
        order = np.lexsort((ids, -max_probability))
    return [int(ids[index]) for index in order]


def _by_member_distance(
    ids: list[int], vectors: np.ndarray, refs: np.ndarray, mode: str, take: int
) -> list[int]:
    """Rank by mean cosine distance to the labeled members of one class.

    One class makes every kNN probability 1 and every entropy 0, so certainty
    and uncertainty would otherwise share one catalog-index order.
    """
    distances = cosine_distance(vectors, refs)
    mean = distances.mean(axis=1)
    if mode == "uncertain":
        order = sorted(range(len(ids)), key=lambda i: (-float(mean[i]), ids[i]))
    else:
        order = sorted(range(len(ids)), key=lambda i: (float(mean[i]), ids[i]))
    return [ids[index] for index in order[:take]]


def _by_certainty(
    ids: list[int],
    vectors: np.ndarray,
    graph_vectors: np.ndarray,
    graph_labels: list[str],
    mode: str,
    knn_k: int,
    take: int,
) -> list[int]:
    if graph_vectors.size == 0 or len(graph_labels) == 0:
        return _diversity(ids, vectors, take)
    if len(graph_labels) != len(graph_vectors):
        raise ValueError("graph labels must align with graph vectors")
    if len(set(graph_labels)) < 2:
        return _by_member_distance(ids, vectors, graph_vectors, mode, take)
    _classes, probabilities = knn_class_probabilities(
        vectors, graph_vectors, graph_labels, knn_k
    )
    ranked = certainty_order(ids, probabilities, mode)
    return ranked[:take]
