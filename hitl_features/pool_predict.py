"""Fast category guesses for the current search pool.

Predictions use cosine kNN on L2-normalized mapped features. They are only
defined for the rows already loaded with the session. The full catalog is not
read.
"""

from __future__ import annotations

import numpy as np

from hitl_features.graph import CategoryGraph
from hitl_features.space import l2_normalize

PREDICT_K = 15


def predict_pool(
    graph: CategoryGraph,
    pool_indices: list[int],
    mapped_features: np.ndarray,
    k: int = PREDICT_K,
) -> list[dict]:
    """Return root and child guesses for pool rows that still need a label.

    Each record is ``index`` (catalog index), ``category_id``, and ``level``
    (``root`` or ``child``). A row with no assignment is compared with rows
    that already have a root. A row assigned to a parent, but not to one of
    its children, is compared with rows under that same parent that already
    have a child. Ties follow the nearest neighbor.
    """
    indices = [int(value) for value in pool_indices]
    vectors = np.asarray(mapped_features, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != len(indices) or not indices:
        return []
    row_of = {index: row for row, index in enumerate(indices)}
    records: list[dict] = []
    records.extend(_root_predictions(graph, indices, vectors, row_of, int(k)))
    records.extend(_child_predictions(graph, vectors, row_of, int(k)))
    records.sort(key=lambda item: (item["index"], 0 if item["level"] == "root" else 1))
    return records


def _root_predictions(graph, indices, vectors, row_of, k: int) -> list[dict]:
    labeled = [
        index
        for index in indices
        if graph.category_of(index) is not None and not graph.is_recycled_sample(index)
    ]
    missing = [index for index in indices if graph.category_of(index) is None]
    if not labeled or not missing:
        return []
    labels = [_root_id(graph, graph.category_of(index)) for index in labeled]
    voted = _vote(
        vectors[[row_of[index] for index in missing]],
        vectors[[row_of[index] for index in labeled]],
        labels,
        k,
    )
    return [
        {"index": index, "category_id": category_id, "level": "root"}
        for index, category_id in zip(missing, voted)
    ]


def _child_predictions(graph, vectors, row_of, k: int) -> list[dict]:
    records = []
    for parent in list(graph.categories.values()):
        if not graph.children(parent.id):
            continue
        queries = [index for index in graph.samples(parent.id) if index in row_of]
        refs: list[int] = []
        labels: list[str] = []
        for index, owner in graph.assignments.items():
            if index not in row_of or owner == parent.id or graph.is_recycled_sample(index):
                continue
            child_id = _child_under(graph, parent.id, owner)
            if child_id is None:
                continue
            refs.append(index)
            labels.append(child_id)
        if not queries or not refs:
            continue
        voted = _vote(
            vectors[[row_of[index] for index in queries]],
            vectors[[row_of[index] for index in refs]],
            labels,
            k,
        )
        for index, category_id in zip(queries, voted):
            records.append({"index": index, "category_id": category_id, "level": "child"})
    return records


def _root_id(graph: CategoryGraph, category_id: str) -> str:
    return graph.path(category_id)[0]


def _child_under(graph: CategoryGraph, parent_id: str, category_id: str) -> str | None:
    path = graph.path(category_id)
    if parent_id not in path:
        return None
    position = path.index(parent_id)
    if position + 1 >= len(path):
        return None
    return path[position + 1]


def _vote(query: np.ndarray, refs: np.ndarray, labels: list[str], k: int) -> list[str]:
    if len(labels) == 0 or len(query) == 0:
        return []
    similarity = l2_normalize(query) @ l2_normalize(refs).T
    width = max(1, min(int(k), similarity.shape[1]))
    if width == similarity.shape[1]:
        nearest = np.argsort(-similarity, axis=1)
    else:
        partial = np.argpartition(-similarity, width - 1, axis=1)[:, :width]
        rows = np.arange(similarity.shape[0])[:, None]
        nearest = partial[rows, np.argsort(-similarity[rows, partial], axis=1)]
    chosen = []
    for row in nearest:
        ordered = [labels[int(column)] for column in row[:width]]
        chosen.append(_majority(ordered))
    return chosen


def _majority(labels: list[str]) -> str:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    best = max(counts.values())
    tied = {label for label, count in counts.items() if count == best}
    for label in labels:
        if label in tied:
            return label
    return labels[0]
