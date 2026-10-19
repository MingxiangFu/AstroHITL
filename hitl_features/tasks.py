"""Task classes on the right of the graph."""

from __future__ import annotations

import numpy as np

from hitl_features.graph import CategoryGraph


def task_labels(
    graph: CategoryGraph, classes: list[tuple[str, list[str]]]
) -> dict[int, str]:
    """Map sample index to a task-class name.

    A category cannot belong to two classes. When a parent and a child are
    assigned to different classes, a sample uses the deeper assignment.
    Samples outside every assigned category are omitted.
    """
    category_to_class: dict[str, str] = {}
    for class_name, category_ids in classes:
        for category_id in category_ids:
            if category_id not in graph.categories:
                raise KeyError(f"unknown category: {category_id}")
            if graph.is_recycled_category(category_id):
                raise ValueError("the recycle bin is not a task class")
            if category_id in category_to_class:
                raise ValueError(f"category {category_id} is in two task classes")
            category_to_class[category_id] = class_name

    labels: dict[int, str] = {}
    for sample_index, direct in graph.assignments.items():
        if graph.is_recycled_sample(sample_index):
            continue
        chosen = None
        for category_id in graph.path(direct):
            if category_id in category_to_class:
                chosen = category_to_class[category_id]
        if chosen is not None:
            labels[int(sample_index)] = chosen
    return labels


def normalize_fractions(train: float, val: float, test: float) -> tuple[float, float, float]:
    """Train must be positive. Val and test may be zero. The three sum to 1."""
    values = (float(train), float(val), float(test))
    if any(not np.isfinite(value) for value in values):
        raise ValueError("fractions must be finite")
    if values[0] <= 0:
        raise ValueError("train fraction must be greater than 0")
    if values[1] < 0 or values[2] < 0:
        raise ValueError("val and test fractions must be at least 0")
    if abs(sum(values) - 1.0) > 1e-4:
        raise ValueError("train, val, and test fractions must sum to 1")
    return values


def fraction_counts(n: int, fractions: tuple[float, float, float]) -> list[int]:
    """Largest-remainder counts. Ties give the extra item to the earlier fraction."""
    size = int(n)
    if size < 0:
        raise ValueError("class size must be non-negative")
    raw = [size * float(part) for part in fractions]
    base = [int(np.floor(value)) for value in raw]
    left = size - sum(base)
    order = sorted(range(3), key=lambda i: (raw[i] - base[i], -i), reverse=True)
    for step in range(left):
        base[order[step]] += 1
    return base


def stratified_split(
    groups: dict[str, list[int]],
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, dict[str, list[int]]]:
    """Split each class independently at the same train/val/test fractions."""
    train_f, val_f, test_f = normalize_fractions(*fractions)
    fracs = (train_f, val_f, test_f)
    rng = np.random.default_rng(int(seed))
    names = ("train", "val", "test")
    result: dict[str, dict[str, list[int]]] = {}
    for name in sorted(groups):
        ids = np.array([int(value) for value in groups[name]], dtype=np.int64)
        rng.shuffle(ids)
        counts = fraction_counts(len(ids), fracs)
        cursor = 0
        split: dict[str, list[int]] = {}
        for label, count in zip(names, counts):
            split[label] = sorted(int(value) for value in ids[cursor : cursor + count].tolist())
            cursor += count
        result[str(name)] = split
    return result


def inference_ids(member_ids, train_ids) -> list[int]:
    """Members of the task categories, excluding the labeled training images."""
    blocked = {int(value) for value in train_ids}
    return sorted(int(value) for value in member_ids if int(value) not in blocked)
