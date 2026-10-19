"""Cosine geometry on raw or mapped vectors."""

from __future__ import annotations

import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance, range 0 to 2. Shape (len(left), len(right))."""
    if len(right) == 0:
        return np.zeros((len(left), 0), dtype=np.float64)
    if len(left) == 0:
        width = np.asarray(right).shape[-1]
        return np.zeros((0, len(right)), dtype=np.float64) if width else np.zeros((0, 0))
    return 1.0 - l2_normalize(left) @ l2_normalize(right).T
