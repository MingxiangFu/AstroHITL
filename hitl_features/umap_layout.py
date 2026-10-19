"""Two-dimensional layouts for the candidate pool."""

from __future__ import annotations

import numpy as np


def embed_points(matrix: np.ndarray, seed: int) -> np.ndarray:
    """Return one (x, y) row per sample.

    Pools smaller than four points use a fixed layout. Larger pools use UMAP.
    The coordinates are not a progress fraction, so the caller shows an
    indeterminate wait.
    """
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("umap input must be a matrix")
    count = int(values.shape[0])
    if count == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if count == 1:
        return np.zeros((1, 2), dtype=np.float32)
    if count < 4:
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, count - 1),
        min_dist=0.1,
        metric="euclidean",
        random_state=int(seed),
        n_jobs=1,
    )
    return np.asarray(reducer.fit_transform(values), dtype=np.float32)
