"""Fixed portable subset of the catalog."""

from __future__ import annotations

import zipfile
from pathlib import Path

import h5py
import numpy as np

SUBSET_SIZE = 10000
SUBSET_SEED = 0


def choose_subset_indices(
    catalog_indices,
    required_indices,
    *,
    size: int = SUBSET_SIZE,
    seed: int = SUBSET_SEED,
) -> list[int]:
    """Keep every required catalog id, then fill to ``size`` with a fixed seed.

    The result is sorted and still uses the original catalog ids.
    """
    catalog = [int(value) for value in catalog_indices]
    present = set(catalog)
    required = sorted({int(value) for value in required_indices if int(value) in present})
    if len(required) > int(size):
        raise ValueError(f"{len(required)} required ids do not fit in a subset of {size}")
    required_set = set(required)
    remaining = np.array([value for value in catalog if value not in required_set], dtype=np.int64)
    need = int(size) - len(required)
    if need > len(remaining):
        raise ValueError("catalog does not contain enough rows to fill the subset")
    rng = np.random.default_rng(int(seed))
    if need:
        extra = rng.choice(remaining, size=need, replace=False)
        chosen = required + [int(value) for value in np.asarray(extra).reshape(-1)]
    else:
        chosen = list(required)
    return sorted(chosen)


def write_feature_table(path: Path | str, matrix: np.ndarray, indices) -> None:
    """Write features and catalog ids with no HDF5 compression filter."""
    features = np.asarray(matrix, dtype=np.float32)
    ids = np.asarray([int(value) for value in indices], dtype=np.int64).reshape(-1)
    if features.shape[0] != ids.shape[0]:
        raise ValueError("features and catalog ids must have the same length")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as handle:
        feature_set = handle.create_dataset("mae_features", data=features)
        index_set = handle.create_dataset("index", data=ids)
        if feature_set.compression is not None or index_set.compression is not None:
            raise RuntimeError("subset features must be stored without a compression filter")


def write_stored_zip(path: Path | str, members: list[tuple[str, Path]]) -> None:
    """Pack files in one zip archive without compressing the members."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as handle:
        for arcname, source in members:
            handle.write(Path(source), arcname=arcname)
