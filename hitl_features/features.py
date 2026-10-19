"""Partial reads of pretrained HDF5 embeddings."""

from __future__ import annotations

from typing import Sequence

import h5py
import numpy as np

FEATURE_KEYS = ("features", "embeddings", "mae_features")


def _feature_key(handle: h5py.File) -> str:
    for key in FEATURE_KEYS:
        if key in handle:
            return key
    names = ", ".join(FEATURE_KEYS)
    raise KeyError(f"No feature dataset among {names}")


def feature_matrix_shape(path: str) -> tuple[int, int]:
    with h5py.File(path, "r") as handle:
        dataset = handle[_feature_key(handle)]
        return int(dataset.shape[0]), int(dataset.shape[1])


def catalog_indices(path: str) -> np.ndarray:
    """Catalog id of each feature row.

    A subset file stores the original catalog id in ``index``. When that array
    is missing, or it is exactly ``0 .. n-1``, the row number is the catalog id.
    """
    with h5py.File(path, "r") as handle:
        dataset = handle[_feature_key(handle)]
        n_rows = int(dataset.shape[0])
        if "index" not in handle:
            return np.arange(n_rows, dtype=np.int64)
        values = np.asarray(handle["index"], dtype=np.int64).reshape(-1)
    if values.shape[0] != n_rows:
        raise ValueError("index dataset length does not match the feature rows")
    return values


def _is_row_identity(catalog: np.ndarray) -> bool:
    return bool(np.array_equal(catalog, np.arange(catalog.shape[0], dtype=np.int64)))


def _catalog_row_map(catalog: np.ndarray) -> dict[int, int] | None:
    if _is_row_identity(catalog):
        return None
    mapping: dict[int, int] = {}
    for row, value in enumerate(catalog.tolist()):
        key = int(value)
        if key in mapping:
            raise ValueError(f"duplicate catalog index {key}")
        mapping[key] = row
    return mapping


def filter_catalog_indices(path: str, indices: Sequence[int]) -> list[int]:
    """Keep ids that name a row, using the catalog index when the file is a subset."""
    catalog = catalog_indices(path)
    if _is_row_identity(catalog):
        n_rows = int(catalog.shape[0])
        return [int(value) for value in indices if 0 <= int(value) < n_rows]
    present = set(_catalog_row_map(catalog) or {})
    return [int(value) for value in indices if int(value) in present]


def load_feature_rows(path: str, indices: Sequence[int]) -> np.ndarray:
    """Load rows in the requested order.

    ``indices`` are catalog ids. On a subset file they are resolved through the
    stored ``index`` dataset instead of the physical row number. Ids may repeat
    and need not be sorted.
    """
    wanted = np.asarray(list(indices), dtype=np.int64).reshape(-1)
    if wanted.size == 0:
        with h5py.File(path, "r") as handle:
            dim = int(handle[_feature_key(handle)].shape[1])
        return np.zeros((0, dim), dtype=np.float32)
    unique, inverse = np.unique(wanted, return_inverse=True)
    with h5py.File(path, "r") as handle:
        dataset = handle[_feature_key(handle)]
        catalog = (
            np.asarray(handle["index"], dtype=np.int64).reshape(-1)
            if "index" in handle
            else np.arange(dataset.shape[0], dtype=np.int64)
        )
        if catalog.shape[0] != dataset.shape[0]:
            raise ValueError("index dataset length does not match the feature rows")
        row_of = _catalog_row_map(catalog)
        if row_of is None:
            loaded = np.asarray(dataset[unique.tolist()], dtype=np.float32)
        else:
            try:
                rows = [row_of[int(value)] for value in unique.tolist()]
            except KeyError as exc:
                raise KeyError(f"catalog index {exc.args[0]} is not in the feature file") from exc
            loaded = np.asarray(dataset[rows], dtype=np.float32)
    return loaded[inverse]
