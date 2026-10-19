from pathlib import Path

import h5py
import numpy as np
import pytest

from hitl_features.features import (
    catalog_indices,
    feature_matrix_shape,
    filter_catalog_indices,
    load_feature_rows,
)
from hitl_features.images import resolve_image_path


def test_load_feature_rows_preserves_request_order(tmp_path: Path):
    path = tmp_path / "feat.h5"
    matrix = np.arange(20, dtype=np.float32).reshape(5, 4)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("mae_features", data=matrix)
    rows = load_feature_rows(path, [4, 1, 1])
    assert rows.shape == (3, 4)
    assert np.allclose(rows[0], matrix[4])
    assert np.allclose(rows[1], matrix[1])
    assert feature_matrix_shape(path) == (5, 4)


def test_fama_base_matrix_shape():
    path = Path("<path-to-features>")
    if not path.is_file():
        pytest.skip("full FAMA feature catalog is not available")
    assert feature_matrix_shape(str(path)) == (451400, 768)


def test_subset_rows_are_addressed_by_catalog_index(tmp_path: Path):
    path = tmp_path / "subset.h5"
    matrix = np.array([[1, 0], [0, 1], [2, 3], [4, 5]], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("mae_features", data=matrix)
        handle.create_dataset("index", data=np.array([8, 3, 11, 2], dtype=np.int64))
    rows = load_feature_rows(path, [11, 8])
    assert np.allclose(rows[0], matrix[2])
    assert np.allclose(rows[1], matrix[0])
    assert list(catalog_indices(path)) == [8, 3, 11, 2]
    assert filter_catalog_indices(path, [11, 99, 2]) == [11, 2]


def test_resolve_image_path_uses_index_filename(tmp_path: Path):
    (tmp_path / "12.jpg").write_bytes(b"jpg")
    assert resolve_image_path(tmp_path, 12) == tmp_path / "12.jpg"
    assert resolve_image_path(tmp_path, 99) is None
