import csv
import zipfile
from pathlib import Path

import h5py
import numpy as np

from hitl_features.features import catalog_indices, load_feature_rows
from hitl_features.subset import choose_subset_indices, write_feature_table, write_stored_zip

ROOT = Path(__file__).resolve().parents[1]


def test_subset_keeps_required_ids_and_a_fixed_fill():
    catalog = list(range(20))
    required = [5, 1, 9, 5, 99]
    chosen = choose_subset_indices(catalog, required, size=8, seed=0)
    assert chosen == sorted(chosen)
    assert len(chosen) == 8
    assert {1, 5, 9} <= set(chosen)
    assert 99 not in chosen
    assert choose_subset_indices(catalog, required, size=8, seed=0) == chosen


def test_feature_table_and_zip_are_uncompressed(tmp_path: Path):
    matrix = np.arange(8, dtype=np.float32).reshape(4, 2)
    feature_path = tmp_path / "features.h5"
    write_feature_table(feature_path, matrix, [10, 20, 30, 40])
    with h5py.File(feature_path, "r") as handle:
        assert handle["mae_features"].compression is None
        assert handle["index"].compression is None
    loaded = load_feature_rows(feature_path, [30, 10])
    assert np.allclose(loaded[0], matrix[2])
    assert np.allclose(loaded[1], matrix[0])
    source = tmp_path / "10.jpg"
    source.write_bytes(b"jpg-bytes")
    archive = tmp_path / "images.zip"
    write_stored_zip(archive, [("10.jpg", source)])
    with zipfile.ZipFile(archive) as handle:
        info = handle.getinfo("10.jpg")
        assert info.compress_type == zipfile.ZIP_STORED
        assert handle.read("10.jpg") == b"jpg-bytes"


def test_shipped_subset_keeps_original_catalog_indexes():
    feature_path = ROOT / "datasets" / "fama_base" / "features.h5"
    ae_path = ROOT / "datasets" / "ae" / "features.h5"
    labels = ROOT / "subset" / "expert_testset.csv"
    archive = ROOT / "subset" / "images.zip"
    assert feature_path.is_file() and ae_path.is_file() and labels.is_file() and archive.is_file()
    assert not (ROOT / "subset" / "features.h5").exists()
    ids = catalog_indices(str(feature_path))
    ae_ids = catalog_indices(str(ae_path))
    assert list(map(int, ae_ids)) == list(map(int, ids))
    assert len(ids) == 10000
    assert len(set(map(int, ids))) == 10000
    with open(labels, newline="") as handle:
        required = {int(row["index"]) for row in csv.DictReader(handle)}
    assert required <= set(map(int, ids))
    for path in (feature_path, ae_path):
        with h5py.File(path, "r") as handle:
            assert handle["mae_features"].compression is None
            assert handle["index"].compression is None
            assert handle["mae_features"].shape == (10000, 768)
    sample = int(ids[100])
    assert load_feature_rows(str(feature_path), [sample]).shape == (1, 768)
    with zipfile.ZipFile(archive) as handle:
        info = handle.getinfo(f"{sample}.jpg")
        assert info.compress_type == zipfile.ZIP_STORED
        assert "0.jpg" not in handle.namelist()
