"""Thumbnails from a freshly unpacked portable subset."""

import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from hitl_features.web.app import create_app

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "subset" / "images.zip"


def test_relative_subset_images_are_served_by_catalog_index(tmp_path, monkeypatch):
    if not ARCHIVE.is_file():
        pytest.skip("portable image archive is not in this checkout")
    image_root = tmp_path / "subset" / "images"
    image_root.mkdir(parents=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        archive.extract("13621.jpg", image_root)
        archive.extract("63830.jpg", image_root)
    feature_path = tmp_path / "subset" / "features.h5"
    with h5py.File(feature_path, "w") as handle:
        handle.create_dataset("features", data=np.zeros((2, 4), dtype=np.float32))
        handle.create_dataset("index", data=np.array([13621, 63830], dtype=np.int64))
    monkeypatch.chdir(tmp_path)
    app = create_app(tmp_path / "sessions")
    client = app.test_client()
    created = client.post(
        "/api/sessions",
        json={
            "feature_path": "subset/features.h5",
            "img_dir": "subset/images",
            "pool_indices": [13621, 63830],
            "pool_size": 2,
            "seed": 0,
            "out_dim": 4,
            "round_size": 2,
            "epochs": 1,
            "device": "cpu",
        },
    )
    assert created.status_code == 200, created.get_json()
    session_id = created.get_json()["snapshot"]["id"]
    for index in (13621, 63830):
        response = client.get(f"/media/{session_id}/{index}")
        assert response.status_code == 200
        assert response.mimetype == "image/jpeg"
        assert response.data[:2] == b"\xff\xd8"
    missing = client.get(f"/media/{session_id}/999999")
    assert missing.status_code == 404
