"""Contact sheets draw catalog jpegs, not only index placeholders."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hitl_features.figures import save_comparison_figures

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "subset" / "images.zip"
FULL_TREE = Path("<path-to-images>")
INDEX = 13621


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"))


def _quantized(rgb: np.ndarray, step: int = 16) -> set[tuple[int, int, int]]:
    values = (rgb.reshape(-1, 3).astype(np.int16) // step) * step
    return {tuple(int(channel) for channel in row) for row in values}


def _overlap(sheet: np.ndarray, jpeg: np.ndarray) -> int:
    return len(_quantized(sheet) & _quantized(jpeg))


def test_contact_sheet_draws_unpacked_subset_jpeg(tmp_path, monkeypatch):
    if not ARCHIVE.is_file():
        pytest.skip("portable image archive is not in this checkout")
    image_root = tmp_path / "subset" / "images"
    image_root.mkdir(parents=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        archive.extract(f"{INDEX}.jpg", image_root)
    monkeypatch.chdir(tmp_path)
    examples_high = [(INDEX, 0.91)]
    examples_low = [(INDEX, 0.04)]
    placeholder = save_comparison_figures(
        tmp_path / "placeholder",
        [],
        high=examples_high,
        low=examples_low,
    )
    drawn = save_comparison_figures(
        tmp_path / "drawn",
        [],
        high=examples_high,
        low=examples_low,
        img_dir="subset/images",
    )
    jpeg = _rgb(image_root / f"{INDEX}.jpg")
    blank = _rgb(tmp_path / "placeholder" / placeholder["contact_sheet"])
    sheet = _rgb(tmp_path / "drawn" / drawn["contact_sheet"])
    blank_overlap = _overlap(blank, jpeg)
    sheet_overlap = _overlap(sheet, jpeg)
    assert sheet_overlap > blank_overlap
    assert sheet_overlap >= 5


def test_contact_sheet_draws_full_thumbnail_tree(tmp_path):
    path = FULL_TREE / f"{INDEX}.jpg"
    if not path.is_file():
        pytest.skip("full thumbnail tree is not available")
    drawn = save_comparison_figures(
        tmp_path,
        [],
        high=[(INDEX, 0.91)],
        low=[(INDEX, 0.04)],
        img_dir=FULL_TREE,
    )
    sheet = _rgb(tmp_path / drawn["contact_sheet"])
    jpeg = _rgb(path)
    assert _overlap(sheet, jpeg) >= 5
