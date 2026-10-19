"""Resolve a thumbnail from a sample index."""

from __future__ import annotations

from pathlib import Path

_EXTENSIONS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
_PAD_WIDTHS = (4, 5, 6, 8)


def resolve_image_path(img_dir: Path | str, index: int) -> Path | None:
    root = Path(img_dir)
    if not root.is_dir():
        return None
    stems = [str(int(index))]
    for width in _PAD_WIDTHS:
        stems.append(f"{int(index):0{width}d}")
    for stem in stems:
        for ext in _EXTENSIONS:
            candidate = root / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
    return None
