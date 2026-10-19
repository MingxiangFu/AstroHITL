"""ROC, precision-recall, and a high/low score contact sheet."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve

from hitl_features.images import resolve_image_path

ARM_COLORS = {
    "A_recommended_raw": "#1f4e79",
    "B_recommended_mapped": "#c45911",
    "C_random_raw": "#548235",
    "D_random_mapped": "#7030a0",
}


def high_low_examples(sample_ids, scores, k: int = 4) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    ids = [int(value) for value in sample_ids]
    if len(ids) != len(values) or len(ids) == 0:
        return [], []
    order = np.argsort(values, kind="mergesort")
    take = min(int(k), len(order))
    low = [(ids[int(i)], float(values[int(i)])) for i in order[:take]]
    high = [(ids[int(i)], float(values[int(i)])) for i in order[-take:][::-1]]
    return high, low


def save_comparison_figures(
    directory: Path | str,
    arms: list[dict],
    high: list[tuple[int, float]],
    low: list[tuple[int, float]],
    img_dir: Path | str | None = None,
) -> dict:
    """Write ROC and PR curves on shared axes, plus a contact sheet.

    PNG files use uncompressed pixels. ``arms`` items have ``name``, ``y``,
    and ``scores``. ``img_dir`` is a thumbnail directory of ``{catalog_index}.jpg``
    files. A relative directory is resolved from the process working directory.
    A missing thumbnail stays an index and score placeholder.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    _save_curve(root / "roc.png", arms, kind="roc")
    _save_curve(root / "pr.png", arms, kind="pr")
    _save_contact_sheet(root / "contact_sheet.png", high, low, img_dir=img_dir)
    return {"roc": "roc.png", "pr": "pr.png", "contact_sheet": "contact_sheet.png"}


def _save_curve(path: Path, arms: list[dict], kind: str) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    for arm in arms:
        labels = np.asarray(arm["y"]).astype(int).reshape(-1)
        scores = np.asarray(arm["scores"], dtype=np.float64).reshape(-1)
        if len(labels) != len(scores) or len(np.unique(labels)) < 2:
            continue
        name = str(arm["name"])
        color = ARM_COLORS.get(name)
        if kind == "roc":
            x_values, y_values, _ = roc_curve(labels, scores)
            axis.plot(x_values, y_values, label=name, color=color)
        else:
            precision, recall, _ = precision_recall_curve(labels, scores)
            axis.plot(recall, precision, label=name, color=color)
    if kind == "roc":
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.set_title("ROC")
    else:
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title("Precision-recall")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, pil_kwargs={"compress_level": 0})
    plt.close(figure)


def _save_contact_sheet(
    path: Path,
    high: list[tuple[int, float]],
    low: list[tuple[int, float]],
    img_dir: Path | str | None = None,
) -> None:
    columns = max(len(high), len(low), 1)
    figure, axes = plt.subplots(2, columns, figsize=(2.1 * columns, 4.2), squeeze=False)
    image_root = _thumbnail_root(img_dir)
    _draw_row(axes[0], high, "High score", columns, image_root)
    _draw_row(axes[1], low, "Low score", columns, image_root)
    figure.tight_layout()
    figure.savefig(path, pil_kwargs={"compress_level": 0})
    plt.close(figure)


def _thumbnail_root(img_dir: Path | str | None) -> Path | None:
    if not img_dir:
        return None
    root = Path(img_dir).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    return root


def _thumbnail_array(img_dir: Path | None, index: int) -> np.ndarray | None:
    if img_dir is None:
        return None
    path = resolve_image_path(img_dir, int(index))
    if path is None:
        return None
    with Image.open(Path(path).resolve()) as handle:
        return np.asarray(handle.convert("RGB"))


def _draw_row(
    axes,
    examples: list[tuple[int, float]],
    title: str,
    columns: int,
    img_dir: Path | None,
) -> None:
    for column in range(columns):
        axis = axes[column]
        axis.set_xticks([])
        axis.set_yticks([])
        if column == 0:
            axis.set_ylabel(title)
        if column >= len(examples):
            axis.axis("off")
            continue
        index, score = examples[column]
        image = _thumbnail_array(img_dir, index)
        if image is None:
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.add_patch(plt.Rectangle((0, 0), 1, 1, color="#d6e3f0"))
            axis.text(0.5, 0.62, f"#{int(index)}", ha="center", va="center", fontsize=11)
            axis.text(0.5, 0.32, f"{float(score):.3f}", ha="center", va="center", fontsize=9)
            continue
        axis.imshow(image)
        axis.set_title(f"#{int(index)}  {float(score):.3f}", fontsize=8)
