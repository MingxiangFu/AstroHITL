"""Equal-width score bins for a fixed pool sample."""

from __future__ import annotations

import numpy as np

N_BINS = 5
PER_BIN = 4
FIXED_SAMPLE_SIZE = 40


def fixed_score_indices(pool_indices: list[int], seed: int, n: int = FIXED_SAMPLE_SIZE) -> list[int]:
    """Seeded sample of the bound pool. The same seed always returns the same ids."""
    pool = np.array(sorted(int(value) for value in pool_indices), dtype=np.int64)
    if len(pool) == 0 or int(n) <= 0:
        return []
    take = min(int(n), len(pool))
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(pool, size=take, replace=False)
    return sorted(int(value) for value in np.asarray(chosen).reshape(-1))


def equal_width_bins(
    indices: list[int],
    scores,
    labels: list[str] | None = None,
    *,
    n_bins: int = N_BINS,
    per_bin: int = PER_BIN,
) -> list[dict]:
    """Split scores into equal-width bins and keep up to ``per_bin`` thumbnails."""
    sample_ids = [int(value) for value in indices]
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(sample_ids) != len(values):
        raise ValueError("indices and scores must align")
    if labels is not None and len(labels) != len(sample_ids):
        raise ValueError("labels must align with indices")
    bins = int(n_bins)
    if bins < 1:
        raise ValueError("n_bins must be positive")
    if len(sample_ids) == 0:
        return [
            {"low": 0.0, "high": 1.0, "count": 0, "images": []}
            for _ in range(bins)
        ]
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    width = (hi - lo) / bins
    grouped: list[list[dict]] = [[] for _ in range(bins)]
    for position, (index, score) in enumerate(zip(sample_ids, values)):
        if float(score) <= lo:
            bucket = 0
        else:
            bucket = int((float(score) - lo) / width)
            if bucket >= bins:
                bucket = bins - 1
        image = {"index": index, "score": float(score)}
        if labels is not None:
            image["label"] = str(labels[position])
        grouped[bucket].append(image)
    result = []
    for i, images in enumerate(grouped):
        images.sort(key=lambda item: (-item["score"], item["index"]))
        result.append(
            {
                "low": lo + i * width,
                "high": lo + (i + 1) * width,
                "count": len(images),
                "images": images[: int(per_bin)],
            }
        )
    return result
