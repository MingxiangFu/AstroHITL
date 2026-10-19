import numpy as np

from hitl_features.score_board import (
    FIXED_SAMPLE_SIZE,
    N_BINS,
    PER_BIN,
    equal_width_bins,
    fixed_score_indices,
)


def test_fixed_score_sample_is_seeded_and_stable():
    assert FIXED_SAMPLE_SIZE == 40
    pool = list(range(100))
    first = fixed_score_indices(pool, seed=7)
    second = fixed_score_indices(pool, seed=7)
    assert first == second
    assert first == sorted(first)
    assert len(first) == 40
    assert fixed_score_indices(pool, seed=8) != first
    assert len(fixed_score_indices([3, 1, 2], seed=0)) == 3


def test_equal_width_bins_cover_the_range_and_cap_thumbnails():
    assert N_BINS == 5
    assert PER_BIN == 4
    indices = list(range(20))
    scores = np.linspace(0.0, 1.0, 20)
    bins = equal_width_bins(indices, scores, labels=["c"] * 20)
    assert len(bins) == 5
    assert sum(item["count"] for item in bins) == 20
    assert all(len(item["images"]) <= 4 for item in bins)
    assert bins[0]["low"] == 0.0
    assert abs((bins[0]["high"] - bins[0]["low"]) - 0.2) < 1e-9
    assert bins[-1]["high"] == 1.0
    assert all(image["label"] == "c" for item in bins for image in item["images"])

    flat = equal_width_bins([1, 2, 3, 4, 5], [0.5, 0.5, 0.5, 0.5, 0.5])
    assert flat[0]["count"] == 5
    assert len(flat[0]["images"]) == 4
    assert sum(item["count"] for item in flat[1:]) == 0
    assert "label" not in flat[0]["images"][0]
