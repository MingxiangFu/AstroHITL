import numpy as np

from hitl_features.preview import RECENT_N, REPRESENTATIVE_M, category_preview


def test_preview_defaults_are_two_recent_and_three_representative():
    assert RECENT_N == 2
    assert REPRESENTATIVE_M == 3
    ids = list(range(10))
    vectors = np.eye(10)
    orders = list(range(10))
    preview = category_preview(ids, vectors, orders)
    assert preview["recent"] == [9, 8]
    assert len(preview["representative"]) == 3
    assert set(preview["recent"]).isdisjoint(preview["representative"])


def test_small_class_omits_the_empty_side():
    vectors = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
    preview = category_preview([4, 7], vectors, [1, 2])
    assert preview["recent"] == [7, 4]
    assert preview["representative"] == []


def test_representatives_skip_recent_images_and_prefer_a_spread_of_the_dense_pool():
    ids = [0, 1, 2]
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
    preview = category_preview(ids, vectors, [0, 0, 0], n=0, m=2)
    assert preview["recent"] == []
    assert preview["representative"] == [0, 2]

    dense = np.array([[1.0, 0.01 * (i - 7)] for i in range(15)], dtype=np.float64)
    outliers = np.array([[-1.0, 0.0], [-1.0, 0.2], [-1.0, -0.2]], dtype=np.float64)
    vectors = np.vstack([dense, outliers])
    orders = list(range(1, 16)) + [100, 101, 102]
    preview = category_preview(list(range(18)), vectors, orders)
    assert preview["recent"] == [17, 16]
    assert len(preview["representative"]) == 3
    assert set(preview["representative"]).issubset(set(range(15)))
    assert set(preview["recent"]).isdisjoint(preview["representative"])
