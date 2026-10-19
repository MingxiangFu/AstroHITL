import numpy as np

from hitl_features.recommend import recommend


def test_random_excludes_graph_and_matches_seed():
    ids = np.array([10, 11, 12, 13, 14])
    vectors = np.eye(5, dtype=np.float64)
    graph_ids = np.array([10])
    first = recommend(
        ids, vectors, graph_ids, vectors[:1], None, "random",
        n=2, rng=np.random.default_rng(0), n_categories=2, round_index=3,
    )
    second = recommend(
        ids, vectors, graph_ids, vectors[:1], None, "random",
        n=2, rng=np.random.default_rng(0), n_categories=2, round_index=3,
    )
    assert first == second
    assert 10 not in first
    assert len(first) == 2


def test_diversity_starts_at_lowest_index_then_spreads():
    ids = np.array([0, 1, 2])
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]], dtype=np.float64)
    chosen = recommend(
        ids, vectors, np.array([], dtype=int), np.zeros((0, 2)), None, "diversity",
        n=2, rng=np.random.default_rng(0), n_categories=1, round_index=2,
    )
    assert chosen == [0, 2]


def test_unknown_picks_the_point_farthest_from_the_graph():
    ids = np.array([0, 1, 2])
    vectors = np.array([[1.0, 0.0], [-1.0, 0.0], [0.8, 0.2]], dtype=np.float64)
    graph = np.array([[1.0, 0.0]], dtype=np.float64)
    chosen = recommend(
        ids, vectors, np.array([9]), graph, ["a"], "unknown",
        n=1, rng=np.random.default_rng(0), n_categories=1, round_index=2,
    )
    assert chosen == [1]


def test_unknown_with_empty_graph_falls_back_to_diversity():
    ids = np.array([0, 1, 2])
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]], dtype=np.float64)
    chosen = recommend(
        ids, vectors, np.array([], dtype=int), np.zeros((0, 2)), None, "unknown",
        n=2, rng=np.random.default_rng(0), n_categories=0, round_index=2,
    )
    assert chosen == [0, 2]


def test_certain_ranks_by_max_probability_not_the_hard_label():
    ids = np.array([1, 2])
    vectors = np.array([[0.2, 1.0], [1.0, 0.0]], dtype=np.float64)
    graph_ids = np.array([10, 11, 12, 13])
    graph_vectors = np.array(
        [[1.0, 0.0], [0.95, 0.05], [0.9, 0.1], [0.0, 1.0]], dtype=np.float64
    )
    chosen = recommend(
        ids, vectors, graph_ids, graph_vectors, ["A", "A", "A", "B"], "certain",
        n=1, rng=np.random.default_rng(0), n_categories=2, round_index=2, knn_k=3,
    )
    assert chosen == [2]


def test_uncertain_ranks_entropy_then_lower_max_probability():
    from hitl_features.recommend import certainty_order

    peaked = [1.0, 0.0, 0.0]
    mixed = [0.5, 0.5, 0.0]
    flatter = [0.4, 0.4, 0.2]
    order = certainty_order(
        [10, 11, 12],
        np.array([peaked, mixed, flatter], dtype=np.float64),
        "uncertain",
    )
    assert order[0] == 12
    tied = certainty_order(
        [4, 9],
        np.array([mixed, mixed], dtype=np.float64),
        "uncertain",
    )
    assert tied == [4, 9]


def test_uncertain_tie_breaks_toward_lower_max_probability(monkeypatch):
    from hitl_features.recommend import certainty_order

    def equal_entropy(probabilities):
        return np.zeros(len(probabilities), dtype=np.float64)

    monkeypatch.setattr("hitl_features.recommend.normalized_entropy", equal_entropy)
    order = certainty_order(
        [2, 7],
        np.array([[0.9, 0.1], [0.6, 0.4]], dtype=np.float64),
        "uncertain",
    )
    assert order == [7, 2]


def test_uncertain_recommend_prefers_the_mixed_neighbor_vote():
    ids = np.array([1, 2])
    vectors = np.array([[0.2, 1.0], [1.0, 0.0]], dtype=np.float64)
    graph_ids = np.array([10, 11, 12, 13])
    graph_vectors = np.array(
        [[1.0, 0.0], [0.95, 0.05], [0.9, 0.1], [0.0, 1.0]], dtype=np.float64
    )
    chosen = recommend(
        ids, vectors, graph_ids, graph_vectors, ["A", "A", "A", "B"], "uncertain",
        n=1, rng=np.random.default_rng(0), n_categories=2, round_index=2, knn_k=3,
    )
    assert chosen == [1]


def test_algorithmic_round_one_matches_random_then_unknown():
    ids = np.array([0, 1, 2, 3])
    vectors = np.array([[1.0, 0.0], [-1.0, 0.0], [0.2, 0.9], [-0.2, -1.0]], dtype=np.float64)
    graph_ids = np.array([8])
    graph_vectors = np.array([[1.0, 0.0]], dtype=np.float64)
    algorithmic = recommend(
        ids, vectors, graph_ids, graph_vectors, ["a"], "algorithmic",
        n=1, rng=np.random.default_rng(1), n_categories=2, round_index=1,
    )
    random_choice = recommend(
        ids, vectors, graph_ids, graph_vectors, ["a"], "random",
        n=1, rng=np.random.default_rng(1), n_categories=2, round_index=1,
    )
    unknown = recommend(
        ids, vectors, graph_ids, graph_vectors, ["a"], "unknown",
        n=1, rng=np.random.default_rng(1), n_categories=2, round_index=2,
    )
    later = recommend(
        ids, vectors, graph_ids, graph_vectors, ["a"], "algorithmic",
        n=1, rng=np.random.default_rng(1), n_categories=2, round_index=2,
    )
    assert algorithmic == random_choice
    assert later == unknown
    assert later == [1]


def test_single_class_certain_and_uncertain_follow_mean_distance():
    """One class gives every sample probability 1 and entropy 0.

    That tie is the kNN math. Certain then follows the smallest mean cosine
    distance to the labeled members, and uncertain the largest.
    """
    ids = np.array([30, 10, 20])
    vectors = np.array(
        [[0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]],
        dtype=np.float64,
    )
    refs = np.array([[1.0, 0.0], [0.8, 0.2]], dtype=np.float64)
    labels = ["A", "A"]
    graph_ids = np.array([1, 2])
    certain = recommend(
        ids, vectors, graph_ids, refs, labels, "certain",
        n=3, rng=np.random.default_rng(0), n_categories=1, round_index=2,
    )
    uncertain = recommend(
        ids, vectors, graph_ids, refs, labels, "uncertain",
        n=3, rng=np.random.default_rng(0), n_categories=1, round_index=2,
    )
    assert certain != uncertain
    assert certain == [20, 30, 10]
    assert uncertain == [10, 30, 20]
