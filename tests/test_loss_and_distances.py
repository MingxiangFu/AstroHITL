import numpy as np
import torch
import torch.nn.functional as F

from hitl_features.distances import mean_within_between
from hitl_features.graph import CategoryGraph
from hitl_features.loss import build_pair_targets, weighted_supcon_loss


def _unit(*rows):
    return F.normalize(torch.tensor(rows, dtype=torch.float32), dim=-1)


def test_zero_strength_matches_batch_without_that_negative():
    vectors = _unit([1.0, 0.0], [0.9, 0.1], [0.0, 1.0])
    positive = torch.zeros(3, 3)
    positive[0, 1] = positive[1, 0] = 1
    negative = torch.zeros(3, 3)
    masked = weighted_supcon_loss(vectors, positive, negative)
    pair = weighted_supcon_loss(vectors[:2], positive[:2, :2], negative[:2, :2])
    assert torch.allclose(masked, pair, atol=1e-5)


def test_higher_strength_increases_loss_when_negative_is_identical():
    vectors = _unit([1.0, 0.0], [1.0, 0.0], [1.0, 0.0])
    positive = torch.zeros(3, 3)
    positive[0, 1] = positive[1, 0] = 1
    weak = torch.zeros(3, 3)
    strong = torch.zeros(3, 3)
    weak[0, 2] = weak[2, 0] = 0.5
    strong[0, 2] = strong[2, 0] = 2.0
    weak_loss = weighted_supcon_loss(vectors, positive, weak)
    strong_loss = weighted_supcon_loss(vectors, positive, strong)
    assert float(strong_loss) > float(weak_loss)


def test_ancestor_pair_is_neither_positive_nor_negative():
    graph = CategoryGraph()
    parent = graph.create_category("parent")
    child = graph.create_category("child", parent_id=parent)
    other = graph.create_category("other")
    graph.set_strength(parent, other, 1.5)
    graph.assign(0, parent)
    graph.assign(1, child)
    graph.assign(2, other)
    positive, negative = build_pair_targets(graph, [0, 1, 2])
    assert positive[0, 1] == 0 and negative[0, 1] == 0
    assert negative[1, 2] == 1.5
    assert positive[0, 0] == 0


def test_mean_within_skips_singletons_but_between_keeps_them():
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    labels = ["a", "a", "b", "c"]
    within, between = mean_within_between(vectors, labels)
    assert abs(within - 0.0) < 1e-6
    assert abs(between - 1.4) < 1e-6


def test_between_distance_of_opposite_unit_vectors():
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
    labels = ["a", "a", "b", "b"]
    within, between = mean_within_between(vectors, labels)
    assert abs(within - 0.0) < 1e-6
    assert abs(between - 2.0) < 1e-6
