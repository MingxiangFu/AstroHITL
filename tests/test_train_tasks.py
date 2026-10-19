import numpy as np
import pytest
import torch

from hitl_features.classify import fit_classifier, predict_proba
from hitl_features.four_way import run_four_way
from hitl_features.graph import CategoryGraph
from hitl_features.mapper import FeatureMapper
from hitl_features.rank import fit_ranker, pairwise_preference_accuracy
from hitl_features.tasks import task_labels
from hitl_features.train import train_mapper


def test_epochs_zero_does_not_change_weights():
    mapper = FeatureMapper(4, 2, hidden_dim=3)
    before = [p.detach().clone() for p in mapper.parameters()]
    graph = CategoryGraph()
    cat = graph.create_category("a")
    graph.assign(0, cat)
    graph.assign(1, cat)
    result = train_mapper(
        mapper, np.ones((2, 4), dtype=np.float32), [0, 1], graph, epochs=0, device="cpu"
    )
    assert result["steps"] == 0
    for old, new in zip(before, mapper.parameters()):
        assert torch.equal(old, new)


def test_training_decreases_loss_and_separates_categories():
    rng = np.random.default_rng(0)
    features = rng.normal(scale=0.1, size=(20, 16)).astype(np.float32)
    features[:10, 0] += 1.0
    features[10:, 1] += 1.0
    graph = CategoryGraph()
    left = graph.create_category("left")
    right = graph.create_category("right")
    for index in range(10):
        graph.assign(index, left)
    for index in range(10, 20):
        graph.assign(index, right)
    mapper = FeatureMapper(16, 8, hidden_dim=16)
    result = train_mapper(
        mapper, features, list(range(20)), graph, epochs=40, device="cpu"
    )
    assert result["steps"] == 40
    assert result["history"][-1] < result["history"][0]
    with torch.no_grad():
        mapped = mapper(torch.from_numpy(features)).numpy()
    from hitl_features.distances import mean_within_between

    labels = ["left"] * 10 + ["right"] * 10
    within, between = mean_within_between(mapped, labels)
    assert within < between


def test_task_labels_prefer_the_deeper_assigned_category():
    graph = CategoryGraph()
    parent = graph.create_category("spiral")
    child = graph.create_category("barred", parent_id=parent)
    other = graph.create_category("smooth")
    graph.assign(1, parent)
    graph.assign(2, child)
    graph.assign(3, other)
    labels = task_labels(
        graph,
        [("spirals", [parent]), ("barred", [child])],
    )
    assert labels == {1: "spirals", 2: "barred"}


def test_task_labels_reject_a_category_in_two_classes():
    graph = CategoryGraph()
    cat = graph.create_category("a")
    graph.assign(1, cat)
    with pytest.raises(ValueError):
        task_labels(graph, [("one", [cat]), ("two", [cat])])


def test_four_way_mapped_recommended_beats_raw_noise():
    rng = np.random.default_rng(0)
    y_rec = np.array([0, 1] * 20)
    y_rand = np.array([0, 1] * 20)
    y_eval = np.array([0, 1] * 20)

    def noise(rows):
        return rng.normal(size=(rows, 4)).astype(np.float32)

    rec_mapped = np.zeros((40, 4), dtype=np.float32)
    rec_mapped[:, 0] = y_rec
    eval_mapped = np.zeros((40, 4), dtype=np.float32)
    eval_mapped[:, 0] = y_eval
    report = run_four_way(
        noise(40), rec_mapped, y_rec,
        noise(40), noise(40), y_rand,
        noise(40), eval_mapped, y_eval,
        seed=0, epochs=40, device="cpu",
    )
    assert report["B_recommended_mapped"]["roc_auc"] > 0.9
    assert report["B_recommended_mapped"]["roc_auc"] > report["A_recommended_raw"]["roc_auc"] + 0.2


def test_ranker_orders_two_groups():
    rng = np.random.default_rng(0)
    features = rng.normal(scale=0.05, size=(30, 4)).astype(np.float32)
    groups = ["high"] * 15 + ["low"] * 15
    features[:15, 0] += 1.0
    features[15:, 0] -= 1.0
    model = fit_ranker(
        features, groups, ["high", "low"], strength=1.0, seed=0, epochs=40, device="cpu"
    )
    scores = model.score(features)
    accuracy = pairwise_preference_accuracy(scores, groups, ["high", "low"])
    assert accuracy > 0.9


def test_classifier_proba_shape():
    features = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    labels = np.array([0, 1, 0, 1])
    model = fit_classifier(features, labels, seed=0, epochs=5, device="cpu")
    proba = predict_proba(model, features, device="cpu")
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
