"""Train/val/test fractions and which images a classification result may show."""

import h5py
import numpy as np
import pytest

from hitl_features.session import SessionStore
from hitl_features.tasks import (
    fraction_counts,
    inference_ids,
    normalize_fractions,
    stratified_split,
)
from hitl_features.web.app import create_app


def test_split_fractions_stay_complete_and_disjoint():
    with pytest.raises(ValueError):
        normalize_fractions(0, 0.5, 0.5)
    with pytest.raises(ValueError):
        normalize_fractions(0.6, -0.1, 0.5)
    with pytest.raises(ValueError):
        normalize_fractions(0.5, 0.5, 0.5)

    assert fraction_counts(6, (0.6, 0.2, 0.2)) == [4, 1, 1]
    assert fraction_counts(8, (0.5, 0.5, 0.0)) == [4, 4, 0]
    assert fraction_counts(5, (0.5, 0.5, 0.0)) == [3, 2, 0]
    assert fraction_counts(1, (0.6, 0.2, 0.2)) == [1, 0, 0]

    groups = {"left": list(range(8)), "right": list(range(100, 108))}
    fractions = (0.5, 0.5, 0.0)
    first = stratified_split(groups, fractions, seed=3)
    again = stratified_split(groups, fractions, seed=3)
    assert first == again
    for name, members in groups.items():
        parts = first[name]
        assert parts["test"] == []
        flat = parts["train"] + parts["val"] + parts["test"]
        assert sorted(flat) == sorted(members)
        assert len(flat) == len(set(flat))
        assert [len(parts[label]) for label in ("train", "val", "test")] == fraction_counts(
            len(members), fractions
        )


def test_inference_ids_drop_train_and_outsiders():
    members = [1, 2, 3, 4]
    assert inference_ids(members, [2, 4]) == [1, 3]
    assert 9 not in inference_ids(members, [])


def _session(tmp_path, matrix, pool):
    feature_path = tmp_path / "feat.h5"
    with h5py.File(feature_path, "w") as handle:
        handle.create_dataset("features", data=np.asarray(matrix, dtype=np.float32))
    app = create_app(tmp_path / "sessions")
    client = app.test_client()
    created = client.post(
        "/api/sessions",
        json={
            "feature_path": str(feature_path),
            "pool_indices": list(pool),
            "pool_size": len(pool),
            "seed": 0,
            "out_dim": 4,
            "round_size": 2,
            "epochs": 1,
            "device": "cpu",
        },
    )
    assert created.status_code == 200, created.get_json()
    return client, created.get_json()["snapshot"]["id"]


def test_task_result_samples_skip_training_images_and_other_categories(tmp_path):
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(20, 4)).astype(np.float32)
    matrix[:8, 0] += 3
    matrix[8:16, 0] -= 3
    client, session_id = _session(tmp_path, matrix, list(range(20)))
    session = client.application.config["STORE"].get(session_id)
    left = session.graph.create_category("left")
    right = session.graph.create_category("right")
    other = session.graph.create_category("other")
    for index in range(8):
        session.graph.assign(index, left, round_index=1)
    for index in range(8, 16):
        session.graph.assign(index, right, round_index=1)
    for index in range(16, 20):
        session.graph.assign(index, other, round_index=1)
    classes = [
        {"name": "left", "category_ids": [left]},
        {"name": "right", "category_ids": [right]},
    ]
    refused = client.post(
        f"/api/sessions/{session_id}/tasks/classification",
        json={"classes": classes, "epochs": 1, "fractions": {"train": 0, "val": 0.5, "test": 0.5}},
    )
    assert refused.status_code == 400
    trained = client.post(
        f"/api/sessions/{session_id}/tasks/classification",
        json={
            "classes": classes,
            "epochs": 1,
            "fractions": {"train": 0.5, "val": 0.5, "test": 0},
        },
    )
    assert trained.status_code == 200, trained.get_json()
    summary = trained.get_json()["snapshot"]["task_summary"]
    assert summary["kind"] == "classification"
    fractions = summary["fractions"]
    assert abs(fractions["train"] + fractions["val"] + fractions["test"] - 1) < 1e-6
    assert summary["splits"]["train"]["n"] == 8
    assert summary["splits"]["val"]["n"] == 8
    assert summary["splits"]["test"]["n"] == 0
    groups = {"left": list(range(8)), "right": list(range(8, 16))}
    split = stratified_split(groups, (0.5, 0.5, 0.0), seed=0)
    train_ids = split["left"]["train"] + split["right"]["train"]
    expected = inference_ids(list(range(16)), train_ids)
    shown = [item["index"] for item in summary["inference"]]
    assert shown == expected
    assert set(shown).isdisjoint(train_ids)
    assert set(shown).isdisjoint(set(range(16, 20)))
    assert set(shown) | set(train_ids) == set(range(16))
    assert all(0.0 <= float(item["score"]) <= 1.0 for item in summary["inference"])
    loaded = SessionStore(tmp_path / "sessions").get(session_id)
    assert [item["index"] for item in loaded.task_summary["inference"]] == expected
