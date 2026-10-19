"""Recycle bin, training with a full recommend list, and chosen CSV columns."""

import h5py
import numpy as np

from hitl_features.tasks import task_labels
from hitl_features.web.app import create_app


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


def test_mapper_trains_while_images_remain_unplaced(tmp_path, monkeypatch):
    seen = {}

    def spy(mapper, features, sample_ids, graph, **kwargs):
        seen["ids"] = list(sample_ids)
        return {"loss": 0.0, "history": [0.0], "steps": 1}

    monkeypatch.setattr("hitl_features.session.train_mapper", spy)
    matrix = np.eye(4, dtype=np.float32)
    client, session_id = _session(tmp_path, matrix, [0, 1, 2, 3])
    session = client.application.config["STORE"].get(session_id)
    left = session.graph.create_category("left")
    right = session.graph.create_category("right")
    session.graph.assign(0, left)
    session.graph.assign(1, right)
    session.unplaced = {2, 3}
    trained = client.post(f"/api/sessions/{session_id}/confirm", json={"epochs": 1})
    assert trained.status_code == 200, trained.get_json()
    assert seen["ids"] == [0, 1]
    body = trained.get_json()["snapshot"]
    assert body["can_confirm"] is True
    assert {item["index"] for item in body["images"] if item["role"] == "unplaced"} == {2, 3}


def test_recycle_bin_is_left_out_of_training_knn_and_tasks(tmp_path, monkeypatch):
    seen = {}

    def spy(mapper, features, sample_ids, graph, **kwargs):
        seen["ids"] = list(sample_ids)
        return {"loss": 0.0, "history": [0.0], "steps": 1}

    monkeypatch.setattr("hitl_features.session.train_mapper", spy)
    matrix = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    client, session_id = _session(tmp_path, matrix, [0, 1, 2, 3, 4])
    session = client.application.config["STORE"].get(session_id)
    left = session.graph.create_category("left")
    right = session.graph.create_category("right")
    session.graph.assign(0, left)
    session.graph.assign(1, left)
    session.graph.assign(2, right)
    moved = client.post(f"/api/sessions/{session_id}/recycle", json={"sample_index": 1})
    assert moved.status_code == 200, moved.get_json()
    snap = moved.get_json()["snapshot"]
    assert snap["recycle"]["count"] == 1
    assert all(cat["id"] != "recycle" for cat in snap["categories"])
    assert all(cat["id"] != "recycle" for cat in snap["all_categories"])
    assert [item["index"] for item in snap["labeled_images"]] == [0, 2]
    refused = client.post(
        f"/api/sessions/{session_id}/task-draft",
        json={"classes": [{"name": "bin", "category_ids": ["recycle"]}]},
    )
    assert refused.status_code == 400
    labels = task_labels(session.graph, [("Left", [left]), ("Right", [right])])
    assert 1 not in labels
    assert labels[0] == "Left"
    trained = client.post(f"/api/sessions/{session_id}/confirm", json={"epochs": 1})
    assert trained.status_code == 200, trained.get_json()
    assert seen["ids"] == [0, 2]
    chosen = client.post(
        f"/api/sessions/{session_id}/recommend",
        json={"mechanism": "certain", "category_ids": [left]},
    )
    assert chosen.status_code == 200, chosen.get_json()
    assert chosen.get_json()["chosen"][0] == 4


def test_import_hierarchy_uses_chosen_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "hitl_features.session.train_mapper",
        lambda *args, **kwargs: {"loss": 0.0, "history": [0.0], "steps": 1},
    )
    matrix = np.ones((13, 3), dtype=np.float32)
    client, session_id = _session(tmp_path, matrix, [10, 11, 12])
    csv_text = "obj,main,sub\n10,spiral,barred\n11,spiral,plain\n12,smooth,\n"
    missing = client.post(
        f"/api/sessions/{session_id}/import-labels",
        json={"csv": csv_text},
    )
    assert missing.status_code == 400
    imported = client.post(
        f"/api/sessions/{session_id}/import-labels",
        json={
            "csv": csv_text,
            "index_column": "obj",
            "label_columns": ["main", "sub"],
        },
    )
    assert imported.status_code == 200, imported.get_json()
    session = client.application.config["STORE"].get(session_id)
    barred = session.graph.category_of(10)
    assert session.graph.categories[barred].name == "barred"
    assert session.graph.categories[session.graph.categories[barred].parent_id].name == "spiral"
    assert session.graph.categories[session.graph.category_of(12)].parent_id is None
    assert session.graph.class_count() == 4
    assert "recycle" not in {
        cat["id"] for cat in imported.get_json()["snapshot"]["all_categories"]
    }
