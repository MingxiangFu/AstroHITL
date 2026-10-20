"""Subclass recommend, within-category screening, and full-pool result projection."""

import h5py
import numpy as np
import pytest

from hitl_features.session import SessionStore, category_color
from hitl_features.web.app import create_app


def _session(tmp_path, matrix, pool, round_size=8):
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
            "out_dim": 2,
            "round_size": round_size,
            "epochs": 1,
            "name": "demo",
            "device": "cpu",
        },
    )
    assert created.status_code == 200, created.get_json()
    session_id = created.get_json()["snapshot"]["id"]
    return client, client.application.config["STORE"].get(session_id)


def test_subclass_recommend_uses_the_full_pool_or_says_why(tmp_path, monkeypatch):
    matrix = np.zeros((34, 2), dtype=np.float32)
    matrix[:15, 0] = 1
    matrix[15:30, 1] = 1
    matrix[30, 0] = 1
    matrix[31, 1] = 1
    matrix[32, 0] = 1
    matrix[33, 0] = 1
    _client, session = _session(tmp_path, matrix, list(range(34)))
    parent = session.graph.create_category("A")
    child_a = session.graph.create_category("a", parent_id=parent)
    child_b = session.graph.create_category("b", parent_id=parent)
    other = session.graph.create_category("other")
    for index in range(15):
        session.graph.assign(index, child_a)
    for index in range(15, 30):
        session.graph.assign(index, child_b)
    session.graph.assign(32, parent)
    session.graph.assign(33, other)
    session.recommend_set = [child_a]
    with pytest.raises(ValueError, match="subclass recommend needs a trained mapper"):
        session.recommend_round("random")

    session.mapper_trained = True
    monkeypatch.setattr(session, "_pool_matrix", lambda mapped=False: session.features)
    lonely = session.graph.create_category("lonely")
    lonely_child = session.graph.create_category("lonely-child", parent_id=lonely)
    session.recommend_set = [lonely_child]
    with pytest.raises(ValueError, match="subclass recommend needs labeled examples"):
        session.recommend_round("random")

    session.recommend_set = [child_a]
    from_root = set(session.recommend_round("random"))
    session.view_parent = parent
    from_parent = set(session.recommend_round("random"))
    assert from_root == {30, 32, 33}
    assert from_parent == {30, 32, 33}
    assert 31 not in from_root

    session.recommend_round("random", within_labeled=True)
    assert session.round_ids == [32]


def test_projection_rename_and_delete_cover_both_task_kinds(tmp_path):
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(12, 4)).astype(np.float32)
    matrix[:6, 0] += 4
    matrix[6:, 0] -= 4
    client, session = _session(tmp_path, matrix, list(range(12)), round_size=2)
    left = session.graph.create_category("left")
    right = session.graph.create_category("right")
    for index in range(6):
        session.graph.assign(index, left)
    for index in range(6, 12):
        session.graph.assign(index, right)
    classes = [
        {"name": "left", "category_ids": [left]},
        {"name": "right", "category_ids": [right]},
    ]
    trained = client.post(
        f"/api/sessions/{session.session_id}/tasks/classification",
        json={"classes": classes, "epochs": 1, "features": "raw"},
    )
    assert trained.status_code == 200, trained.get_json()
    inference = trained.get_json()["snapshot"]["task_summary"]["inference"]
    assert inference
    assert all(item["label"] in {"left", "right"} for item in inference)
    assert all(item["color"] == category_color(left) or item["color"] == category_color(right) for item in inference)

    projected = client.post(
        f"/api/sessions/{session.session_id}/tasks/project",
        json={"count": 4},
    )
    assert projected.status_code == 200, projected.get_json()
    images = projected.get_json()["projection"]["images"]
    assert len(images) == 4
    assert {item["index"] for item in images} <= set(range(12))
    assert all(item["label"] in {"left", "right"} and item["color"] for item in images)

    ranked = client.post(
        f"/api/sessions/{session.session_id}/tasks/ranking",
        json={
            "classes": classes,
            "order": ["left", "right"],
            "strength": 1,
            "epochs": 1,
            "features": "raw",
        },
    )
    assert ranked.status_code == 200, ranked.get_json()
    ranked_projection = client.post(
        f"/api/sessions/{session.session_id}/tasks/project",
        json={"count": 3},
    )
    assert ranked_projection.status_code == 200, ranked_projection.get_json()
    ranked_images = ranked_projection.get_json()["projection"]["images"]
    assert len(ranked_images) == 3
    assert all(item["label"] is None for item in ranked_images)
    assert all(isinstance(item["score"], float) for item in ranked_images)

    renamed = client.post(
        f"/api/sessions/{session.session_id}/tasks/rename",
        json={"task_id": "t2", "name": "Kept result"},
    )
    assert renamed.status_code == 200, renamed.get_json()
    names = {item["id"]: item["name"] for item in renamed.get_json()["snapshot"]["tasks"]}
    assert names["t2"] == "Kept result"
    refused = client.post(
        f"/api/sessions/{session.session_id}/tasks/rename",
        json={"task_id": "t2", "name": "a|b"},
    )
    assert refused.status_code == 400

    deleted = client.post(
        f"/api/sessions/{session.session_id}/tasks/delete",
        json={"task_id": "t2"},
    )
    assert deleted.status_code == 200, deleted.get_json()
    snapshot = deleted.get_json()["snapshot"]
    assert [item["id"] for item in snapshot["tasks"]] == ["t1"]
    assert snapshot["tasks"][0]["active"] is True
    assert snapshot["projection"] is None


def test_certain_caption_names_the_scope_not_the_pool_vote(tmp_path, monkeypatch):
    """Pool kNN can vote another root while certainty still ranks distance to A."""
    matrix = np.zeros((24, 2), dtype=np.float32)
    matrix[:2, 0] = 1.0
    matrix[2:22, 1] = 1.0
    matrix[22, 0] = 1.0
    matrix[23] = [0.2, 0.9]
    _client, session = _session(tmp_path, matrix, list(range(24)), round_size=2)
    scope_a = session.graph.create_category("scope-a")
    scope_b = session.graph.create_category("scope-b")
    for index in range(2):
        session.graph.assign(index, scope_a)
    for index in range(2, 22):
        session.graph.assign(index, scope_b)
    session.mapper_trained = True
    monkeypatch.setattr(session, "_pool_matrix", lambda mapped=False: session.features)
    chosen = session.recommend_round("certain", [scope_a])
    assert chosen == [22, 23]
    images = [item for item in session.snapshot()["images"] if item["role"] == "unplaced"]
    assert [item["index"] for item in images] == [22, 23]
    assert images[0]["prediction"] == "scope-b"
    assert images[0]["label"] == "scope-a"
    assert images[1]["label"] == "scope-a"
    assert images[0]["score"] > images[1]["score"]
    assert images[0]["color"] == category_color(scope_a)


def test_empty_scope_keeps_a_scored_slice_and_names_that_class(tmp_path, monkeypatch):
    """No kNN hit in the scope still recommends the closest slice of the pool."""
    matrix = np.zeros((31, 2), dtype=np.float32)
    matrix[:2, 0] = 1.0
    matrix[2:4, 0] = -1.0
    matrix[4:24, 1] = 1.0
    matrix[24, 1] = 1.0
    matrix[25, 0] = 1.0
    matrix[26:30, 1] = 1.0
    matrix[30, 0] = -1.0
    client, session = _session(tmp_path, matrix, list(range(31)), round_size=2)
    parent = session.graph.create_category("group")
    child_a = session.graph.create_category("near-a", parent_id=parent)
    child_c = session.graph.create_category("near-c", parent_id=parent)
    child_b = session.graph.create_category("other", parent_id=parent)
    session.graph.assign(0, child_a)
    session.graph.assign(1, child_a)
    session.graph.assign(2, child_c)
    session.graph.assign(3, child_c)
    for index in range(4, 24):
        session.graph.assign(index, child_b)
    session.mapper_trained = True
    monkeypatch.setattr(session, "_pool_matrix", lambda mapped=False: session.features)

    chosen = session.recommend_round("diversity", [child_a, child_c], scope_factor=1)
    assert chosen == [25, 30]
    images = [item for item in session.snapshot()["images"] if item["role"] == "unplaced"]
    assert [item["index"] for item in images] == [25, 30]
    assert [item["label"] for item in images] == ["near-a", "near-c"]
    assert images[0]["prediction"] == "group"
    assert images[1]["prediction"] == "group"
    assert abs(images[0]["score"] - (2 / 15)) < 1e-6
    assert abs(images[1]["score"] - (2 / 15)) < 1e-6
    assert images[0]["color"] == category_color(child_a)
    assert images[1]["color"] == category_color(child_c)

    wider = session.recommend_round("diversity", scope_factor=2)
    assert wider == [24, 25]
    assert 30 not in wider
    assert session.snapshot()["scope_factor"] == 2
    refused = client.post(
        f"/api/sessions/{session.session_id}/recommend",
        json={"mechanism": "diversity", "scope_factor": 0},
    )
    assert refused.status_code == 400
    from hitl_features.session import SessionStore

    loaded = SessionStore(tmp_path / "sessions").get(session.session_id)
    assert loaded.scope_factor == 2
