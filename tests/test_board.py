"""Board labeling: scoped recommend, hierarchy files, and stroke metrics."""

import h5py
import numpy as np

from hitl_features.distances import mean_cross_distance, mean_within_distance
from hitl_features.labels import export_hierarchy_csv, graph_from_hierarchy, parse_hierarchy_csv
from hitl_features.session import SessionStore
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


def test_inside_category_does_not_recommend_outside_samples(tmp_path):
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    client, session_id = _session(tmp_path, matrix, [0, 1, 2, 3, 4])
    session = client.application.config["STORE"].get(session_id)
    parent = session.graph.create_category("spiral")
    child = session.graph.create_category("barred", parent_id=parent)
    other = session.graph.create_category("smooth")
    session.graph.assign(0, child)
    session.graph.assign(1, child)
    session.graph.assign(2, other)
    session.graph.assign(3, parent)
    entered = client.post(f"/api/sessions/{session_id}/view", json={"category_id": parent})
    assert entered.status_code == 200
    chosen = client.post(
        f"/api/sessions/{session_id}/recommend",
        json={"mechanism": "random", "category_ids": []},
    )
    assert chosen.status_code == 200
    assert chosen.get_json()["chosen"] == [3]
    limited = client.post(
        f"/api/sessions/{session_id}/recommend",
        json={"mechanism": "certain", "category_ids": [child]},
    )
    assert limited.get_json()["chosen"] == [3]


def test_empty_recommend_set_uses_every_category_at_this_level(tmp_path):
    matrix = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    client, session_id = _session(tmp_path, matrix, [0, 1, 2, 3])
    session = client.application.config["STORE"].get(session_id)
    left = session.graph.create_category("left")
    right = session.graph.create_category("right")
    session.graph.assign(0, left)
    session.graph.assign(1, right)
    only_left = client.post(
        f"/api/sessions/{session_id}/recommend",
        json={"mechanism": "unknown", "category_ids": [left]},
    )
    assert only_left.status_code == 200
    assert only_left.get_json()["chosen"][0] == 3
    both = client.post(
        f"/api/sessions/{session_id}/recommend",
        json={"mechanism": "unknown", "category_ids": []},
    )
    assert both.status_code == 200
    assert set(both.get_json()["chosen"]) <= {2, 3}


def test_hierarchy_csv_round_trip_and_trains(tmp_path):
    matrix = np.ones((13, 3), dtype=np.float32)
    client, session_id = _session(tmp_path, matrix, [10, 11, 12])
    csv_text = "index,label_1,label_2\n10,spiral,barred\n11,spiral,plain\n12,smooth,\n99,other,\n"
    imported = client.post(
        f"/api/sessions/{session_id}/import-labels",
        json={"csv": csv_text},
    )
    assert imported.status_code == 200, imported.get_json()
    payload = imported.get_json()
    assert payload["imported"]["n_assigned"] == 3
    assert payload["imported"]["n_skipped"] == 1
    session = client.application.config["STORE"].get(session_id)
    barred = session.graph.category_of(10)
    assert session.graph.categories[barred].name == "barred"
    assert session.graph.categories[session.graph.categories[barred].parent_id].name == "spiral"
    assert session.graph.categories[session.graph.category_of(12)].parent_id is None
    exported = client.get(f"/api/sessions/{session_id}/labels.csv")
    assert exported.status_code == 200
    text = exported.data.decode()
    assert text.splitlines()[0] == "index,label_1,label_2"
    rows = parse_hierarchy_csv(text)
    assert rows == [(10, ["spiral", "barred"]), (11, ["spiral", "plain"]), (12, ["smooth"])]
    again = export_hierarchy_csv(graph_from_hierarchy(rows))
    assert "10,spiral,barred" in again


def test_task_class_keeps_several_categories(tmp_path):
    matrix = np.eye(4, dtype=np.float32)
    client, session_id = _session(tmp_path, matrix, [0, 1, 2, 3])
    session = client.application.config["STORE"].get(session_id)
    left = session.graph.create_category("left")
    right = session.graph.create_category("right")
    session.graph.assign(0, left)
    session.graph.assign(1, right)
    drafted = client.post(
        f"/api/sessions/{session_id}/task-draft",
        json={
            "classes": [{"name": "broad", "category_ids": [left, right]}],
            "order": ["broad"],
            "strength": 1,
        },
    )
    assert drafted.status_code == 200, drafted.get_json()
    classes = drafted.get_json()["snapshot"]["task_draft"]["classes"]
    assert [item["name"] for item in classes] == ["broad"]
    assert classes[0]["category_ids"] == [left, right]
    stored = client.application.config["STORE"].get(session_id)
    assert stored.task_draft["classes"][0]["category_ids"] == [left, right]


def test_mapped_distance_thickens_with_separation():
    close = mean_within_distance(np.array([[1.0, 0.0], [1.0, 0.0]]))
    apart = mean_cross_distance(np.array([[1.0, 0.0]]), np.array([[-1.0, 0.0]]))
    assert close == 0.0
    assert apart == 2.0
    assert apart > close
