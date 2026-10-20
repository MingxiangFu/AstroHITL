import h5py
import numpy as np

from hitl_features.score_board import fixed_score_indices
from hitl_features.web.app import create_app


def _app(tmp_path, matrix, pool, round_size=4):
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
            "round_size": round_size,
            "epochs": 1,
            "name": "demo",
            "device": "cpu",
        },
    )
    assert created.status_code == 200
    return client, created.get_json()["snapshot"]["id"]


def test_parent_count_and_inside_recommend_stays_in_the_category(tmp_path):
    matrix = np.zeros((6, 2), dtype=np.float32)
    matrix[0] = [1.0, 0.0]
    matrix[1] = [0.9, 0.1]
    matrix[2] = [-1.0, 0.0]
    matrix[3] = [1.0, 0.05]
    matrix[4] = [-1.0, 0.05]
    matrix[5] = [0.0, 1.0]
    client, session_id = _app(tmp_path, matrix, [0, 1, 2, 3, 4, 5], round_size=1)
    store = client.application.config["STORE"]
    session = store.get(session_id)
    parent = session.graph.create_category("spiral")
    child = session.graph.create_category("barred", parent_id=parent)
    other = session.graph.create_category("smooth")
    session.graph.assign(0, child, round_index=1)
    session.graph.assign(1, child, round_index=1)
    session.graph.assign(2, other, round_index=1)
    snapshot = session.snapshot()
    spiral = [cat for cat in snapshot["categories"] if cat["name"] == "spiral"][0]
    assert spiral["count"] == 2
    assert 0 in spiral["recent"] or 1 in spiral["recent"]
    assert set(spiral["recent"]).isdisjoint(spiral["representative"])
    session.graph.assign(3, parent)
    entered = client.post(
        f"/api/sessions/{session_id}/view",
        json={"category_id": parent},
    )
    assert entered.status_code == 200
    refused = client.post(
        f"/api/sessions/{session_id}/recommend",
        json={"mechanism": "certain"},
    )
    assert refused.status_code == 400
    assert "trained mapper" in refused.get_json()["error"]
    chosen = client.post(
        f"/api/sessions/{session_id}/recommend",
        json={"mechanism": "certain", "within_labeled": True},
    )
    assert chosen.status_code == 200
    assert chosen.get_json()["chosen"] == [3]


def test_new_recommendation_replaces_unplaced_and_keeps_assigned(tmp_path):
    matrix = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    client, session_id = _app(tmp_path, matrix, [0, 1, 2, 3], round_size=3)
    first = client.post(
        f"/api/sessions/{session_id}/recommend",
        json={"mechanism": "random"},
    )
    assert first.status_code == 200
    images = [item["index"] for item in first.get_json()["snapshot"]["images"]]
    assert len(images) == 3
    placed = client.post(
        f"/api/sessions/{session_id}/place",
        json={"sample_index": images[0], "new_name": "alpha"},
    )
    assert placed.status_code == 200
    second = client.post(
        f"/api/sessions/{session_id}/recommend",
        json={"mechanism": "diversity"},
    )
    assert second.status_code == 200
    body = second.get_json()
    again = [item["index"] for item in body["snapshot"]["images"]]
    assert images[0] not in again
    assert set(again) == {0, 1, 2, 3} - {images[0]}
    assert set(body["chosen"]) == set(again)
    assert images[1] in again and images[2] in again


def test_task_training_bins_the_same_fixed_pool(tmp_path):
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(12, 6)).astype(np.float32)
    matrix[:6, 0] += 2
    client, session_id = _app(tmp_path, matrix, list(range(12)), round_size=12)
    store = client.application.config["STORE"]
    session = store.get(session_id)
    left = session.graph.create_category("left")
    right = session.graph.create_category("right")
    for index in range(6):
        session.graph.assign(index, left, round_index=1)
    for index in range(6, 12):
        session.graph.assign(index, right, round_index=1)
    classes = [
        {"name": "left", "category_ids": [left]},
        {"name": "right", "category_ids": [right]},
    ]
    classified = client.post(
        f"/api/sessions/{session_id}/tasks/classification",
        json={"classes": classes, "epochs": 1},
    )
    assert classified.status_code == 200
    board = classified.get_json()["snapshot"]["score_board"]
    assert board["kind"] == "classification"
    assert board["indices"] == fixed_score_indices(list(range(12)), seed=0)
    assert len(board["bins"]) == 5
    assert all(len(item["images"]) <= 4 for item in board["bins"])
    assert all(image["label"] in {"left", "right"} for item in board["bins"] for image in item["images"])
    again = client.post(
        f"/api/sessions/{session_id}/tasks/classification",
        json={"classes": classes, "epochs": 1},
    )
    assert again.get_json()["snapshot"]["score_board"]["indices"] == board["indices"]
    ranked = client.post(
        f"/api/sessions/{session_id}/tasks/ranking",
        json={"classes": classes, "order": ["left", "right"], "strength": 1, "epochs": 1},
    )
    assert ranked.status_code == 200
    rank_board = ranked.get_json()["snapshot"]["score_board"]
    assert rank_board["kind"] == "ranking"
    assert rank_board["indices"] == board["indices"]
    assert all("label" not in image for item in rank_board["bins"] for image in item["images"])
    assigned = {image["index"] for item in rank_board["bins"] for image in item["images"] if image["in_graph"]}
    assert assigned
    kept = client.post(
        f"/api/sessions/{session_id}/place",
        json={"sample_index": sorted(assigned)[0], "category_id": left},
    )
    assert kept.status_code == 200
    kept_board = kept.get_json()["snapshot"]["score_board"]
    still = [
        image
        for item in kept_board["bins"]
        for image in item["images"]
        if image["index"] == sorted(assigned)[0]
    ]
    assert still and still[0]["in_graph"] is True
    kinds = [task["kind"] for task in ranked.get_json()["snapshot"]["tasks"]]
    assert kinds == ["classification", "classification", "ranking"]
    first = ranked.get_json()["snapshot"]["tasks"][0]
    selected = client.post(
        f"/api/sessions/{session_id}/tasks/select",
        json={"task_id": first["id"]},
    )
    assert selected.status_code == 200
    assert selected.get_json()["snapshot"]["score_board"]["kind"] == "classification"
    assert selected.get_json()["snapshot"]["score_board"]["task_id"] == first["id"]
    task_root = session.directory / "tasks"
    assert (task_root / "t1" / "result.json").is_file()
    assert (task_root / "t2" / "result.json").is_file()
    assert (task_root / "t3" / "result.json").is_file()


def test_score_image_can_be_placed_and_stays_in_the_result(tmp_path):
    matrix = np.eye(8, dtype=np.float32)
    client, session_id = _app(tmp_path, matrix, list(range(8)), round_size=4)
    session = client.application.config["STORE"].get(session_id)
    left = session.graph.create_category("left")
    right = session.graph.create_category("right")
    for index in (0, 1):
        session.graph.assign(index, left)
    for index in (2, 3):
        session.graph.assign(index, right)
    classes = [
        {"name": "left", "category_ids": [left]},
        {"name": "right", "category_ids": [right]},
    ]
    trained = client.post(
        f"/api/sessions/{session_id}/tasks/classification",
        json={"classes": classes, "epochs": 1},
    )
    assert trained.status_code == 200
    images = [
        image
        for item in trained.get_json()["snapshot"]["score_board"]["bins"]
        for image in item["images"]
    ]
    fresh = next(image for image in images if image["index"] not in {0, 1, 2, 3})
    assert fresh["in_graph"] is False
    placed = client.post(
        f"/api/sessions/{session_id}/place",
        json={"sample_index": fresh["index"], "category_id": left},
    )
    assert placed.status_code == 200
    after = [
        image
        for item in placed.get_json()["snapshot"]["score_board"]["bins"]
        for image in item["images"]
        if image["index"] == fresh["index"]
    ]
    assert after and after[0]["in_graph"] is True
    left_ids = [item["index"] for item in placed.get_json()["snapshot"]["images"]]
    assert fresh["index"] not in left_ids


def test_unassign_returns_the_sample_to_the_root_waiting_list(tmp_path):
    matrix = np.eye(4, dtype=np.float32)
    client, session_id = _app(tmp_path, matrix, [0, 1, 2, 3], round_size=2)
    placed = client.post(
        f"/api/sessions/{session_id}/place",
        json={"sample_index": 1, "new_name": "alpha"},
    )
    assert placed.status_code == 200
    category_id = placed.get_json()["snapshot"]["categories"][0]["id"]
    removed = client.post(
        f"/api/sessions/{session_id}/unassign",
        json={"sample_index": 1},
    )
    assert removed.status_code == 200
    body = removed.get_json()["snapshot"]
    assert body["view_parent"] is None
    assert body["categories"][0]["count"] == 0
    waiting = next(image for image in body["images"] if image["index"] == 1)
    assert waiting["role"] == "waiting"
    session = client.application.config["STORE"].get(session_id)
    assert session.graph.category_of(1) is None
    again = client.post(
        f"/api/sessions/{session_id}/place",
        json={"sample_index": 1, "category_id": category_id},
    )
    assert again.status_code == 200
    assert again.get_json()["snapshot"]["categories"][0]["count"] == 1


def test_card_position_is_saved_and_reloaded(tmp_path):
    matrix = np.eye(3, dtype=np.float32)
    client, session_id = _app(tmp_path, matrix, [0, 1, 2], round_size=1)
    placed = client.post(
        f"/api/sessions/{session_id}/place",
        json={"sample_index": 0, "new_name": "alpha"},
    )
    category_id = placed.get_json()["snapshot"]["categories"][0]["id"]
    moved = client.post(
        f"/api/sessions/{session_id}/position",
        json={"category_id": category_id, "x": 0.2, "y": 0.4},
    )
    assert moved.status_code == 200
    card = moved.get_json()["snapshot"]["categories"][0]
    assert card["x"] == 0.2 and card["y"] == 0.4
    session = client.application.config["STORE"].get(session_id)
    from hitl_features.session import LabSession

    loaded = LabSession.load(session.directory)
    assert loaded.graph.positions[category_id] == (0.2, 0.4)
    rejected = client.post(
        f"/api/sessions/{session_id}/position",
        json={"category_id": category_id, "x": 1.5, "y": 0.4},
    )
    assert rejected.status_code == 400


def test_label_file_builds_a_graph_and_leaves_the_session_editable(tmp_path):
    matrix = np.eye(4, dtype=np.float32)
    feature_path = tmp_path / "feat.h5"
    import h5py

    with h5py.File(feature_path, "w") as handle:
        handle.create_dataset("mae_features", data=matrix)
        handle.create_dataset("index", data=np.array([10, 20, 30, 40], dtype=np.int64))
    app = create_app(tmp_path / "sessions")
    client = app.test_client()
    created = client.post(
        "/api/sessions",
        json={
            "feature_path": str(feature_path),
            "pool_indices": [10, 40],
            "pool_size": 2,
            "seed": 0,
            "out_dim": 4,
            "round_size": 1,
            "epochs": 1,
            "name": "demo",
            "device": "cpu",
        },
    )
    assert created.status_code == 200
    session_id = created.get_json()["snapshot"]["id"]
    imported = client.post(
        f"/api/sessions/{session_id}/import-labels",
        json={
            "csv": "index,label\n10,spiral\n20,smooth\n30,smooth\n99,other\n",
            "index_column": "index",
            "label_column": "label",
        },
    )
    assert imported.status_code == 200, imported.get_json()
    payload = imported.get_json()
    names = {cat["name"] for cat in payload["snapshot"]["all_categories"]}
    assert names == {"spiral", "smooth"}
    assert payload["imported"]["n_skipped"] == 1
    assert payload["imported"]["n_assigned"] == 3
    session = client.application.config["STORE"].get(session_id)
    assert session.pool_indices == [10, 40, 20, 30]
    assert np.allclose(session.features[session.index_to_row[20]], matrix[1])
    assert session.graph.category_of(40) is None
    placed = client.post(
        f"/api/sessions/{session_id}/place",
        json={"sample_index": 40, "new_name": "extra"},
    )
    assert placed.status_code == 200
    assert {cat["name"] for cat in placed.get_json()["snapshot"]["all_categories"]} == {
        "spiral",
        "smooth",
        "extra",
    }
