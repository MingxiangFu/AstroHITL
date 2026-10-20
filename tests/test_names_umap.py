"""Sibling names, one training feature space, and the pool UMAP."""

import h5py
import numpy as np
import pytest

from hitl_features.graph import CategoryGraph
from hitl_features.labels import merge_hierarchy
from hitl_features.modelsizes import head_parameter_count
from hitl_features.umap_layout import embed_points
from hitl_features.web.app import create_app


def test_import_reuses_a_sibling_and_a_child_may_repeat_its_parent():
    graph = CategoryGraph()
    parent = graph.create_category("barred")
    merge_hierarchy(graph, [(1, ["barred"]), (2, ["barred", "barred"])])
    roots = [cat for cat in graph.children(None) if cat.id != "recycle"]
    assert [cat.id for cat in roots] == [parent]
    child = graph.category_of(2)
    assert graph.categories[child].parent_id == parent
    assert graph.name_path(child) == "barred|barred"
    assert graph.samples(parent) == [1]
    again = graph.create_category("Recycle")
    assert again != "recycle"


def test_embed_points_uses_a_fixed_layout_below_four_rows():
    small = embed_points(np.ones((3, 2), dtype=np.float32), seed=0)
    assert small.shape == (3, 2)
    assert np.isfinite(small).all()


def _client(tmp_path, matrix, pool, out_dim=2):
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
            "out_dim": out_dim,
            "round_size": 2,
            "epochs": 1,
            "name": "demo",
            "device": "cpu",
        },
    )
    assert created.status_code == 200, created.get_json()
    return client, created.get_json()["snapshot"]


def test_head_counts_follow_the_selected_feature_width(tmp_path):
    matrix = np.ones((4, 4500), dtype=np.float32)
    client, snapshot = _client(tmp_path, matrix, [0, 1, 2, 3], out_dim=2)
    session_id = snapshot["id"]
    session = client.application.config["STORE"].get(session_id)
    assert snapshot["feature_source"] == "raw"
    assert snapshot["mapper_trained"] is False
    small = next(item for item in snapshot["head_sizes"] if item["hidden"] == [64, 32])
    assert small["parameters"] == head_parameter_count(4500, 2, (64, 32))
    refused_mapped = client.post(
        f"/api/sessions/{session_id}/feature-source",
        json={"features": "mapped"},
    )
    assert refused_mapped.status_code == 400
    session.mapper_trained = True
    mapped = client.post(
        f"/api/sessions/{session_id}/feature-source",
        json={"features": "mapped"},
    )
    assert mapped.status_code == 200, mapped.get_json()
    mapped_body = mapped.get_json()["snapshot"]
    assert mapped_body["feature_source"] == "mapped"
    mapped_small = next(item for item in mapped_body["head_sizes"] if item["hidden"] == [64, 32])
    assert mapped_small["parameters"] == head_parameter_count(2, 2, (64, 32))
    wide = client.post(
        f"/api/sessions/{session_id}/head-size",
        json={"hidden": [2048, 1024]},
    )
    assert wide.status_code == 200, wide.get_json()
    switched = client.post(
        f"/api/sessions/{session_id}/feature-source",
        json={"features": "raw"},
    )
    assert switched.status_code == 200, switched.get_json()
    body = switched.get_json()["snapshot"]
    assert body["feature_source"] == "raw"
    assert body["head_hidden"] == [64, 32]
    raw_small = next(item for item in body["head_sizes"] if item["hidden"] == [64, 32])
    assert raw_small["parameters"] == head_parameter_count(4500, 2, (64, 32))
    assert session.feature_source == "raw"
    refused = client.post(
        f"/api/sessions/{session_id}/feature-source",
        json={"features": "both"},
    )
    assert refused.status_code == 400


def test_classification_trains_only_the_selected_arm(tmp_path):
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(12, 8)).astype(np.float32)
    matrix[:6, 0] += 2
    client, snapshot = _client(tmp_path, matrix, list(range(12)), out_dim=2)
    session_id = snapshot["id"]
    session = client.application.config["STORE"].get(session_id)
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
        f"/api/sessions/{session_id}/tasks/classification",
        json={"classes": classes, "epochs": 1, "fractions": {"train": 0.5, "val": 0.5, "test": 0}},
    )
    assert trained.status_code == 200, trained.get_json()
    result = trained.get_json()["result"]
    summary = trained.get_json()["snapshot"]["task_summary"]
    assert result["raw"]["accuracy"] >= 0
    assert result["mapped"] is None
    assert summary["splits"]["train"]["raw_accuracy"] is not None
    assert summary["splits"]["train"]["mapped_accuracy"] is None
    assert summary["inference"]
    task_dir = session.directory / "tasks" / "t1"
    assert (task_dir / "raw.pt").is_file()
    assert not (task_dir / "mapped.pt").is_file()
    barred = client.post(
        f"/api/sessions/{session_id}/task-draft",
        json={"classes": [{"name": "a|b", "category_ids": []}], "order": [], "strength": 1},
    )
    assert barred.status_code == 400
    raw = client.post(
        f"/api/sessions/{session_id}/tasks/ranking",
        json={
            "classes": classes,
            "order": ["left", "right"],
            "strength": 1,
            "epochs": 1,
            "features": "raw",
        },
    )
    assert raw.status_code == 200, raw.get_json()
    ranking = raw.get_json()["result"]
    assert ranking["raw_pairwise_accuracy"] is not None
    assert ranking["mapped_pairwise_accuracy"] is None
    assert session.feature_source == "raw"
    assert (session.directory / "tasks" / "t2" / "raw.pt").is_file()


def test_umap_failure_is_reported_instead_of_an_empty_body(tmp_path, monkeypatch):
    matrix = np.arange(24, dtype=np.float32).reshape(6, 4)
    client, snapshot = _client(tmp_path, matrix, list(range(6)), out_dim=2)
    session_id = snapshot["id"]

    def broken(values, seed):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr("hitl_features.umap_layout.embed_points", broken)
    failed = client.post(f"/api/sessions/{session_id}/umap", json={"stream": True})
    assert failed.status_code == 200
    text = failed.get_data(as_text=True)
    assert '"event": "error"' in text
    assert "embedding failed" in text


def test_loaded_mapper_moves_onto_the_session_device(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    matrix = np.ones((4, 8), dtype=np.float32)
    client, snapshot = _client(tmp_path, matrix, [0, 1, 2, 3], out_dim=2)
    session = client.application.config["STORE"].get(snapshot["id"])
    session.mapper.cpu()
    session.device = "cuda"
    rows = session._rows([0, 1], mapped=True)
    assert rows.shape == (2, session.out_dim)
    assert next(session.mapper.parameters()).is_cuda


def test_umap_draws_the_pool_and_recolors_without_a_second_fit(tmp_path, monkeypatch):
    matrix = np.arange(24, dtype=np.float32).reshape(6, 4)
    client, snapshot = _client(tmp_path, matrix, list(range(6)), out_dim=2)
    session_id = snapshot["id"]
    session = client.application.config["STORE"].get(session_id)
    calls = {"n": 0}
    real = embed_points

    def counted(values, seed):
        calls["n"] += 1
        return real(values, seed)

    monkeypatch.setattr("hitl_features.umap_layout.embed_points", counted)
    first = client.post(f"/api/sessions/{session_id}/umap", json={})
    assert first.status_code == 200, first.get_json()
    payload = first.get_json()["umap"]
    assert len(payload["raw"]) == 6
    assert len(payload["mapped"]) == 6
    assert all(item["color"] is None for item in payload["raw"])
    assert calls["n"] == 2
    left = session.graph.create_category("left")
    session.graph.assign(1, left)
    session.graph.assign(2, "recycle")
    second = client.post(f"/api/sessions/{session_id}/umap", json={})
    assert second.status_code == 200, second.get_json()
    colored = second.get_json()["umap"]["raw"]
    by_index = {item["index"]: item for item in colored}
    assert by_index[1]["color"]
    assert by_index[2]["color"] is None
    assert by_index[0]["color"] is None
    assert calls["n"] == 2
    raw_xy = [(item["index"], item["x"], item["y"]) for item in colored]
    assert np.isfinite([item["x"] for item in colored]).all()
    session._bump_layout()
    third = client.post(f"/api/sessions/{session_id}/umap", json={})
    assert third.status_code == 200, third.get_json()
    refit = third.get_json()["umap"]["raw"]
    assert [(item["index"], item["x"], item["y"]) for item in refit] == raw_xy
    assert calls["n"] == 3
