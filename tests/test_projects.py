"""Named projects stay in the session directory and load after a restart."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from hitl_features.session import SessionStore
from hitl_features.web.app import create_app


def _features(tmp_path: Path) -> Path:
    feature_path = tmp_path / "feat.h5"
    matrix = np.arange(12 * 6, dtype=np.float32).reshape(12, 6)
    matrix[:6, 0] += 4
    with h5py.File(feature_path, "w") as handle:
        handle.create_dataset("features", data=matrix)
    return feature_path


def _create(root: Path, feature_path: Path, name: str):
    return SessionStore(root).create(
        feature_path=str(feature_path),
        pool_size=12,
        seed=0,
        out_dim=4,
        round_size=2,
        epochs=1,
        device="cpu",
        pool_indices=list(range(12)),
        name=name,
    )


def test_leaf_filter_sits_with_the_current_category_browser():
    html = Path("hitl_features/web/templates/index.html").read_text()
    rec = html.split('id="zone-rec"', 1)[1].split('id="zone-graph"', 1)[0]
    images = rec.index('id="images"')
    leaf = rec.index('id="leaf-only-row"')
    well = rec.index('id="well-recommend"')
    assert images < leaf < well
    assert 'id="project-name"' in html
    assert 'id="saved-projects"' in html
    assert 'id="load-project"' in html


def test_project_names_reject_empty_duplicates_and_path_marks(tmp_path: Path):
    root = tmp_path / "sessions"
    feature_path = _features(tmp_path)
    _create(root, feature_path, "  mergers  ")
    store = SessionStore(root)
    assert [item["name"] for item in store.list_projects()] == ["mergers"]
    with pytest.raises(ValueError, match="required"):
        _create(root, feature_path, "   ")
    with pytest.raises(ValueError, match="already exists"):
        _create(root, feature_path, "mergers")
    for bad in ("a|b", "a/b", "a\\b"):
        with pytest.raises(ValueError, match="path separator"):
            _create(root, feature_path, bad)


def test_saved_project_reloads_graph_mapper_and_task_models(tmp_path: Path):
    feature_path = _features(tmp_path)
    app = create_app(tmp_path / "sessions")
    client = app.test_client()
    created = client.post(
        "/api/sessions",
        json={
            "name": "mergers",
            "feature_path": str(feature_path),
            "pool_indices": list(range(12)),
            "pool_size": 12,
            "seed": 0,
            "out_dim": 4,
            "round_size": 2,
            "epochs": 1,
            "device": "cpu",
        },
    )
    assert created.status_code == 200, created.get_json()
    session_id = created.get_json()["snapshot"]["id"]
    session = app.config["STORE"].get(session_id)
    left = session.graph.create_category("left")
    right = session.graph.create_category("right")
    for index in range(6):
        session.graph.assign(index, left)
    for index in range(6, 12):
        session.graph.assign(index, right)
    trained = client.post(f"/api/sessions/{session_id}/confirm", json={"epochs": 1})
    assert trained.status_code == 200, trained.get_json()
    classes = [
        {"name": "left", "category_ids": [left]},
        {"name": "right", "category_ids": [right]},
    ]
    classified = client.post(
        f"/api/sessions/{session_id}/tasks/classification",
        json={
            "classes": classes,
            "epochs": 1,
            "features": "raw",
            "fractions": {"train": 0.5, "val": 0.5, "test": 0},
        },
    )
    assert classified.status_code == 200, classified.get_json()
    ranked = client.post(
        f"/api/sessions/{session_id}/tasks/ranking",
        json={
            "classes": classes,
            "order": ["left", "right"],
            "strength": 1,
            "epochs": 1,
            "features": "raw",
        },
    )
    assert ranked.status_code == 200, ranked.get_json()
    mapper_before = {
        key: value.detach().cpu().clone() for key, value in session.mapper.state_dict().items()
    }
    listed = client.get("/api/sessions")
    assert listed.status_code == 200
    assert listed.get_json()["projects"] == [{"id": session_id, "name": "mergers"}]

    reloaded = create_app(tmp_path / "sessions")
    again = reloaded.test_client().get(f"/api/sessions/{session_id}")
    assert again.status_code == 200
    snapshot = again.get_json()["snapshot"]
    assert snapshot["name"] == "mergers"
    assert snapshot["mapper_trained"] is True
    assert {item["name"] for item in snapshot["all_categories"]} == {"left", "right"}
    assert {item["kind"] for item in snapshot["tasks"]} == {"classification", "ranking"}
    loaded = reloaded.config["STORE"].get(session_id)
    assert (loaded.directory / "mapper.pt").is_file()
    mapper_after = loaded.mapper.state_dict()
    for key, value in mapper_before.items():
        assert torch.allclose(value, mapper_after[key].detach().cpu())
    classification = next(task for task in loaded.tasks if task["kind"] == "classification")
    ranking = next(task for task in loaded.tasks if task["kind"] == "ranking")
    assert (loaded.directory / "tasks" / classification["id"] / "raw.pt").is_file()
    assert (loaded.directory / "tasks" / ranking["id"] / "raw.pt").is_file()
    classifier = loaded._load_classifier(classification)
    with torch.no_grad():
        logits = classifier(torch.from_numpy(loaded.features[:2]))
    assert tuple(logits.shape) == (2, 2)
    assert torch.isfinite(logits).all()
    scores = loaded._load_ranker(ranking).score(loaded.features[:2])
    assert len(scores) == 2
    assert np.isfinite(scores).all()
