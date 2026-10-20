"""Release UI: merged import, model sizes, and streamed training progress."""

from pathlib import Path

import h5py
import numpy as np
import torch

from hitl_features.classify import MlpClassifier
from hitl_features.mapper import FeatureMapper
from hitl_features.modelsizes import (
    CLASSIFIER_CAP,
    classifier_options,
    head_parameter_count,
    mapper_options,
)
from hitl_features.web.app import create_app


def _session(tmp_path, matrix, pool, epochs=1):
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
            "epochs": epochs,
            "name": "demo",
            "device": "cpu",
        },
    )
    assert created.status_code == 200, created.get_json()
    return client, created.get_json()["snapshot"]


def test_board_template_keeps_the_graph_free_of_the_edge_table():
    html = Path("hitl_features/web/templates/index.html").read_text()
    graph = html.split('id="zone-graph"', 1)[1].split('id="zone-task"', 1)[0]
    task = html.split('id="zone-task"', 1)[1]
    assert 'id="edges"' not in graph
    assert 'id="distances"' not in graph
    assert 'id="edges"' in task
    assert 'id="distances"' in task
    assert 'id="mode-recommend"' in html
    assert 'id="mode-labeled"' in html
    assert 'id="mode-recycle"' in html
    assert 'id="mode-mapper"' in html
    assert 'id="mode-task"' in html
    assert 'id="busy-rec"' in html
    assert 'id="busy-task"' in html
    assert 'id="busy-graph"' in html
    assert 'id="umap-status"' in graph
    assert 'id="mode-mapper" type="button" role="radio" aria-checked="true"' in html
    assert 'id="panel-task" hidden' in html
    assert 'id="edge-distance" type="button" role="radio" data-edge="distance" aria-checked="true"' in html
    assert 'id="feature-path" size="72" value="datasets/fama_base/features.h5"' in html
    assert 'id="mode-umap"' in graph
    assert 'id="umap-board"' in graph
    assert 'id="umap-raw"' in graph
    assert 'id="umap-mapped"' in graph
    size_row = task.split('class="head-size-row"', 1)[1].split("</div>", 2)
    size_markup = size_row[0] + size_row[1]
    assert 'id="source-mapped"' in size_markup
    assert 'id="source-raw"' in size_markup
    assert 'id="head-size"' in size_markup


def test_parameter_options_match_the_modules_and_respect_the_cap():
    in_dim, out_dim, n_classes = 768, 32, 2
    mapper_choices = mapper_options(in_dim, out_dim)
    assert [item["hidden_dim"] for item in mapper_choices] == [64, 256, 512, 1024, 2048]
    for item in mapper_choices:
        model = FeatureMapper(in_dim, out_dim, hidden_dim=item["hidden_dim"])
        counted = sum(parameter.numel() for parameter in model.parameters())
        assert counted == item["parameters"]
    choices = classifier_options(in_dim, n_classes)
    assert choices[0]["hidden"] == [64, 32]
    assert any(item["parameters"] > choices[0]["parameters"] for item in choices[1:])
    assert all(item["parameters"] <= CLASSIFIER_CAP for item in choices)
    largest = max(choices, key=lambda item: item["parameters"])
    step = largest["hidden"][0] + 64
    assert head_parameter_count(in_dim, n_classes, (step, step // 2)) > CLASSIFIER_CAP
    for item in choices:
        model = MlpClassifier(in_dim, n_classes, hidden=tuple(item["hidden"]))
        counted = sum(parameter.numel() for parameter in model.parameters())
        assert counted == item["parameters"]


def test_import_reuses_an_existing_label_and_keeps_other_categories(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "hitl_features.session.train_mapper",
        lambda *args, **kwargs: {"loss": 0.0, "history": [0.0], "steps": 1},
    )
    matrix = np.ones((14, 3), dtype=np.float32)
    client, snapshot = _session(tmp_path, matrix, [10, 11, 12, 13])
    session_id = snapshot["id"]
    session = client.application.config["STORE"].get(session_id)
    spiral = session.graph.create_category("spiral")
    left = session.graph.create_category("left")
    session.graph.assign(10, spiral)
    session.graph.assign(13, "recycle")
    csv_text = "obj,main,sub\n10,spiral,barred\n11,spiral,plain\n12,elliptical,\n"
    imported = client.post(
        f"/api/sessions/{session_id}/import-labels",
        json={"csv": csv_text, "index_column": "obj", "label_columns": ["main", "sub"]},
    )
    assert imported.status_code == 200, imported.get_json()
    roots = [cat.name for cat in session.graph.children(None) if cat.id != "recycle"]
    assert roots.count("spiral") == 1
    assert "left" in roots
    assert "elliptical" in roots
    barred = session.graph.category_of(10)
    assert session.graph.categories[barred].name == "barred"
    assert session.graph.categories[barred].parent_id == spiral
    assert session.graph.category_of(11) != spiral
    assert session.graph.categories[session.graph.category_of(11)].parent_id == spiral
    assert session.graph.is_recycled_sample(13)
    assert imported.get_json()["snapshot"]["recycle"]["images"][0]["index"] == 13
    again = client.post(
        f"/api/sessions/{session_id}/import-labels",
        json={"csv": csv_text, "index_column": "obj", "label_columns": ["main", "sub"]},
    )
    assert again.status_code == 200, again.get_json()
    roots_again = [cat.name for cat in session.graph.children(None) if cat.id != "recycle"]
    assert roots_again.count("spiral") == 1
    assert left in {cat.id for cat in session.graph.children(None)}


def test_mapper_size_change_discards_the_old_module(tmp_path, monkeypatch):
    seen = {}

    def spy(mapper, features, sample_ids, graph, **kwargs):
        seen["hidden"] = mapper.hidden_dim
        seen["mapper"] = mapper
        return {"loss": 0.0, "history": [0.0], "steps": 1}

    monkeypatch.setattr("hitl_features.session.train_mapper", spy)
    matrix = np.eye(4, dtype=np.float32)
    client, snapshot = _session(tmp_path, matrix, [0, 1, 2, 3])
    session_id = snapshot["id"]
    session = client.application.config["STORE"].get(session_id)
    assert snapshot["mapper_hidden"] == 256
    assert any(item["hidden_dim"] == 256 for item in snapshot["mapper_sizes"])
    assert any(item["hidden"] == [64, 32] for item in snapshot["head_sizes"])
    assert all(item["parameters"] <= CLASSIFIER_CAP for item in snapshot["head_sizes"])
    left = session.graph.create_category("left")
    session.graph.assign(0, left)
    with torch.no_grad():
        session.mapper.net[0].weight.fill_(3.0)
    session.mapper_trained = True
    old = session.mapper
    same = client.post(f"/api/sessions/{session_id}/mapper-size", json={"hidden_dim": 256})
    assert same.status_code == 200, same.get_json()
    assert session.mapper is old
    assert session.mapper_trained is True
    changed = client.post(f"/api/sessions/{session_id}/mapper-size", json={"hidden_dim": 64})
    assert changed.status_code == 200, changed.get_json()
    body = changed.get_json()["snapshot"]
    assert body["mapper_hidden"] == 64
    assert body["mapper_trained"] is False
    assert session.pool_predictions == []
    assert session.mapper is not old
    assert session.mapper.hidden_dim == 64
    assert tuple(session.mapper.net[0].weight.shape) != tuple(old.net[0].weight.shape)
    refused = client.post(f"/api/sessions/{session_id}/mapper-size", json={"hidden_dim": 7})
    assert refused.status_code == 400
    trained = client.post(f"/api/sessions/{session_id}/confirm", json={"epochs": 1})
    assert trained.status_code == 200, trained.get_json()
    assert seen["hidden"] == 64
    assert seen["mapper"] is session.mapper
    huge = client.post(
        f"/api/sessions/{session_id}/head-size",
        json={"hidden": [100000, 100000]},
    )
    assert huge.status_code == 400
    larger = max(snapshot["head_sizes"], key=lambda item: item["parameters"])
    picked = client.post(
        f"/api/sessions/{session_id}/head-size",
        json={"hidden": larger["hidden"]},
    )
    assert picked.status_code == 200, picked.get_json()
    assert list(session.head_hidden) == larger["hidden"]


def test_training_stream_emits_a_known_fraction(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "hitl_features.session.train_mapper",
        lambda *args, **kwargs: {"loss": 0.0, "history": [0.0], "steps": 1},
    )
    matrix = np.eye(4, dtype=np.float32)
    client, snapshot = _session(tmp_path, matrix, [0, 1])
    session_id = snapshot["id"]
    session = client.application.config["STORE"].get(session_id)
    left = session.graph.create_category("left")
    right = session.graph.create_category("right")
    session.graph.assign(0, left)
    session.graph.assign(1, right)
    streamed = client.post(
        f"/api/sessions/{session_id}/confirm",
        json={"epochs": 1, "stream": True},
    )
    assert streamed.status_code == 200
    text = streamed.get_data(as_text=True)
    assert '"event": "progress"' in text
    assert '"done": 1' in text
    assert '"event": "done"' in text
