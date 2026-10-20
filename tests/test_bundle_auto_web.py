import csv
import json
from pathlib import Path

import h5py
import numpy as np

from hitl_features.auto_test import AutoTestConfig, run_auto_test
from hitl_features.bundle import load_bundle, save_bundle
from hitl_features.graph import CategoryGraph
from hitl_features.mapper import FeatureMapper
from hitl_features.web.app import create_app


def test_bundle_roundtrip(tmp_path: Path):
    graph = CategoryGraph()
    left = graph.create_category("left")
    right = graph.create_category("right")
    graph.assign(3, left, round_index=1)
    graph.set_strength(left, right, 0.5)
    mapper = FeatureMapper(4, 2, hidden_dim=3)
    save_bundle(
        tmp_path,
        graph=graph,
        mapper=mapper,
        feature_path="features.h5",
        distance_row={"round": 1, "mean_within": 0.1, "mean_between": 0.8, "loss": 0.2, "trained": True},
    )
    save_bundle(
        tmp_path,
        graph=graph,
        mapper=mapper,
        feature_path="features.h5",
        distance_row={"round": 2, "mean_within": 0.05, "mean_between": 1.1, "loss": 0.1, "trained": True},
    )
    loaded = load_bundle(tmp_path)
    assert loaded.graph.category_of(3) == left
    assert loaded.graph.relation_strength(left, right) == 0.5
    assert loaded.mapper.out_dim == 2
    lines = (tmp_path / "assignments.jsonl").read_text().strip().splitlines()
    assert json.loads(lines[0])["sample_index"] == 3
    history = (tmp_path / "distance_history.csv").read_text().strip().splitlines()
    assert len(history) == 3


def test_auto_test_refuses_locked_holdout(tmp_path: Path):
    config = AutoTestConfig(
        feature_path="missing.h5",
        labels_csv=str(tmp_path / "expert_testset_locked.csv"),
        label_column="label",
        positive="mergers",
        negatives=["other"],
        output_dir=str(tmp_path / "out"),
    )
    try:
        run_auto_test(config)
    except ValueError as exc:
        assert "locked" in str(exc)
    else:
        raise AssertionError("locked holdout was read")


def test_auto_test_writes_four_way_report(tmp_path: Path):
    feature_path = tmp_path / "feat.h5"
    labels_path = tmp_path / "labels.csv"
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(40, 6)).astype(np.float32)
    matrix[:20, 0] += 2
    with h5py.File(feature_path, "w") as handle:
        handle.create_dataset("features", data=matrix)
    with labels_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "label"])
        writer.writeheader()
        for index in range(40):
            writer.writerow({"index": index, "label": "mergers" if index < 20 else "other"})
    output = tmp_path / "run"
    report = run_auto_test(
        AutoTestConfig(
            feature_path=str(feature_path),
            labels_csv=str(labels_path),
            label_column="label",
            positive="mergers",
            negatives=["other"],
            pool_size=28,
            round_size=4,
            rounds=2,
            out_dim=4,
            seed=0,
            epochs=2,
            eval_fraction=0.3,
            classifier_epochs=5,
            device="cpu",
            output_dir=str(output),
            hidden_dim=8,
        )
    )
    assert set(report["four_way"]) == {
        "A_recommended_raw",
        "B_recommended_mapped",
        "C_random_raw",
        "D_random_mapped",
    }
    assert len(report["rounds"]) == 2
    assert (output / "report.json").is_file()
    assert report["locked_holdout"] == "not read"
    for name in ("roc.png", "pr.png", "contact_sheet.png"):
        assert (output / name).is_file()
        assert (output / name).read_bytes().startswith(b"\x89PNG")


def test_web_round_places_trains_and_reloads(tmp_path: Path):
    feature_path = tmp_path / "feat.h5"
    matrix = np.arange(16 * 6, dtype=np.float32).reshape(16, 6)
    with h5py.File(feature_path, "w") as handle:
        handle.create_dataset("mae_features", data=matrix)
    image = tmp_path / "3.jpg"
    image.write_bytes(b"jpg")
    app = create_app(tmp_path / "sessions")
    client = app.test_client()
    created = client.post(
        "/api/sessions",
        json={
            "feature_path": str(feature_path),
            "img_dir": str(tmp_path),
            "pool_size": 16,
            "seed": 0,
            "out_dim": 4,
            "round_size": 4,
            "epochs": 1,
            "name": "demo",
            "device": "cpu",
        },
    )
    assert created.status_code == 200
    snapshot = created.get_json()["snapshot"]
    session_id = snapshot["id"]
    early = client.post(f"/api/sessions/{session_id}/confirm", json={})
    assert early.status_code == 409
    recommended = client.post(
        f"/api/sessions/{session_id}/recommend",
        json={"mechanism": "random"},
    )
    images = [item["index"] for item in recommended.get_json()["snapshot"]["images"]]
    assert len(images) == 4
    first = client.post(
        f"/api/sessions/{session_id}/place",
        json={"sample_index": images[0], "new_name": "alpha"},
    )
    alpha = first.get_json()["snapshot"]["categories"][0]["id"]
    client.post(
        f"/api/sessions/{session_id}/place",
        json={"sample_index": images[1], "new_name": "beta"},
    )
    beta = [
        cat["id"]
        for cat in client.get(f"/api/sessions/{session_id}").get_json()["snapshot"]["categories"]
        if cat["name"] == "beta"
    ][0]
    client.post(
        f"/api/sessions/{session_id}/place",
        json={"sample_index": images[2], "category_id": alpha},
    )
    client.post(
        f"/api/sessions/{session_id}/place",
        json={"sample_index": images[3], "category_id": beta},
    )
    strengthened = client.post(
        f"/api/sessions/{session_id}/strength",
        json={"a": alpha, "b": beta, "strength": 1.5},
    )
    assert strengthened.status_code == 200
    assert strengthened.get_json()["snapshot"]["strengths"][0]["strength"] == 1.5
    trained = client.post(f"/api/sessions/{session_id}/confirm", json={"epochs": 1})
    assert trained.status_code == 200
    assert trained.get_json()["distance"]["round"] == 1
    media = client.get(f"/media/{session_id}/3")
    assert media.status_code == 200
    reloaded = create_app(tmp_path / "sessions")
    again = reloaded.test_client().get(f"/api/sessions/{session_id}")
    assert again.status_code == 200
    assert again.get_json()["snapshot"]["round_number"] == 1
    assert len(again.get_json()["snapshot"]["all_categories"]) == 2
