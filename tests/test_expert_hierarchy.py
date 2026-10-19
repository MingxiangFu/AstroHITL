import csv
from pathlib import Path

import h5py
import numpy as np

from hitl_features.bundle import load_bundle
from hitl_features.classify import evaluate_multiclass, fit_classifier
from hitl_features.expert_test import ExpertTestConfig, _place, pairwise_score_agreement, run_expert_hierarchy
from hitl_features.graph import CategoryGraph


def test_place_nests_secondary_under_main():
    graph = CategoryGraph()
    leaf = _place(graph, {}, {}, 4, "alpha", "red", 1)
    category = graph.categories[leaf]
    parent = graph.categories[category.parent_id]
    assert category.name == "red"
    assert parent.name == "alpha"
    assert graph.divergence_weight(leaf, parent.id) is None


def test_multiclass_accuracy_on_separated_features():
    features = np.zeros((30, 4), dtype=np.float32)
    labels = np.array([0, 1, 2] * 10)
    features[labels == 0, 0] = 1
    features[labels == 1, 1] = 1
    features[labels == 2, 2] = 1
    model = fit_classifier(features, labels, seed=0, epochs=30, device="cpu")
    metrics = evaluate_multiclass(model.cpu(), features, labels, device="cpu")
    assert metrics["accuracy"] > 0.9
    assert metrics["macro_roc_auc"] > 0.9


def test_pairwise_score_agreement_is_one_when_order_matches():
    reference = np.array([0.1, 0.4, 0.9, 0.2])
    assert pairwise_score_agreement(reference, reference) == 1.0


def test_expert_hierarchy_runs_main_and_subgraph(tmp_path: Path):
    feature_path, labels_path = _write_toy_catalog(tmp_path)
    output = tmp_path / "run"
    report = run_expert_hierarchy(
        ExpertTestConfig(
            feature_path=str(feature_path),
            labels_csv=str(labels_path),
            pool_size=60,
            round_size=10,
            rounds=2,
            out_dim=4,
            seed=0,
            epochs=2,
            eval_fraction=0.25,
            classifier_epochs=4,
            device="cpu",
            output_dir=str(output),
            hidden_dim=8,
        )
    )
    assert set(report["main_classification"]) == {
        "A_recommended_raw",
        "B_recommended_mapped",
        "C_random_raw",
        "D_random_mapped",
    }
    assert report["main_classification"]["B_recommended_mapped"]["accuracy"] is not None
    assert report["subgraphs"]
    assert report["main_score_agreement"]["mapped"]["mean_pairwise_agreement"] is not None
    bundle = load_bundle(output / "bundle")
    labels = _labels_by_index(labels_path)
    for index, category_id in bundle.graph.assignments.items():
        category = bundle.graph.categories[category_id]
        parent = bundle.graph.categories[category.parent_id]
        main, secondary = labels[index]
        assert parent.name == main
        assert category.name == secondary
    assert report["locked_holdout"] == "not read"


def test_expert_hierarchy_refuses_locked_holdout(tmp_path: Path):
    try:
        run_expert_hierarchy(
            ExpertTestConfig(
                feature_path="missing.h5",
                labels_csv=str(tmp_path / "expert_testset_locked.csv"),
                output_dir=str(tmp_path / "out"),
            )
        )
    except ValueError as exc:
        assert "locked" in str(exc)
    else:
        raise AssertionError("locked holdout was read")


def _write_toy_catalog(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    rows = []
    for index in range(80):
        main = "alpha" if index < 40 else "beta"
        if main == "alpha":
            secondary = "red" if index % 2 == 0 else "blue"
        else:
            secondary = "red" if index % 2 == 0 else "green"
        rows.append((index, main, secondary))
    features = rng.normal(scale=0.05, size=(80, 8)).astype(np.float32)
    for index, main, secondary in rows:
        if main == "alpha":
            features[index, 0] += 1
        else:
            features[index, 1] += 1
        if secondary == "red":
            features[index, 2] += 1
        elif secondary == "blue":
            features[index, 3] += 1
        else:
            features[index, 4] += 1
    feature_path = tmp_path / "feat.h5"
    with h5py.File(feature_path, "w") as handle:
        handle.create_dataset("features", data=features)
    labels_path = tmp_path / "labels.csv"
    score_fields = [
        "mapped_alpha_combined_score",
        "mapped_beta_combined_score",
        "mapped_red_combined_score",
        "mapped_blue_combined_score",
        "mapped_green_combined_score",
    ]
    with labels_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "main_label_expert", "secondary_label_expert", *score_fields],
        )
        writer.writeheader()
        for index, main, secondary in rows:
            row = {
                "index": index,
                "main_label_expert": main,
                "secondary_label_expert": secondary,
                "mapped_alpha_combined_score": 1.0 if main == "alpha" else 0.0,
                "mapped_beta_combined_score": 1.0 if main == "beta" else 0.0,
                "mapped_red_combined_score": 1.0 if secondary == "red" else 0.0,
                "mapped_blue_combined_score": 1.0 if secondary == "blue" else 0.0,
                "mapped_green_combined_score": 1.0 if secondary == "green" else 0.0,
            }
            writer.writerow(row)
    return feature_path, labels_path


def _labels_by_index(path: Path) -> dict[int, tuple[str, str]]:
    with path.open(newline="") as handle:
        return {
            int(row["index"]): (row["main_label_expert"], row["secondary_label_expert"])
            for row in csv.DictReader(handle)
        }
