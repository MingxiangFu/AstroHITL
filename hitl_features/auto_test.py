"""Scripted graph, feature map, and four-way comparison."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hitl_features.bundle import save_bundle
from hitl_features.distances import mean_within_between
from hitl_features.features import load_feature_rows
from hitl_features.figures import high_low_examples, save_comparison_figures
from hitl_features.four_way import run_four_way
from hitl_features.graph import CategoryGraph
from hitl_features.mapper import FeatureMapper
from hitl_features.rank import fit_ranker, pairwise_preference_accuracy
from hitl_features.recommend import recommend
from hitl_features.train import train_mapper


@dataclass
class AutoTestConfig:
    feature_path: str
    labels_csv: str
    label_column: str
    positive: str
    negatives: list[str]
    index_column: str = "index"
    pool_size: int = 5000
    round_size: int = 12
    rounds: int = 8
    out_dim: int = 32
    seed: int = 0
    epochs: int = 30
    eval_fraction: float = 0.3
    classifier_epochs: int = 40
    device: str | None = None
    output_dir: str = "runs/auto_test"
    hidden_dim: int = 256
    img_dir: str = ""


def refuse_locked_holdout(path: str) -> None:
    if "expert_testset_locked" in Path(path).name:
        raise ValueError("refusing locked holdout expert_testset_locked")


def run_auto_test(config: AutoTestConfig) -> dict:
    refuse_locked_holdout(config.labels_csv)
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    labeled = _read_labels(config)
    eval_ids, pool_ids = _split_pool(labeled, config)
    label_of = {index: label for index, label in labeled}
    vectors = _load_vectors(config.feature_path, pool_ids + eval_ids)
    in_dim = next(iter(vectors.values())).shape[0]

    graph = CategoryGraph()
    positive_id = graph.create_category(config.positive)
    other_id = graph.create_category("other")
    mapper = FeatureMapper(in_dim, config.out_dim, hidden_dim=config.hidden_dim)
    rng = np.random.default_rng(config.seed + 1)
    unlabeled = set(pool_ids)
    assigned: list[int] = []
    history: list[dict] = []
    mapper_trained = False

    for round_number in range(1, int(config.rounds) + 1):
        if not unlabeled:
            break
        candidate_ids = np.array(sorted(unlabeled), dtype=np.int64)
        candidate_vectors = np.stack([vectors[int(i)] for i in candidate_ids])
        graph_ids = np.array(assigned, dtype=np.int64)
        graph_vectors = (
            np.stack([_embed_one(mapper, vectors[i], mapper_trained, device) for i in assigned])
            if assigned
            else np.zeros((0, in_dim if not mapper_trained else config.out_dim), dtype=np.float32)
        )
        use_vectors = candidate_vectors
        if mapper_trained:
            use_vectors = _embed_many(mapper, candidate_vectors, device)
            graph_vectors = _embed_many(
                mapper, np.stack([vectors[i] for i in assigned]), device
            )
        chosen = recommend(
            candidate_ids,
            use_vectors,
            graph_ids,
            graph_vectors,
            [graph.category_of(i) for i in assigned],
            "algorithmic",
            n=config.round_size,
            rng=rng,
            n_categories=graph.class_count(),
            round_index=round_number,
        )
        for sample_id in chosen:
            category_id = positive_id if label_of[sample_id] == config.positive else other_id
            graph.assign(sample_id, category_id, round_index=round_number)
            assigned.append(sample_id)
            unlabeled.discard(sample_id)
        train_ids = list(assigned)
        train_x = np.stack([vectors[i] for i in train_ids])
        trained = train_mapper(
            mapper, train_x, train_ids, graph, epochs=config.epochs, device=device
        )
        if trained["steps"] > 0:
            mapper_trained = True
        mapped = _embed_many(mapper, train_x, device)
        within, between = mean_within_between(
            mapped, [graph.category_of(i) for i in train_ids]
        )
        history.append(
            {
                "round": round_number,
                "mean_within": within,
                "mean_between": between,
                "loss": trained["loss"],
                "trained": trained["steps"] > 0,
            }
        )

    if len(assigned) == 0:
        raise RuntimeError("the scripted rounds assigned no images")
    if len(unlabeled) < len(assigned):
        raise ValueError(
            "not enough unlabeled pool images for a same-size random draw; "
            "raise pool_size or lower rounds"
        )
    random_ids = [
        int(value)
        for value in rng.choice(sorted(unlabeled), size=len(assigned), replace=False)
    ]
    recommended_y = np.array(
        [1 if label_of[i] == config.positive else 0 for i in assigned], dtype=np.int64
    )
    random_y = np.array(
        [1 if label_of[i] == config.positive else 0 for i in random_ids], dtype=np.int64
    )
    eval_y = np.array(
        [1 if label_of[i] == config.positive else 0 for i in eval_ids], dtype=np.int64
    )
    recommended_raw = np.stack([vectors[i] for i in assigned])
    random_raw = np.stack([vectors[i] for i in random_ids])
    eval_raw = np.stack([vectors[i] for i in eval_ids])
    recommended_mapped = _embed_many(mapper, recommended_raw, device)
    random_mapped = _embed_many(mapper, random_raw, device)
    eval_mapped = _embed_many(mapper, eval_raw, device)
    four, arm_scores = run_four_way(
        recommended_raw,
        recommended_mapped,
        recommended_y,
        random_raw,
        random_mapped,
        random_y,
        eval_raw,
        eval_mapped,
        eval_y,
        seed=config.seed,
        epochs=config.classifier_epochs,
        device=device,
        with_scores=True,
    )
    ranking = _ranking_check(
        recommended_raw,
        recommended_mapped,
        recommended_y,
        eval_raw,
        eval_mapped,
        eval_y,
        seed=config.seed,
        epochs=config.classifier_epochs,
        device=device,
    )
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for row in history:
        save_bundle(
            output / "bundle",
            graph=graph,
            mapper=mapper,
            feature_path=config.feature_path,
            distance_row=row,
        )
    report = {
        "positive": config.positive,
        "negatives": list(config.negatives),
        "seed": config.seed,
        "device": device,
        "n_recommended": len(assigned),
        "n_eval": len(eval_ids),
        "recommended_positives": int(recommended_y.sum()),
        "rounds": history,
        "four_way": four,
        "ranking": ranking,
        "figures": _save_auto_figures(
            output, arm_scores, eval_ids, eval_y, img_dir=config.img_dir
        ),
        "locked_holdout": "not read",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def _save_auto_figures(
    output: Path,
    arm_scores: dict,
    eval_ids: list[int],
    eval_y: np.ndarray,
    img_dir: str = "",
) -> dict:
    arms = []
    for name in (
        "A_recommended_raw",
        "B_recommended_mapped",
        "C_random_raw",
        "D_random_mapped",
    ):
        scores = arm_scores.get(name)
        if scores is None:
            continue
        arms.append({"name": name, "y": eval_y, "scores": scores})
    source = arm_scores.get("B_recommended_mapped")
    if source is None:
        for name in (
            "A_recommended_raw",
            "B_recommended_mapped",
            "C_random_raw",
            "D_random_mapped",
        ):
            if arm_scores.get(name) is not None:
                source = arm_scores[name]
                break
    if source is None:
        high, low = [], []
    else:
        high, low = high_low_examples(eval_ids, source, k=4)
    return save_comparison_figures(output, arms, high, low, img_dir=img_dir or None)


def _ranking_check(
    recommended_raw,
    recommended_mapped,
    recommended_y,
    eval_raw,
    eval_mapped,
    eval_y,
    *,
    seed: int,
    epochs: int,
    device: str,
) -> dict:
    if len(np.unique(recommended_y)) < 2 or len(np.unique(eval_y)) < 2:
        return {"note": "ranking needs both classes in the recommended set and the eval set"}
    groups_train = ["positive" if int(y) == 1 else "other" for y in recommended_y]
    groups_eval = ["positive" if int(y) == 1 else "other" for y in eval_y]
    raw_model = fit_ranker(
        recommended_raw, groups_train, ["positive", "other"], 1.0,
        seed=seed, epochs=epochs, device=device,
    )
    mapped_model = fit_ranker(
        recommended_mapped, groups_train, ["positive", "other"], 1.0,
        seed=seed, epochs=epochs, device=device,
    )
    return {
        "raw_pairwise_accuracy": pairwise_preference_accuracy(
            raw_model.score(eval_raw), groups_eval, ["positive", "other"]
        ),
        "mapped_pairwise_accuracy": pairwise_preference_accuracy(
            mapped_model.score(eval_mapped), groups_eval, ["positive", "other"]
        ),
    }


def _embed_one(mapper, vector, trained: bool, device: str) -> np.ndarray:
    if not trained:
        return np.asarray(vector, dtype=np.float32)
    return _embed_many(mapper, np.asarray(vector, dtype=np.float32).reshape(1, -1), device)[0]


def _embed_many(mapper: FeatureMapper, features: np.ndarray, device: str) -> np.ndarray:
    mapper.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
        return mapper(inputs).cpu().numpy()


def _read_labels(config: AutoTestConfig) -> list[tuple[int, str]]:
    allowed = {config.positive, *config.negatives}
    rows: list[tuple[int, str]] = []
    with open(config.labels_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row[config.label_column]
            if label in allowed:
                rows.append((int(row[config.index_column]), label))
    if not rows:
        raise ValueError("no labeled rows matched the positive and negative classes")
    return rows


def _split_pool(
    labeled: list[tuple[int, str]], config: AutoTestConfig
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(config.seed)
    by_label: dict[str, list[int]] = {}
    for index, label in labeled:
        by_label.setdefault(label, []).append(index)
    eval_ids: list[int] = []
    pool_ids: list[int] = []
    for ids in by_label.values():
        order = np.array(ids, dtype=np.int64)
        rng.shuffle(order)
        n_eval = int(round(len(order) * float(config.eval_fraction)))
        if len(order) > 1:
            n_eval = min(max(n_eval, 1), len(order) - 1)
        else:
            n_eval = 0
        eval_ids.extend(int(value) for value in order[:n_eval])
        pool_ids.extend(int(value) for value in order[n_eval:])
    if len(pool_ids) > int(config.pool_size):
        picked = rng.choice(np.array(pool_ids), size=int(config.pool_size), replace=False)
        pool_ids = [int(value) for value in picked]
    if not pool_ids or not eval_ids:
        raise ValueError("split produced an empty pool or eval set")
    return eval_ids, pool_ids


def _load_vectors(feature_path: str, indices: list[int]) -> dict[int, np.ndarray]:
    unique = sorted(set(int(i) for i in indices))
    loaded = load_feature_rows(feature_path, unique)
    return {index: loaded[row] for row, index in enumerate(unique)}
