"""Hierarchy test from expert main and secondary labels.

The top graph uses main_label_expert. Inside each main category, a subgraph
uses secondary_label_expert. Catalog probabilities in
mapped_{class}_combined_score are a separate check on the trained classifiers.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from hitl_features.auto_test import refuse_locked_holdout
from hitl_features.bundle import save_bundle
from hitl_features.classify import evaluate_multiclass, fit_classifier, predict_proba
from hitl_features.distances import mean_within_between
from hitl_features.features import filter_catalog_indices, load_feature_rows
from hitl_features.graph import CategoryGraph
from hitl_features.mapper import FeatureMapper
from hitl_features.recommend import recommend
from hitl_features.train import train_mapper


@dataclass
class ExpertSample:
    index: int
    main: str
    secondary: str
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class ExpertTestConfig:
    feature_path: str
    labels_csv: str
    index_column: str = "index"
    main_column: str = "main_label_expert"
    secondary_column: str = "secondary_label_expert"
    pool_size: int = 400
    round_size: int = 16
    rounds: int = 4
    out_dim: int = 32
    seed: int = 0
    epochs: int = 8
    eval_fraction: float = 0.3
    classifier_epochs: int = 20
    device: str | None = None
    output_dir: str = "runs/expert_hierarchy"
    hidden_dim: int = 128


def run_expert_hierarchy(config: ExpertTestConfig) -> dict:
    refuse_locked_holdout(config.labels_csv)
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    samples = _read_samples(config)
    present = set(filter_catalog_indices(config.feature_path, [sample.index for sample in samples]))
    kept = [sample for sample in samples if sample.index in present]
    dropped = len(samples) - len(kept)
    eval_samples, pool_samples = _split_pool(kept, config)
    vectors = _load_vectors(
        config.feature_path,
        [sample.index for sample in pool_samples + eval_samples],
    )
    in_dim = next(iter(vectors.values())).shape[0]
    by_index = {sample.index: sample for sample in kept}

    graph = CategoryGraph()
    main_nodes: dict[str, str] = {}
    leaf_nodes: dict[tuple[str, str], str] = {}
    mapper = FeatureMapper(in_dim, config.out_dim, hidden_dim=config.hidden_dim)
    rng = np.random.default_rng(config.seed + 1)
    unlabeled = {sample.index for sample in pool_samples}
    assigned: list[int] = []
    history: list[dict] = []
    mapper_trained = False
    print(f"expert hierarchy: {len(pool_samples)} pool, {len(eval_samples)} eval, device {device}")

    for round_number in range(1, int(config.rounds) + 1):
        if not unlabeled:
            break
        candidate_ids = np.array(sorted(unlabeled), dtype=np.int64)
        candidate_raw = np.stack([vectors[int(i)] for i in candidate_ids])
        use_vectors, graph_vectors = _round_vectors(
            mapper, vectors, candidate_raw, assigned, mapper_trained, config.out_dim, in_dim, device
        )
        chosen = recommend(
            candidate_ids,
            use_vectors,
            np.array(assigned, dtype=np.int64),
            graph_vectors,
            [graph.category_of(i) for i in assigned],
            "algorithmic",
            n=config.round_size,
            rng=rng,
            n_categories=graph.class_count(),
            round_index=round_number,
        )
        for sample_id in chosen:
            sample = by_index[sample_id]
            _place(graph, main_nodes, leaf_nodes, sample_id, sample.main, sample.secondary, round_number)
            assigned.append(sample_id)
            unlabeled.discard(sample_id)
        train_x = np.stack([vectors[i] for i in assigned])
        trained = train_mapper(
            mapper, train_x, assigned, graph, epochs=config.epochs, device=device
        )
        if trained["steps"] > 0:
            mapper_trained = True
        mapped = _embed_many(mapper, train_x, device)
        leaf_names = [f"{by_index[i].main}/{by_index[i].secondary}" for i in assigned]
        main_names = [by_index[i].main for i in assigned]
        leaf_within, leaf_between = mean_within_between(mapped, leaf_names)
        main_within, main_between = mean_within_between(mapped, main_names)
        history.append(
            {
                "round": round_number,
                "n_assigned": len(assigned),
                "mean_within": leaf_within,
                "mean_between": leaf_between,
                "main_mean_within": main_within,
                "main_mean_between": main_between,
                "loss": trained["loss"],
                "trained": trained["steps"] > 0,
            }
        )
        print(
            f"round {round_number}: assigned {len(assigned)}, "
            f"leaf within {leaf_within}, between {leaf_between}, loss {trained['loss']}"
        )

    if not assigned:
        raise RuntimeError("the scripted rounds assigned no images")
    if len(unlabeled) < len(assigned):
        raise ValueError("not enough unlabeled pool images for a same-size random draw")
    random_ids = [int(v) for v in rng.choice(sorted(unlabeled), size=len(assigned), replace=False)]
    recommended = [by_index[i] for i in assigned]
    random_samples = [by_index[i] for i in random_ids]

    main_classification = _four_way(
        recommended, random_samples, eval_samples, vectors, mapper, device, config,
        label_of=lambda sample: sample.main,
    )
    main_scores = _score_agreement(
        recommended, eval_samples, vectors, mapper, device, config,
        label_of=lambda sample: sample.main,
    )
    subgraphs = _subgraph_reports(
        recommended, unlabeled, by_index, eval_samples, vectors, mapper, device, config, rng
    )
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if history:
        save_bundle(
            output / "bundle",
            graph=graph,
            mapper=mapper,
            feature_path=config.feature_path,
            distance_row={
                "round": history[-1]["round"],
                "mean_within": history[-1]["mean_within"],
                "mean_between": history[-1]["mean_between"],
                "loss": history[-1]["loss"],
                "trained": history[-1]["trained"],
            },
        )
    report = _clean(
        {
            "labels_csv": config.labels_csv,
            "main_column": config.main_column,
            "secondary_column": config.secondary_column,
            "dropped_out_of_range": dropped,
            "n_pool": len(pool_samples),
            "n_eval": len(eval_samples),
            "n_recommended": len(assigned),
            "recommended_main_counts": dict(Counter(sample.main for sample in recommended)),
            "rounds": history,
            "main_classification": main_classification,
            "main_score_agreement": main_scores,
            "subgraphs": subgraphs,
            "locked_holdout": "not read",
        }
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    _write_history(output / "distance_history.csv", history)
    return report


def _place(graph, main_nodes, leaf_nodes, index, main, secondary, round_index) -> str:
    if main not in main_nodes:
        main_nodes[main] = graph.create_category(main)
    key = (main, secondary)
    if key not in leaf_nodes:
        leaf_nodes[key] = graph.create_category(secondary, parent_id=main_nodes[main])
    graph.assign(index, leaf_nodes[key], round_index=round_index)
    return leaf_nodes[key]


def _four_way(recommended, random_samples, eval_samples, vectors, mapper, device, config, label_of) -> dict:
    arms = {}
    for name, train_samples, mapped in (
        ("A_recommended_raw", recommended, False),
        ("B_recommended_mapped", recommended, True),
        ("C_random_raw", random_samples, False),
        ("D_random_mapped", random_samples, True),
    ):
        arms[name] = _classify_arm(
            train_samples, eval_samples, vectors, mapper, device, config, label_of, mapped
        )["metrics"]
    return arms


def _classify_arm(train_samples, eval_samples, vectors, mapper, device, config, label_of, mapped) -> dict:
    train_names = [label_of(sample) for sample in train_samples]
    classes = sorted(set(train_names))
    if len(classes) < 2:
        return {
            "metrics": {
                "accuracy": None,
                "macro_roc_auc": None,
                "note": "training split has one class",
                "classes": classes,
            },
            "proba": None,
            "classes": classes,
            "eval_samples": [],
        }
    mapping = {name: index for index, name in enumerate(classes)}
    y = np.array([mapping[name] for name in train_names], dtype=np.int64)
    train_x = _matrix(train_samples, vectors, mapper, device, mapped)
    model = fit_classifier(
        train_x, y, seed=config.seed, epochs=config.classifier_epochs, device=device
    )
    eval_device = device if str(device).startswith("cuda") else "cpu"
    if eval_device == "cpu":
        model = model.cpu()
    kept = [sample for sample in eval_samples if label_of(sample) in mapping]
    if len(kept) < 2 or len({label_of(sample) for sample in kept}) < 2:
        return {
            "metrics": {
                "accuracy": None,
                "macro_roc_auc": None,
                "note": "eval split does not contain two training classes",
                "classes": classes,
            },
            "proba": None,
            "classes": classes,
            "eval_samples": [],
        }
    eval_x = _matrix(kept, vectors, mapper, device, mapped)
    eval_y = np.array([mapping[label_of(sample)] for sample in kept], dtype=np.int64)
    metrics = evaluate_multiclass(model, eval_x, eval_y, device=eval_device)
    metrics["classes"] = classes
    metrics["n_eval_used"] = len(kept)
    proba = predict_proba(model, eval_x, device=eval_device)
    return {"metrics": metrics, "proba": proba, "classes": classes, "eval_samples": kept}


def _score_agreement(train_samples, eval_samples, vectors, mapper, device, config, label_of) -> dict:
    raw = _classify_arm(
        train_samples, eval_samples, vectors, mapper, device, config, label_of, mapped=False
    )
    mapped = _classify_arm(
        train_samples, eval_samples, vectors, mapper, device, config, label_of, mapped=True
    )
    return {
        "raw": _agreement_table(raw),
        "mapped": _agreement_table(mapped),
    }


def _agreement_table(arm: dict) -> dict:
    if arm["proba"] is None:
        return {"mean_pairwise_agreement": None, "by_class": {}, "note": arm["metrics"].get("note")}
    by_class = {}
    for class_name in arm["classes"]:
        column = arm["classes"].index(class_name)
        reference = np.array(
            [sample.scores.get(class_name, np.nan) for sample in arm["eval_samples"]],
            dtype=np.float64,
        )
        by_class[class_name] = pairwise_score_agreement(arm["proba"][:, column], reference)
    values = [value for value in by_class.values() if value is not None]
    return {
        "mean_pairwise_agreement": float(np.mean(values)) if values else None,
        "by_class": by_class,
    }


def _subgraph_reports(
    recommended, unlabeled, by_index, eval_samples, vectors, mapper, device, config, rng
) -> dict:
    grouped: dict[str, list[ExpertSample]] = defaultdict(list)
    for sample in recommended:
        grouped[sample.main].append(sample)
    reports = {}
    for main, group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        if len({sample.secondary for sample in group}) < 2:
            continue
        remaining = [
            by_index[index]
            for index in sorted(unlabeled)
            if by_index[index].main == main
        ]
        if len(remaining) < len(group):
            reports[main] = {"note": "not enough leftover samples for a same-size random draw"}
            continue
        picked = rng.choice(len(remaining), size=len(group), replace=False)
        random_samples = [remaining[int(i)] for i in np.asarray(picked).reshape(-1)]
        for sample in random_samples:
            unlabeled.discard(sample.index)
        branch_eval = [sample for sample in eval_samples if sample.main == main]
        reports[main] = {
            "n_recommended": len(group),
            "secondary_counts": dict(Counter(sample.secondary for sample in group)),
            "classification": _four_way(
                group, random_samples, branch_eval, vectors, mapper, device, config,
                label_of=lambda sample: sample.secondary,
            ),
            "score_agreement": _score_agreement(
                group, branch_eval, vectors, mapper, device, config,
                label_of=lambda sample: sample.secondary,
            ),
        }
    return reports


def pairwise_score_agreement(predicted: np.ndarray, reference: np.ndarray) -> float | None:
    """Fraction of pairs where the higher catalog score also has the higher prediction."""
    pred = np.asarray(predicted, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(ref)
    pred, ref = pred[mask], ref[mask]
    if len(pred) < 3:
        return None
    diff_ref = ref[:, None] - ref[None, :]
    diff_pred = pred[:, None] - pred[None, :]
    upper = np.triu(np.ones((len(pred), len(pred)), dtype=bool), k=1)
    usable = upper & (diff_ref != 0)
    if not np.any(usable):
        return None
    return float(np.mean(((diff_ref * diff_pred) > 0)[usable]))


def _round_vectors(mapper, vectors, candidate_raw, assigned, trained, out_dim, in_dim, device):
    if not trained:
        width = in_dim
        graph_vectors = (
            np.stack([vectors[i] for i in assigned])
            if assigned
            else np.zeros((0, width), dtype=np.float32)
        )
        return candidate_raw, graph_vectors
    use_vectors = _embed_many(mapper, candidate_raw, device)
    graph_vectors = (
        _embed_many(mapper, np.stack([vectors[i] for i in assigned]), device)
        if assigned
        else np.zeros((0, out_dim), dtype=np.float32)
    )
    return use_vectors, graph_vectors


def _matrix(samples, vectors, mapper, device, mapped: bool) -> np.ndarray:
    raw = np.stack([vectors[sample.index] for sample in samples])
    if not mapped:
        return raw
    return _embed_many(mapper, raw, device)


def _embed_many(mapper: FeatureMapper, features: np.ndarray, device: str) -> np.ndarray:
    mapper.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
        return mapper(inputs).cpu().numpy()


def _read_samples(config: ExpertTestConfig) -> list[ExpertSample]:
    samples: list[ExpertSample] = []
    with open(config.labels_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        score_columns = {
            name[len("mapped_") : -len("_combined_score")]: name
            for name in fields
            if name.startswith("mapped_") and name.endswith("_combined_score")
        }
        for row in reader:
            main = (row.get(config.main_column) or "").strip()
            secondary = (row.get(config.secondary_column) or "").strip()
            if not main or not secondary or main in _MISSING or secondary in _MISSING:
                continue
            scores = {}
            for label, column in score_columns.items():
                raw = (row.get(column) or "").strip()
                if raw in _MISSING:
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if np.isfinite(value):
                    scores[label] = value
            samples.append(
                ExpertSample(int(row[config.index_column]), main, secondary, scores)
            )
    if not samples:
        raise ValueError("no expert rows had both a main and a secondary label")
    return samples


_MISSING = {"", "NA", "NA_Value", "nan", "None"}


def _split_pool(
    samples: list[ExpertSample], config: ExpertTestConfig
) -> tuple[list[ExpertSample], list[ExpertSample]]:
    rng = np.random.default_rng(config.seed)
    by_main: dict[str, list[ExpertSample]] = defaultdict(list)
    for sample in samples:
        by_main[sample.main].append(sample)
    eval_samples: list[ExpertSample] = []
    pool_samples: list[ExpertSample] = []
    for group in by_main.values():
        order = np.array(group, dtype=object)
        rng.shuffle(order)
        n_eval = int(round(len(order) * float(config.eval_fraction)))
        if len(order) > 1:
            n_eval = min(max(n_eval, 1), len(order) - 1)
        else:
            n_eval = 0
        eval_samples.extend(order[:n_eval].tolist())
        pool_samples.extend(order[n_eval:].tolist())
    if len(pool_samples) > int(config.pool_size):
        picked = rng.choice(len(pool_samples), size=int(config.pool_size), replace=False)
        pool_samples = [pool_samples[int(i)] for i in np.asarray(picked).reshape(-1)]
    if not pool_samples or not eval_samples:
        raise ValueError("split produced an empty pool or eval set")
    return eval_samples, pool_samples


def _load_vectors(feature_path: str, indices: list[int]) -> dict[int, np.ndarray]:
    unique = sorted(set(int(i) for i in indices))
    loaded = load_feature_rows(feature_path, unique)
    return {index: loaded[row] for row, index in enumerate(unique)}


def _write_history(path: Path, history: list[dict]) -> None:
    fields = [
        "round",
        "n_assigned",
        "mean_within",
        "mean_between",
        "main_mean_within",
        "main_mean_between",
        "loss",
        "trained",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
