"""One interactive labeling session and its on-disk bundle."""

from __future__ import annotations

import copy
import json
import pickle
import uuid
from pathlib import Path

import numpy as np
import torch

from hitl_features.bundle import load_bundle, save_bundle
from hitl_features.distances import mean_cross_distance, mean_within_between, mean_within_distance
from hitl_features.features import catalog_indices, filter_catalog_indices, load_feature_rows
from hitl_features.graph import RECYCLE_ID, CategoryGraph
from hitl_features.labels import (
    export_hierarchy_csv,
    merge_hierarchy,
    merge_labels,
    parse_hierarchy_csv,
    parse_label_csv,
)
from hitl_features.mapper import FeatureMapper
from hitl_features.modelsizes import (
    CLASSIFIER_CAP,
    classifier_options,
    head_parameter_count,
    mapper_options,
)
from hitl_features.classify import evaluate_binary, evaluate_multiclass, fit_classifier, predict_proba
from hitl_features.pool_predict import predict_pool
from hitl_features.preview import category_preview
from hitl_features.rank import fit_ranker, pairwise_preference_accuracy
from hitl_features.recommend import recommend
from hitl_features.score_board import equal_width_bins, fixed_score_indices
from hitl_features.tasks import inference_ids, normalize_fractions, stratified_split, task_labels
from hitl_features.train import train_mapper

MECHANISMS = ["random", "diversity", "unknown", "certain", "uncertain"]
_CATEGORY_COLORS = (
    "#2f5d73",
    "#b86a1a",
    "#5c4d86",
    "#a34b2e",
    "#8e2f3d",
    "#2f6f4e",
    "#1f6f78",
    "#8a5a12",
    "#3d4c8a",
    "#6b3f69",
    "#3f6b3a",
    "#8a3f3f",
)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    return value


def _annotate_board(board: dict | None, assigned: set[int]) -> dict | None:
    if not board:
        return None
    cloned = copy.deepcopy(board)
    for bin_row in cloned.get("bins") or []:
        for image in bin_row.get("images") or []:
            image["in_graph"] = int(image["index"]) in assigned
    return cloned


class LabSession:
    def __init__(
        self,
        session_id: str,
        directory: Path,
        feature_path: str,
        pool_indices: list[int],
        features: np.ndarray,
        out_dim: int,
        round_size: int,
        seed: int,
        img_dir: str,
        device: str,
        epochs: int,
    ) -> None:
        self.session_id = session_id
        self.directory = Path(directory)
        self.feature_path = feature_path
        self.pool_indices = [int(value) for value in pool_indices]
        self.features = np.asarray(features, dtype=np.float32)
        self.index_to_row = {index: row for row, index in enumerate(self.pool_indices)}
        self.out_dim = int(out_dim)
        self.round_size = int(round_size)
        self.seed = int(seed)
        self.img_dir = img_dir
        self.device = device
        self.epochs = int(epochs)
        self.graph = CategoryGraph()
        self.mapper = FeatureMapper(self.features.shape[1], self.out_dim)
        self.head_hidden = (64, 32)
        self.feature_source = "mapped"
        self.rng = np.random.default_rng(self.seed + 1)
        self.view_parent: str | None = None
        self.round_ids: list[int] = []
        self.unplaced: set[int] = set()
        self.round_number = 0
        self.mapper_trained = False
        self.distance_history: list[dict] = []
        self.task_draft = {"classes": [], "order": [], "strength": 1.0}
        self.tasks: list[dict] = []
        self.active_task_id: str | None = None
        self._task_seq = 0
        self.score_board = None
        self.task_summary = None
        self.root_waiting: set[int] = set()
        self.pool_predictions: list[dict] = []
        self.recommend_set: list[str] = []
        self.frozen = False
        self.edge_mode = "distance"
        self.shot_size = "small"
        self._mapped_cache = None
        self._layout_token = 0
        self._umap_cache = None

    @property
    def in_dim(self) -> int:
        return int(self.features.shape[1])

    def recommend_round(
        self, mechanism: str, category_ids: list[str] | None = None
    ) -> list[int]:
        if mechanism not in MECHANISMS and mechanism != "algorithmic":
            raise ValueError(f"unknown mechanism: {mechanism}")
        if category_ids is not None:
            self.recommend_set = [
                str(value)
                for value in category_ids
                if not self.graph.is_recycled_category(str(value))
            ]
        level_ids = [
            cat.id
            for cat in self.graph.children(self.view_parent)
            if not self.graph.is_recycled_category(cat.id)
        ]
        allowed = {cat_id for cat_id in self.recommend_set if cat_id in level_ids}
        classes = [cat_id for cat_id in level_ids if not allowed or cat_id in allowed]
        candidates, ref_ids, ref_labels = self._recommend_scope(classes)
        mapped = self.mapper_trained
        matrix = self._pool_matrix(mapped)
        width = matrix.shape[1] if matrix.ndim == 2 else self.in_dim
        candidate_vectors = self._matrix_rows(matrix, candidates, width)
        ref_vectors = self._matrix_rows(matrix, ref_ids, width)
        chosen = recommend(
            np.array(candidates, dtype=np.int64),
            candidate_vectors,
            np.array(ref_ids, dtype=np.int64),
            ref_vectors,
            ref_labels,
            mechanism,
            n=self.round_size,
            rng=self.rng,
            n_categories=len(classes),
            round_index=self.round_number + 1,
        )
        self.round_ids = chosen
        self.unplaced = set(chosen)
        if self.mapper_trained:
            self._refresh_pool_predictions()
        self.save()
        return chosen

    def _recommend_scope(self, class_ids: list[str]) -> tuple[list[int], list[int], list[str]]:
        """Candidates and labeled neighbors for the current level.

        At the root, candidates are unlabeled pool rows and neighbors are
        samples whose root category is in ``class_ids``. Inside a category,
        candidates are samples assigned to that category with no subclass,
        and neighbors are samples under that same category that already have
        a child in ``class_ids``. Samples outside the category are omitted.
        """
        allowed = set(class_ids)
        parent = self.view_parent
        if parent is None:
            candidates = [
                index for index in self.pool_indices if self.graph.category_of(index) is None
            ]
            refs: list[int] = []
            labels: list[str] = []
            for index, owner in self.graph.assignments.items():
                if index not in self.index_to_row or self.graph.is_recycled_sample(index):
                    continue
                root_id = self.graph.path(owner)[0]
                if root_id in allowed:
                    refs.append(index)
                    labels.append(root_id)
            return candidates, refs, labels
        candidates = [
            index for index in self.graph.samples(parent) if index in self.index_to_row
        ]
        refs = []
        labels = []
        for index, owner in self.graph.assignments.items():
            if index not in self.index_to_row or self.graph.is_recycled_sample(index):
                continue
            child_id = _child_under(self.graph, parent, owner)
            if child_id in allowed:
                refs.append(index)
                labels.append(child_id)
        return candidates, refs, labels

    def _matrix_rows(self, matrix: np.ndarray, indices: list[int], width: int) -> np.ndarray:
        if not indices:
            return np.zeros((0, width), dtype=np.float64)
        rows = [self.index_to_row[index] for index in indices]
        return np.asarray(matrix[rows], dtype=np.float64)

    def place(
        self,
        sample_index: int,
        *,
        category_id: str | None = None,
        new_name: str | None = None,
    ) -> str:
        sample_index = int(sample_index)
        if sample_index not in self.index_to_row:
            raise ValueError("sample is outside the annotation pool")
        if new_name:
            category_id = self.graph.create_category(new_name, parent_id=self.view_parent)
        else:
            if category_id is None:
                raise ValueError("a category or a new name is required")
            if self.graph.is_recycled_category(category_id):
                raise ValueError("the recycle bin is not a class")
            category = self.graph.categories[category_id]
            if category.parent_id != self.view_parent:
                raise ValueError("drop the image on a category at the current level")
        self.graph.assign(sample_index, category_id, round_index=self.round_number + 1)
        self.unplaced.discard(sample_index)
        self.root_waiting.discard(sample_index)
        self.save()
        return category_id

    def unassign(self, sample_index: int) -> None:
        """Drop a category assignment and return the image to the root waiting list."""
        sample_index = int(sample_index)
        if sample_index not in self.index_to_row:
            raise ValueError("sample is outside the annotation pool")
        self.graph.unassign(sample_index)
        self.unplaced.discard(sample_index)
        self.root_waiting.add(sample_index)
        self.view_parent = None
        self.save()

    def set_position(self, category_id: str, x: float, y: float) -> None:
        self.graph.set_position(category_id, x, y)
        self.save()

    def enter(self, category_id: str | None) -> None:
        if category_id is not None and category_id not in self.graph.categories:
            raise KeyError(f"unknown category: {category_id}")
        if category_id is not None and self.graph.is_recycled_category(category_id):
            raise ValueError("the recycle bin is not a class")
        self.view_parent = category_id
        self.save()

    def rename(self, category_id: str, name: str) -> None:
        self.graph.rename(category_id, name)
        self.save()

    def set_strength(self, category_a: str, category_b: str, strength: float) -> None:
        self.graph.set_strength(category_a, category_b, strength)
        self.save()

    def confirm(self, epochs: int | None = None, on_progress=None) -> dict:
        return self._fit_mapper(epochs, on_progress=on_progress)

    def recycle_sample(self, sample_index: int) -> None:
        """Move a labeled sample into the recycle bin."""
        sample_index = int(sample_index)
        if sample_index not in self.index_to_row:
            raise ValueError("sample is outside the annotation pool")
        if self.graph.category_of(sample_index) is None:
            raise ValueError("only a labeled sample can move to the recycle bin")
        self.graph.assign(sample_index, RECYCLE_ID, round_index=self.round_number + 1)
        self.unplaced.discard(sample_index)
        self.root_waiting.discard(sample_index)
        self.save()

    def import_labels(
        self, csv_text: str, index_column: str, label_column: str, on_progress=None
    ) -> dict:
        """Add one label column to the current graph, train, and stay editable."""
        pairs = parse_label_csv(csv_text, index_column, label_column)
        present = set(filter_catalog_indices(self.feature_path, [index for index, _ in pairs]))
        kept = [(index, label) for index, label in pairs if index in present]
        skipped = len(pairs) - len(kept)
        if not kept:
            raise ValueError("none of the labeled indexes are in the feature file")
        self._ensure_pool([index for index, _ in kept])
        self._merge_rows(kept, flat=True, on_progress=on_progress)
        trained = self._fit_mapper(self.epochs, on_progress=self._import_train_progress(len(kept), on_progress))
        return {
            "n_assigned": len(kept),
            "n_categories": self.graph.class_count(),
            "n_skipped": skipped,
            "trained": trained["trained"],
        }

    def import_hierarchy(
        self,
        csv_text: str,
        index_column: str | None = None,
        label_columns: list[str] | None = None,
        on_progress=None,
    ) -> dict:
        """Add the chosen label columns to the current graph, then train.

        A label that already exists under the same parent receives the sample.
        A new name becomes a new category. Categories already on the graph stay.
        """
        rows = parse_hierarchy_csv(
            csv_text, index_column=index_column, label_columns=label_columns
        )
        present = set(filter_catalog_indices(self.feature_path, [index for index, _ in rows]))
        kept = [(index, chain) for index, chain in rows if index in present]
        skipped = len(rows) - len(kept)
        if not kept:
            raise ValueError("none of the labeled indexes are in the feature file")
        self._ensure_pool([index for index, _ in kept])
        self._merge_rows(kept, flat=False, on_progress=on_progress)
        trained = self._fit_mapper(self.epochs, on_progress=self._import_train_progress(len(kept), on_progress))
        return {
            "n_assigned": len(kept),
            "n_categories": self.graph.class_count(),
            "n_skipped": skipped,
            "trained": trained["trained"],
        }

    def export_labels(self) -> str:
        return export_hierarchy_csv(self.graph)

    def set_ui(self, payload: dict) -> None:
        if "frozen" in payload:
            self.frozen = bool(payload["frozen"])
        if "edge_mode" in payload:
            mode = str(payload["edge_mode"])
            if mode not in ("strength", "distance"):
                raise ValueError("unknown edge mode")
            self.edge_mode = mode
        if "recommend_ids" in payload:
            self.recommend_set = [str(value) for value in payload["recommend_ids"] or []]
        if "shot_size" in payload:
            size = str(payload["shot_size"])
            if size not in ("large", "small"):
                raise ValueError("unknown shot size")
            self.shot_size = size
        self.save()

    def reset_mapper(self, confirm: bool) -> None:
        if not confirm:
            raise ValueError("reset requires confirm true")
        self._replace_mapper(int(self.mapper.hidden_dim))

    def set_mapper_hidden(self, hidden_dim: int) -> None:
        """Drop the current map and require a new training run.

        The new module is randomly initialized. Old weights are not loaded.
        """
        hidden_dim = int(hidden_dim)
        allowed = {int(item["hidden_dim"]) for item in mapper_options(self.in_dim, self.out_dim)}
        if hidden_dim not in allowed:
            raise ValueError("unknown mapper size")
        if hidden_dim == int(self.mapper.hidden_dim):
            return
        self._replace_mapper(hidden_dim)

    def set_head_hidden(self, hidden) -> None:
        pair = (int(hidden[0]), int(hidden[1]))
        width = self._head_input_dim()
        allowed = {
            (int(item["hidden"][0]), int(item["hidden"][1]))
            for item in classifier_options(width, self._option_class_count())
        }
        if pair not in allowed:
            raise ValueError("unknown model size")
        count = head_parameter_count(width, self._option_class_count(), pair)
        if count > CLASSIFIER_CAP:
            raise ValueError(f"model has {count} parameters, above {CLASSIFIER_CAP}")
        self.head_hidden = pair
        self.save()

    def set_feature_source(self, source: str) -> None:
        """Train later heads on raw catalog features or the current map."""
        chosen = str(source)
        if chosen not in ("raw", "mapped"):
            raise ValueError("unknown feature source")
        if chosen == "mapped" and not self.mapper_trained:
            raise ValueError("mapped features need a trained mapper")
        self.feature_source = chosen
        self._clamp_head_hidden()
        self.save()

    def set_task_draft(self, draft: dict) -> None:
        classes = draft.get("classes") or []
        cleaned = []
        for item in classes:
            category_ids = [str(value) for value in item.get("category_ids", [])]
            if any(self.graph.is_recycled_category(category_id) for category_id in category_ids):
                raise ValueError("the recycle bin is not a task class")
            name = str(item["name"]).strip()
            if "|" in name:
                raise ValueError("task class name cannot contain |")
            cleaned.append(
                {
                    "name": name,
                    "category_ids": category_ids,
                }
            )
        self.task_draft = {
            "classes": cleaned,
            "order": [str(value) for value in draft.get("order") or []],
            "strength": float(draft.get("strength", 1.0)),
        }
        self.save()

    def train_classification(
        self,
        classes: list[dict],
        epochs: int = 40,
        fractions=None,
        on_progress=None,
        features=None,
    ) -> dict:
        pairs = [(item["name"], list(item["category_ids"])) for item in classes]
        labels = task_labels(self.graph, pairs)
        names = [item["name"] for item in classes]
        if len(set(labels.values())) < 2:
            raise ValueError("classification needs at least two classes with samples")
        fracs = _coerce_fractions(fractions)
        groups: dict[str, list[int]] = {}
        for index, name in labels.items():
            groups.setdefault(str(name), []).append(int(index))
        split = stratified_split(groups, fracs, self.seed)
        train_labels = {
            int(index): name
            for name, parts in split.items()
            for index in parts["train"]
        }
        if len(set(train_labels.values())) < 2:
            raise ValueError("classification needs at least two classes with samples")
        self._use_feature_source(features)
        task_id = self._allocate_task_id()
        result, model, eval_device = self._train_heads(
            train_labels, names, epochs, task_id, on_progress=on_progress
        )
        summary = self._classification_summary(
            names, fracs, split, labels, model, eval_device
        )
        self.task_summary = summary
        self._remember_task(task_id, "classification", classes, [], None, result, summary)
        self.save()
        self._write_task_result(task_id, result)
        return result

    def train_ranking(
        self,
        classes: list[dict],
        order: list[str],
        strength: float,
        epochs: int = 40,
        on_progress=None,
        features=None,
    ) -> dict:
        pairs = [(item["name"], list(item["category_ids"])) for item in classes]
        labels = task_labels(self.graph, pairs)
        if float(strength) <= 0:
            raise ValueError("preference strength must be positive")
        indices = sorted(labels)
        groups = [labels[index] for index in indices]
        if len(set(groups)) < 2:
            raise ValueError("ranking needs at least two classes with samples")
        self._use_feature_source(features)
        mapped = self._active_feature_source() == "mapped"
        matrix = self._rows(indices, mapped=mapped)
        self._require_head_under_cap(matrix.shape[1], 1)
        model = fit_ranker(
            matrix,
            groups,
            order,
            float(strength),
            seed=0,
            epochs=epochs,
            device=self.device,
            hidden=self.head_hidden,
            on_progress=self._phase_progress(on_progress, 0, 1),
        )
        task_id = self._allocate_task_id()
        task_dir = self.directory / "tasks" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.net.state_dict(), task_dir / ("mapped.pt" if mapped else "raw.pt"))
        accuracy = pairwise_preference_accuracy(model.score(matrix), groups, order)
        result = {
            "n": len(indices),
            "features": self._active_feature_source(),
            "raw_pairwise_accuracy": None if mapped else accuracy,
            "mapped_pairwise_accuracy": accuracy if mapped else None,
        }
        result["score_board"] = self._ranking_board(model)
        summary = {
            "kind": "ranking",
            "n": int(result["n"]),
            "features": self._active_feature_source(),
            "raw_pairwise_accuracy": result["raw_pairwise_accuracy"],
            "mapped_pairwise_accuracy": result["mapped_pairwise_accuracy"],
        }
        self.task_summary = summary
        self._remember_task(task_id, "ranking", classes, list(order), float(strength), result, summary)
        self.save()
        self._write_task_result(task_id, result)
        return result

    def show_task(self, task_id: str) -> None:
        found = next((task for task in self.tasks if task["id"] == task_id), None)
        if found is None:
            raise KeyError(f"unknown task: {task_id}")
        self.active_task_id = task_id
        self.score_board = found["score_board"]
        self.task_summary = found.get("task_summary")
        self.save()

    def snapshot(self) -> dict:
        children = self._board_categories(self.view_parent)
        strengths = []
        for i, left in enumerate(children):
            for right in children[i + 1 :]:
                strengths.append(
                    {
                        "a": left.id,
                        "b": right.id,
                        "a_name": left.name,
                        "b_name": right.name,
                        "strength": self.graph.relation_strength(left.id, right.id),
                        "distance": self._between_distance(left.id, right.id),
                    }
                )
        crumbs = []
        if self.view_parent is not None:
            for category_id in self.graph.path(self.view_parent):
                crumbs.append(
                    {"id": category_id, "name": self.graph.categories[category_id].name}
                )
        if self.view_parent:
            waiting = self.graph.waiting(self.view_parent)
        else:
            waiting = sorted(self.root_waiting)
        predicted = {int(item["index"]): item for item in self.pool_predictions}
        images = []
        for index in sorted(self.unplaced):
            images.append(_image_row(index, "unplaced", predicted, self.graph))
        images.extend(
            _image_row(index, "waiting", predicted, self.graph)
            for index in waiting
            if index not in self.unplaced
        )
        return {
            "id": self.session_id,
            "feature_path": self.feature_path,
            "img_dir": self.img_dir,
            "pool_size": len(self.pool_indices),
            "out_dim": self.out_dim,
            "in_dim": self.in_dim,
            "round_size": self.round_size,
            "round_number": self.round_number,
            "view_parent": self.view_parent,
            "breadcrumb": crumbs,
            "categories": [self._category_card(cat) for cat in children],
            "all_categories": [
                {
                    "id": cat.id,
                    "name": cat.name,
                    "parent_id": cat.parent_id,
                    "color": category_color(cat.id),
                    "count": len(self.graph.subtree_samples(cat.id)),
                    "path": self.graph.name_path(cat.id),
                }
                for cat in self.graph.children_depth_first()
                if not self.graph.is_recycled_category(cat.id)
            ],
            "labeled_images": [
                _image_row(index, "labeled", predicted, self.graph) for index in self._labeled_ids()
            ],
            "recycle": {
                "id": RECYCLE_ID,
                "count": len(self.graph.samples(RECYCLE_ID)),
                "images": [
                    _image_row(index, "recycled", predicted, self.graph)
                    for index in self.graph.samples(RECYCLE_ID)
                ],
            },
            "strengths": strengths,
            "images": images,
            "mechanisms": MECHANISMS,
            "mapper_trained": self.mapper_trained,
            "mapper_hidden": int(self.mapper.hidden_dim),
            "mapper_sizes": mapper_options(self.in_dim, self.out_dim),
            "feature_source": self._active_feature_source(),
            "head_hidden": [int(self.head_hidden[0]), int(self.head_hidden[1])],
            "head_sizes": classifier_options(self._head_input_dim(), self._option_class_count()),
            "classifier_cap": CLASSIFIER_CAP,
            "layout_token": int(self._layout_token),
            "distances": self.distance_history,
            "recommend_set": list(self.recommend_set),
            "frozen": self.frozen,
            "edge_mode": self.edge_mode,
            "shot_size": self.shot_size,
            "can_confirm": any(index in self.index_to_row for index in self.graph.training_samples()),
            "task_draft": self.task_draft,
            "tasks": [
                {
                    "id": task["id"],
                    "kind": task["kind"],
                    "name": task["name"],
                    "active": task["id"] == self.active_task_id,
                }
                for task in self.tasks
            ],
            "score_board": _annotate_board(self.score_board, set(self.graph.assignments)),
            "task_summary": _json_ready(self.task_summary),
        }

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(self.directory / "pool_features.npy", self.features)
        state = {
            "session_id": self.session_id,
            "feature_path": self.feature_path,
            "pool_indices": self.pool_indices,
            "out_dim": self.out_dim,
            "round_size": self.round_size,
            "seed": self.seed,
            "img_dir": self.img_dir,
            "device": self.device,
            "epochs": self.epochs,
            "view_parent": self.view_parent,
            "round_ids": self.round_ids,
            "unplaced": sorted(self.unplaced),
            "round_number": self.round_number,
            "mapper_trained": self.mapper_trained,
            "head_hidden": [int(self.head_hidden[0]), int(self.head_hidden[1])],
            "feature_source": self.feature_source,
            "distance_history": self.distance_history,
            "task_draft": self.task_draft,
            "tasks": self.tasks,
            "active_task_id": self.active_task_id,
            "task_seq": self._task_seq,
            "score_board": self.score_board,
            "task_summary": self.task_summary,
            "root_waiting": sorted(self.root_waiting),
            "pool_predictions": self.pool_predictions,
            "recommend_set": list(self.recommend_set),
            "frozen": self.frozen,
            "edge_mode": self.edge_mode,
            "shot_size": self.shot_size,
        }
        (self.directory / "session.json").write_text(json.dumps(_json_ready(state), indent=2))
        with (self.directory / "rng.pkl").open("wb") as handle:
            pickle.dump(self.rng, handle)
        history_path = self.directory / "distance_history.csv"
        if history_path.exists():
            history_path.unlink()
        save_bundle(
            self.directory,
            graph=self.graph,
            mapper=self.mapper,
            feature_path=self.feature_path,
            distance_row=None,
        )
        if self.distance_history:
            from hitl_features.bundle import append_distance_row

            for row in self.distance_history:
                append_distance_row(history_path, row)

    @classmethod
    def load(cls, directory: Path) -> "LabSession":
        directory = Path(directory)
        state = json.loads((directory / "session.json").read_text())
        features = np.load(directory / "pool_features.npy")
        session = cls(
            session_id=state["session_id"],
            directory=directory,
            feature_path=state["feature_path"],
            pool_indices=state["pool_indices"],
            features=features,
            out_dim=state["out_dim"],
            round_size=state["round_size"],
            seed=state["seed"],
            img_dir=state.get("img_dir") or "",
            device=state.get("device") or "cpu",
            epochs=state.get("epochs") or 30,
        )
        bundle = load_bundle(directory)
        session.graph = bundle.graph
        session.mapper = bundle.mapper
        session.view_parent = state.get("view_parent")
        session.round_ids = [int(value) for value in state.get("round_ids") or []]
        session.unplaced = {int(value) for value in state.get("unplaced") or []}
        session.round_number = int(state.get("round_number") or 0)
        session.mapper_trained = bool(state.get("mapper_trained"))
        saved_head = state.get("head_hidden") or [64, 32]
        session.head_hidden = (int(saved_head[0]), int(saved_head[1]))
        saved_source = str(state.get("feature_source") or "mapped")
        session.feature_source = saved_source if saved_source in ("raw", "mapped") else "mapped"
        session._clamp_head_hidden()
        session.distance_history = list(state.get("distance_history") or [])
        session.task_draft = state.get("task_draft") or session.task_draft
        session.tasks = list(state.get("tasks") or [])
        session.active_task_id = state.get("active_task_id")
        session._task_seq = int(state.get("task_seq") or 0)
        session.score_board = state.get("score_board")
        session.task_summary = state.get("task_summary")
        session.root_waiting = {int(value) for value in state.get("root_waiting") or []}
        session.pool_predictions = [
            {
                "index": int(item["index"]),
                "category_id": str(item["category_id"]),
                "level": str(item["level"]),
            }
            for item in state.get("pool_predictions") or []
        ]
        session.recommend_set = [str(value) for value in state.get("recommend_set") or []]
        session.frozen = bool(state.get("frozen"))
        session.edge_mode = str(state.get("edge_mode") or "distance")
        session.shot_size = str(state.get("shot_size") or "small")
        rng_path = directory / "rng.pkl"
        if rng_path.is_file():
            with rng_path.open("rb") as handle:
                session.rng = pickle.load(handle)
        return session

    def _pool_matrix(self, mapped: bool) -> np.ndarray:
        if not mapped:
            return self.features
        if self._mapped_cache is None:
            self._mapped_cache = self._rows(self.pool_indices, mapped=True)
        return self._mapped_cache

    def _rows(self, indices: list[int], mapped: bool) -> np.ndarray:
        raw = np.stack([self.features[self.index_to_row[index]] for index in indices])
        if not mapped:
            return raw
        self.mapper.to(self.device)
        self.mapper.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(raw).to(self.device)
            return self.mapper(inputs).cpu().numpy()

    def _active_feature_source(self) -> str:
        """Mapped features stay unavailable until this session has a trained map."""
        if self.feature_source == "mapped" and self.mapper_trained:
            return "mapped"
        return "raw"

    def _head_input_dim(self) -> int:
        return self.in_dim if self._active_feature_source() == "raw" else self.out_dim

    def _use_feature_source(self, features) -> None:
        if features is None:
            return
        source = str(features)
        if source not in ("raw", "mapped"):
            raise ValueError("unknown feature source")
        if source == "mapped" and not self.mapper_trained:
            raise ValueError("mapped features need a trained mapper")
        self.feature_source = source
        self._clamp_head_hidden()

    def _clamp_head_hidden(self) -> None:
        width = self._head_input_dim()
        outputs = self._option_class_count()
        options = classifier_options(width, outputs)
        allowed = {(int(item["hidden"][0]), int(item["hidden"][1])) for item in options}
        count = head_parameter_count(width, outputs, self.head_hidden)
        if self.head_hidden not in allowed or count > CLASSIFIER_CAP:
            self.head_hidden = (64, 32)

    def _bump_layout(self) -> None:
        self._layout_token += 1
        self._umap_cache = None

    def umap_layout(self) -> dict:
        """Raw and mapped coordinates for every sample in the candidate pool.

        Coordinates are cached until the features or the map change. Colors are
        filled on each call so a new label does not refit UMAP.
        """
        if self._umap_cache is None or self._umap_cache.get("token") != self._layout_token:
            from hitl_features.umap_layout import embed_points

            self._umap_cache = {
                "token": self._layout_token,
                "raw": embed_points(self.features, self.seed),
                "mapped": embed_points(self._pool_matrix(mapped=True), self.seed + 1),
            }
        colors: dict[int, str] = {}
        for index, owner in self.graph.assignments.items():
            if index not in self.index_to_row or self.graph.is_recycled_sample(index):
                continue
            colors[int(index)] = category_color(owner)

        def pack(xy: np.ndarray) -> list[dict]:
            rows = []
            for row, index in enumerate(self.pool_indices):
                rows.append(
                    {
                        "index": int(index),
                        "x": float(xy[row, 0]),
                        "y": float(xy[row, 1]),
                        "color": colors.get(int(index)),
                    }
                )
            return rows

        cached = self._umap_cache
        return {
            "token": int(self._layout_token),
            "raw": pack(cached["raw"]),
            "mapped": pack(cached["mapped"]),
        }

    def _train_heads(
        self, labels: dict[int, str], names: list[str], epochs: int, task_id: str, on_progress=None
    ):
        indices = sorted(labels)
        name_to_i = {name: index for index, name in enumerate(names)}
        y = np.array([name_to_i[labels[index]] for index in indices], dtype=np.int64)
        mapped = self._active_feature_source() == "mapped"
        matrix = self._rows(indices, mapped=mapped)
        n_out = int(y.max()) + 1
        self._require_head_under_cap(matrix.shape[1], n_out)
        model = fit_classifier(
            matrix,
            y,
            seed=0,
            epochs=epochs,
            device=self.device,
            hidden=self.head_hidden,
            on_progress=self._phase_progress(on_progress, 0, 1),
        )
        task_dir = self.directory / "tasks" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), task_dir / ("mapped.pt" if mapped else "raw.pt"))
        eval_device = self.device if str(self.device).startswith("cuda") else "cpu"
        if eval_device == "cpu":
            model = model.cpu()
        if len(set(int(value) for value in y.tolist())) > 2:
            metrics = evaluate_multiclass(model, matrix, y, device=eval_device)
        else:
            metrics = evaluate_binary(model, matrix, y, device=eval_device)
        result = {
            "n": len(indices),
            "features": self._active_feature_source(),
            "raw": None,
            "mapped": None,
        }
        result[self._active_feature_source()] = metrics
        result["score_board"] = self._classification_board(model, names, eval_device)
        return result, model, eval_device

    def _classification_summary(
        self, names, fracs, split, labels, model, device
    ) -> dict:
        """Short metrics for each split, plus scores for the non-training members."""
        multiclass = len({name for name, parts in split.items() if parts["train"]}) > 2
        splits = {}
        train_ids: list[int] = []
        for part in ("train", "val", "test"):
            named = {
                int(index): name
                for name, parts in split.items()
                for index in parts[part]
            }
            if part == "train":
                train_ids = sorted(named)
            splits[part] = self._measure_split(model, named, names, multiclass, device)
        members = sorted(int(index) for index in labels)
        chosen = inference_ids(members, train_ids)
        return {
            "kind": "classification",
            "features": self._active_feature_source(),
            "fractions": {"train": fracs[0], "val": fracs[1], "test": fracs[2]},
            "train_ids": train_ids,
            "splits": splits,
            "inference": self._inference_scores(model, chosen, device),
        }

    def _measure_split(self, model, named, names, multiclass, device) -> dict:
        empty = {
            "n": 0,
            "raw_accuracy": None,
            "mapped_accuracy": None,
            "raw_roc_auc": None,
            "mapped_roc_auc": None,
        }
        if not named:
            return empty
        indices = sorted(named)
        name_to_i = {name: index for index, name in enumerate(names)}
        y = np.array([name_to_i[named[index]] for index in indices], dtype=np.int64)
        mapped = self._active_feature_source() == "mapped"
        matrix = self._rows(indices, mapped=mapped)
        if multiclass:
            metrics = evaluate_multiclass(model, matrix, y, device=device)
            auc = metrics.get("macro_roc_auc")
        else:
            metrics = evaluate_binary(model, matrix, y, device=device)
            auc = metrics.get("roc_auc")
        prefix = "mapped" if mapped else "raw"
        row = dict(empty)
        row["n"] = len(indices)
        row[prefix + "_accuracy"] = float(metrics["accuracy"])
        row[prefix + "_roc_auc"] = None if auc is None else float(auc)
        return row

    def _inference_scores(self, model, indices: list[int], device: str) -> list[dict]:
        if not indices:
            return []
        features = self._rows(indices, mapped=self._active_feature_source() == "mapped")
        proba = predict_proba(model, features, device=device)
        picked = np.argmax(proba, axis=1)
        rows = []
        for row, index in enumerate(indices):
            rows.append({"index": int(index), "score": float(proba[row, int(picked[row])])})
        return rows

    def _board_categories(self, parent_id: str | None):
        return [
            cat
            for cat in self.graph.children(parent_id)
            if not self.graph.is_recycled_category(cat.id)
        ]

    def _labeled_ids(self) -> list[int]:
        if self.view_parent is None:
            chosen = [
                index
                for index in self.graph.assignments
                if not self.graph.is_recycled_sample(index)
            ]
        else:
            chosen = [
                index
                for index in self.graph.subtree_samples(self.view_parent)
                if not self.graph.is_recycled_sample(index)
            ]
        return sorted(index for index in chosen if index in self.index_to_row)

    def _option_class_count(self) -> int:
        names = [
            str(item.get("name") or "").strip()
            for item in (self.task_draft.get("classes") or [])
        ]
        named = [name for name in names if name]
        return max(2, len(named))

    def _require_head_under_cap(self, in_dim: int, n_out: int) -> None:
        count = head_parameter_count(in_dim, n_out, self.head_hidden)
        if count > CLASSIFIER_CAP:
            raise ValueError(f"model has {count} parameters, above {CLASSIFIER_CAP}")

    def _replace_mapper(self, hidden_dim: int) -> None:
        self.mapper = FeatureMapper(self.in_dim, self.out_dim, hidden_dim=int(hidden_dim))
        self.mapper_trained = False
        self._mapped_cache = None
        self.pool_predictions = []
        self._bump_layout()
        self.save()

    def _phase_progress(self, on_progress, phase: int, phases: int):
        def inner(done: int, total: int) -> None:
            if on_progress is None or int(total) <= 0:
                return
            on_progress(int(phase) * int(total) + int(done), int(phases) * int(total))

        return inner

    def _import_train_progress(self, n_rows: int, on_progress):
        def inner(done: int, total: int) -> None:
            if on_progress is None:
                return
            on_progress(int(n_rows) + int(done), int(n_rows) + int(total))

        return inner

    def _merge_rows(self, rows, flat: bool, on_progress) -> None:
        epoch_n = max(0, int(self.epochs))
        total = len(rows) + epoch_n
        for step, row in enumerate(rows, start=1):
            if flat:
                merge_labels(self.graph, [row])
            else:
                merge_hierarchy(self.graph, [row])
            if on_progress is not None:
                on_progress(step, total if total else step)
        imported = {int(index) for index, _ in rows}
        self.unplaced -= imported
        self.root_waiting -= imported
        self.round_ids = [index for index in self.round_ids if index not in imported]
        if self.view_parent is not None and self.view_parent not in self.graph.categories:
            self.view_parent = None

    def _fit_mapper(self, epochs: int | None = None, on_progress=None) -> dict:
        assigned = [index for index in self.graph.training_samples() if index in self.index_to_row]
        if not assigned:
            raise RuntimeError("assign at least one image before training")
        step_epochs = self.epochs if epochs is None else int(epochs)
        rows = np.stack([self.features[self.index_to_row[index]] for index in assigned])
        trained = train_mapper(
            self.mapper,
            rows,
            assigned,
            self.graph,
            epochs=step_epochs,
            device=self.device,
            on_progress=on_progress,
        )
        if on_progress is not None and step_epochs > 0:
            on_progress(step_epochs, step_epochs)
        if trained["steps"] > 0:
            self.mapper_trained = True
            self._clamp_head_hidden()
            self._bump_layout()
        self._mapped_cache = None
        mapped = self._rows(assigned, mapped=True)
        within, between = mean_within_between(
            mapped, [self.graph.category_of(index) for index in assigned]
        )
        self.round_number += 1
        self.round_ids = []
        row = {
            "round": self.round_number,
            "mean_within": within,
            "mean_between": between,
            "loss": trained["loss"],
            "trained": trained["steps"] > 0,
        }
        self.distance_history.append(row)
        self._refresh_pool_predictions()
        self.save()
        return row

    def _refresh_pool_predictions(self) -> None:
        """Replace pool guesses from the current mapper, or clear them.

        Without a trained mapper there is nothing to embed, so the full
        catalog is left unread.
        """
        if not self.mapper_trained:
            self.pool_predictions = []
            return
        self.pool_predictions = predict_pool(
            self.graph, self.pool_indices, self._pool_matrix(mapped=True)
        )

    def _ensure_pool(self, indices: list[int]) -> None:
        missing = [index for index in indices if index not in self.index_to_row]
        if not missing:
            return
        rows = load_feature_rows(self.feature_path, missing)
        self.features = np.concatenate([self.features, rows], axis=0)
        self.pool_indices.extend(missing)
        self.index_to_row = {index: row for row, index in enumerate(self.pool_indices)}
        self._mapped_cache = None
        self._bump_layout()

    def _allocate_task_id(self) -> str:
        self._task_seq += 1
        return f"t{self._task_seq}"

    def _remember_task(
        self, task_id: str, kind: str, classes, order, strength, result: dict, task_summary=None
    ) -> None:
        board = result.get("score_board")
        count = sum(1 for task in self.tasks if task["kind"] == kind)
        name = ("Classification " if kind == "classification" else "Ranking ") + str(count + 1)
        if isinstance(board, dict):
            board["name"] = name
            board["task_id"] = task_id
        result["task_id"] = task_id
        result["name"] = name
        self.tasks.append(
            {
                "id": task_id,
                "kind": kind,
                "name": name,
                "classes": classes,
                "order": list(order),
                "strength": strength,
                "summary": {
                    key: value
                    for key, value in result.items()
                    if key != "score_board"
                },
                "score_board": board,
                "task_summary": task_summary,
            }
        )
        self.active_task_id = task_id

    def _write_task_result(self, task_id: str, result: dict) -> None:
        task_dir = self.directory / "tasks" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "result.json").write_text(json.dumps(_json_ready(result), indent=2))

    def _category_card(self, cat) -> dict:
        members = self.graph.subtree_samples(cat.id)
        preview = {"recent": [], "representative": []}
        if members:
            orders = [self.graph.assignment_order.get(index, 0) for index in members]
            vectors = self._rows(members, mapped=self.mapper_trained)
            preview = category_preview(members, vectors, orders)
        position = self.graph.positions.get(cat.id)
        return {
            "id": cat.id,
            "name": cat.name,
            "parent_id": cat.parent_id,
            "path": self.graph.name_path(cat.id),
            "count": len(members),
            "color": category_color(cat.id),
            "x": None if position is None else position[0],
            "y": None if position is None else position[1],
            "recent": preview["recent"],
            "representative": preview["representative"],
            "outline_strength": self._outline_strength(cat.id),
            "outline_distance": self._within_distance(cat.id),
        }

    def _member_matrix(self, category_id: str) -> np.ndarray | None:
        if not self.mapper_trained:
            return None
        indexes = [
            index for index in self.graph.subtree_samples(category_id) if index in self.index_to_row
        ]
        if not indexes:
            return np.zeros((0, self.out_dim), dtype=np.float32)
        return self._rows(indexes, mapped=True)

    def _between_distance(self, left_id: str, right_id: str) -> float | None:
        left = self._member_matrix(left_id)
        right = self._member_matrix(right_id)
        if left is None or right is None:
            return None
        return mean_cross_distance(left, right)

    def _within_distance(self, category_id: str) -> float | None:
        matrix = self._member_matrix(category_id)
        if matrix is None:
            return None
        return mean_within_distance(matrix)

    def _outline_strength(self, category_id: str) -> float | None:
        children = self.graph.children(category_id)
        if len(children) < 2:
            return None
        values = []
        for index, left in enumerate(children):
            for right in children[index + 1 :]:
                values.append(self.graph.relation_strength(left.id, right.id))
        if not values:
            return None
        return float(np.mean(values))

    def _fixed_board_rows(self) -> tuple[list[int], np.ndarray]:
        indices = fixed_score_indices(self.pool_indices, self.seed)
        mapped = self._active_feature_source() == "mapped"
        width = self.out_dim if mapped else self.in_dim
        if not indices:
            return [], np.zeros((0, width), dtype=np.float32)
        return indices, self._rows(indices, mapped=mapped)

    def _classification_board(self, model, names: list[str], device: str) -> dict:
        indices, features = self._fixed_board_rows()
        if not indices:
            scores = []
            labels = []
        else:
            proba = predict_proba(model, features, device=device)
            predicted = proba.argmax(axis=1)
            scores = proba[np.arange(len(predicted)), predicted]
            labels = [
                names[int(class_index)] if int(class_index) < len(names) else str(int(class_index))
                for class_index in predicted
            ]
        return self._store_board("classification", indices, scores, labels)

    def _ranking_board(self, model) -> dict:
        indices, features = self._fixed_board_rows()
        scores = model.score(features) if indices else []
        return self._store_board("ranking", indices, scores, None)

    def _store_board(self, kind: str, indices: list[int], scores, labels) -> dict:
        board = {
            "kind": kind,
            "indices": [int(value) for value in indices],
            "n_bins": 5,
            "per_bin": 4,
            "bins": equal_width_bins(indices, scores, labels=labels),
        }
        self.score_board = board
        return board


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, LabSession] = {}
        for path in self.root.iterdir():
            if (path / "session.json").is_file():
                session = LabSession.load(path)
                self._sessions[session.session_id] = session

    def create(
        self,
        *,
        feature_path: str,
        pool_size: int,
        seed: int,
        out_dim: int,
        round_size: int,
        img_dir: str = "",
        device: str = "cpu",
        epochs: int = 30,
        pool_indices: list[int] | None = None,
    ) -> LabSession:
        if int(out_dim) < 1:
            raise ValueError("out_dim must be positive")
        catalog = catalog_indices(feature_path)
        n_rows = int(catalog.shape[0])
        identity = bool(np.array_equal(catalog, np.arange(n_rows, dtype=np.int64)))
        draw = np.random.default_rng(int(seed))
        if pool_indices is None:
            take = min(int(pool_size), n_rows)
            if take < 1:
                raise ValueError("feature file has no rows")
            if identity:
                chosen = draw.choice(n_rows, size=take, replace=False)
            else:
                chosen = draw.choice(catalog, size=take, replace=False)
            chosen.sort()
            indices = [int(value) for value in np.asarray(chosen).reshape(-1)]
        else:
            indices = [int(value) for value in pool_indices]
            if identity:
                outside = any(index < 0 or index >= n_rows for index in indices)
            else:
                present = {int(value) for value in catalog.tolist()}
                outside = any(index not in present for index in indices)
            if outside:
                raise ValueError("pool index is outside the feature file")
        features = load_feature_rows(feature_path, indices)
        session_id = uuid.uuid4().hex[:8]
        directory = self.root / session_id
        session = LabSession(
            session_id,
            directory,
            feature_path,
            indices,
            features,
            out_dim,
            round_size,
            seed,
            img_dir,
            device,
            epochs,
        )
        session.save()
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> LabSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown session: {session_id}") from exc


def _coerce_fractions(fractions) -> tuple[float, float, float]:
    """Missing fractions stay at 0.6 / 0.2 / 0.2. Any supplied triple must sum to 1."""
    if fractions is None:
        return normalize_fractions(0.6, 0.2, 0.2)
    if isinstance(fractions, dict):
        return normalize_fractions(
            fractions.get("train", 0), fractions.get("val", 0), fractions.get("test", 0)
        )
    train, val, test = fractions
    return normalize_fractions(train, val, test)


def category_color(category_id: str) -> str:
    digits = "".join(ch for ch in str(category_id) if ch.isdigit())
    index = int(digits) if digits else 0
    return _CATEGORY_COLORS[(max(index, 1) - 1) % len(_CATEGORY_COLORS)]


def _child_under(graph: CategoryGraph, parent_id: str, category_id: str) -> str | None:
    path = graph.path(category_id)
    if parent_id not in path:
        return None
    position = path.index(parent_id)
    if position + 1 >= len(path):
        return None
    return path[position + 1]


def _image_row(index: int, role: str, predicted: dict, graph: CategoryGraph) -> dict:
    item = predicted.get(int(index))
    name = None
    level = None
    if item and item.get("category_id") in graph.categories:
        name = graph.categories[item["category_id"]].name
        level = item.get("level")
    return {
        "index": int(index),
        "role": role,
        "prediction": name,
        "prediction_level": level,
    }
