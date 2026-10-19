"""Pool predictions are cosine kNN labels stored after mapper training."""

import h5py
import numpy as np

from hitl_features.graph import CategoryGraph
from hitl_features.pool_predict import PREDICT_K, predict_pool
from hitl_features.session import LabSession, SessionStore


def test_knn_uses_fifteen_neighbors_for_root_and_child():
    assert PREDICT_K == 15
    graph = CategoryGraph()
    smooth = graph.create_category("smooth")
    spiral = graph.create_category("spiral")
    barred = graph.create_category("barred", parent_id=spiral)
    plain = graph.create_category("plain", parent_id=spiral)
    vectors = []
    indices = []
    for index in range(14):
        graph.assign(index, smooth)
        vectors.append([1.0, 0.0])
        indices.append(index)
    for offset in range(20):
        index = 14 + offset
        graph.assign(index, barred)
        vectors.append([1.0, 1.0])
        indices.append(index)
    for offset in range(8):
        index = 40 + offset
        graph.assign(index, plain)
        vectors.append([0.0, 1.0])
        indices.append(index)
    graph.assign(50, spiral)
    vectors.append([1.0, 1.0])
    indices.append(50)
    graph.assign(51, barred)
    vectors.append([1.0, 1.0])
    indices.append(51)
    vectors.append([1.0, 0.0])
    indices.append(60)
    vectors.append([0.0, 1.0])
    indices.append(61)
    predicted = {
        (item["index"], item["level"]): item["category_id"]
        for item in predict_pool(graph, indices, np.asarray(vectors, dtype=np.float64))
    }
    assert predicted[(60, "root")] == smooth
    assert predicted[(61, "root")] == spiral
    assert predicted[(50, "child")] == barred
    assert (51, "root") not in predicted
    assert (51, "child") not in predicted
    assert (0, "root") not in predicted


def test_session_stores_predictions_only_after_training(tmp_path, monkeypatch):
    feature_path = tmp_path / "feat.h5"
    matrix = np.eye(4, dtype=np.float32)
    with h5py.File(feature_path, "w") as handle:
        handle.create_dataset("features", data=matrix)
    session = SessionStore(tmp_path / "sessions").create(
        feature_path=str(feature_path),
        pool_size=4,
        seed=0,
        out_dim=4,
        round_size=1,
        epochs=1,
        device="cpu",
        pool_indices=[0, 1, 2, 3],
    )
    assert session.pool_predictions == []
    parent = session.graph.create_category("spiral")
    child = session.graph.create_category("barred", parent_id=parent)
    session.graph.assign(0, child)
    session.graph.assign(1, child)
    session.graph.assign(2, parent)

    def _refuse_catalog_scan(*_args, **_kwargs):
        raise AssertionError("prediction scanned the feature catalog")

    monkeypatch.setattr("hitl_features.session.load_feature_rows", _refuse_catalog_scan)
    monkeypatch.setattr("hitl_features.session.catalog_indices", _refuse_catalog_scan)
    trained = session.confirm()
    assert trained["trained"] is True
    by_index = {item["index"]: item for item in session.pool_predictions}
    assert by_index[3] == {"index": 3, "category_id": parent, "level": "root"}
    assert by_index[2] == {"index": 2, "category_id": child, "level": "child"}
    assert 0 not in by_index
    assert 1 not in by_index
    loaded = LabSession.load(session.directory)
    assert loaded.pool_predictions == session.pool_predictions
    session.confirm()
    assert len(session.pool_predictions) == 2
    assert session.recommend_round("random") == [3]
    session.reset_mapper(True)
    assert session.mapper_trained is False
    assert session.pool_predictions == []
    assert LabSession.load(session.directory).pool_predictions == []
