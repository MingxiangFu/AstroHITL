"""Binary scores used by the four-way comparison."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    labels = np.asarray(y_true).astype(int).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != len(values):
        raise ValueError("labels and scores must have the same length")
    if len(np.unique(labels)) < 2:
        return {
            "roc_auc": None,
            "average_precision": None,
            "note": "need both classes",
        }
    return {
        "roc_auc": float(roc_auc_score(labels, values)),
        "average_precision": float(average_precision_score(labels, values)),
    }
