"""Four classifiers: recommended or random samples, raw or mapped features."""

from __future__ import annotations

import numpy as np

from hitl_features.classify import evaluate_binary, fit_classifier, predict_proba


def run_four_way(
    recommended_raw: np.ndarray,
    recommended_mapped: np.ndarray,
    recommended_y: np.ndarray,
    random_raw: np.ndarray,
    random_mapped: np.ndarray,
    random_y: np.ndarray,
    eval_raw: np.ndarray,
    eval_mapped: np.ndarray,
    eval_y: np.ndarray,
    *,
    seed: int,
    epochs: int = 40,
    device: str = "cpu",
    with_scores: bool = False,
):
    arms = [
        ("A_recommended_raw", recommended_raw, recommended_y, eval_raw),
        ("B_recommended_mapped", recommended_mapped, recommended_y, eval_mapped),
        ("C_random_raw", random_raw, random_y, eval_raw),
        ("D_random_mapped", random_mapped, random_y, eval_mapped),
    ]
    report: dict = {}
    held_scores: dict = {}
    for name, train_x, train_y, holdout_x in arms:
        labels = np.asarray(train_y).astype(int)
        if len(np.unique(labels)) < 2:
            report[name] = {
                "roc_auc": None,
                "average_precision": None,
                "note": "training split has one class",
            }
            held_scores[name] = None
            continue
        model = fit_classifier(
            train_x, labels, seed=seed, epochs=epochs, device=device
        )
        model_device = device if str(device).startswith("cuda") else "cpu"
        if model_device == "cpu":
            model = model.cpu()
        report[name] = evaluate_binary(model, holdout_x, eval_y, device=model_device)
        if with_scores:
            proba = predict_proba(model, holdout_x, device=model_device)
            held_scores[name] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    if with_scores:
        return report, held_scores
    return report
