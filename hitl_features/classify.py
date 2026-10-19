"""Small MLP classifier shared by the raw and mapped arms."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score

from hitl_features.metrics import binary_metrics
from hitl_features.modelsizes import normalize_hidden


class MlpClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: tuple[int, int] = (64, 32)) -> None:
        super().__init__()
        self.hidden = normalize_hidden(hidden)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), self.hidden[0]),
            nn.ReLU(),
            nn.Linear(self.hidden[0], self.hidden[1]),
            nn.ReLU(),
            nn.Linear(self.hidden[1], int(n_classes)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def fit_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    epochs: int = 40,
    lr: float = 1e-3,
    device: str = "cpu",
    hidden: tuple[int, int] | None = None,
    on_progress=None,
) -> MlpClassifier:
    matrix = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    classes = np.unique(targets)
    if len(classes) < 2:
        raise ValueError("classifier training needs at least two classes")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = MlpClassifier(matrix.shape[1], int(targets.max()) + 1, hidden=normalize_hidden(hidden)).to(
        device
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    inputs = torch.from_numpy(matrix).to(device)
    y = torch.from_numpy(targets).to(device)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    total = int(epochs)
    if on_progress is not None and total > 0:
        on_progress(0, total)
    for step in range(1, total + 1):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), y)
        loss.backward()
        optimizer.step()
        if on_progress is not None:
            on_progress(step, total)
    model.eval()
    return model


def predict_proba(model: MlpClassifier, features: np.ndarray, device: str = "cpu") -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(matrix).to(device))
        proba = torch.softmax(logits, dim=-1)
    return proba.cpu().numpy()


def evaluate_binary(
    model: MlpClassifier, features: np.ndarray, labels: np.ndarray, device: str = "cpu"
) -> dict:
    proba = predict_proba(model, features, device=device)
    scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    metrics = binary_metrics(labels, scores)
    predicted = proba.argmax(axis=1)
    metrics["accuracy"] = float(np.mean(predicted == np.asarray(labels).astype(int)))
    return metrics


def evaluate_multiclass(
    model: MlpClassifier, features: np.ndarray, labels: np.ndarray, device: str = "cpu"
) -> dict:
    """Accuracy plus macro one-vs-rest ROC AUC for classes present in the eval labels."""
    proba = predict_proba(model, features, device=device)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    predicted = proba.argmax(axis=1)
    aucs: list[float] = []
    for class_id in range(proba.shape[1]):
        positive = y == class_id
        if positive.any() and (~positive).any():
            aucs.append(float(roc_auc_score(positive, proba[:, class_id])))
    return {
        "accuracy": float(np.mean(predicted == y)),
        "macro_roc_auc": float(np.mean(aucs)) if aucs else None,
        "n": int(len(y)),
        "n_classes_scored": len(aucs),
    }
