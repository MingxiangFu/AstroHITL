"""One graph, its sample index, and its feature map."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from hitl_features.graph import CategoryGraph
from hitl_features.mapper import FeatureMapper


@dataclass
class Bundle:
    graph: CategoryGraph
    mapper: FeatureMapper
    feature_path: str
    in_dim: int
    out_dim: int


def save_bundle(
    directory: Path | str,
    *,
    graph: CategoryGraph,
    mapper: FeatureMapper,
    feature_path: str,
    distance_row: dict | None = None,
) -> None:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    payload = graph.to_dict()
    (root / "graph.json").write_text(json.dumps(payload, indent=2))
    with (root / "assignments.jsonl").open("w") as handle:
        for row in payload["assignments"]:
            handle.write(json.dumps(row) + "\n")
    torch.save(
        {
            "state_dict": mapper.state_dict(),
            "in_dim": mapper.in_dim,
            "out_dim": mapper.out_dim,
            "hidden_dim": mapper.hidden_dim,
        },
        root / "mapper.pt",
    )
    meta = {
        "feature_path": feature_path,
        "in_dim": mapper.in_dim,
        "out_dim": mapper.out_dim,
        "hidden_dim": mapper.hidden_dim,
    }
    (root / "mapper_meta.json").write_text(json.dumps(meta, indent=2))
    if distance_row is not None:
        append_distance_row(root / "distance_history.csv", distance_row)


def load_bundle(directory: Path | str) -> Bundle:
    root = Path(directory)
    payload = json.loads((root / "graph.json").read_text())
    meta = json.loads((root / "mapper_meta.json").read_text())
    checkpoint = torch.load(root / "mapper.pt", map_location="cpu")
    mapper = FeatureMapper(
        int(checkpoint["in_dim"]),
        int(checkpoint["out_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    mapper.load_state_dict(checkpoint["state_dict"])
    mapper.eval()
    return Bundle(
        graph=CategoryGraph.from_dict(payload),
        mapper=mapper,
        feature_path=str(meta.get("feature_path", "")),
        in_dim=int(meta["in_dim"]),
        out_dim=int(meta["out_dim"]),
    )


def append_distance_row(path: Path, row: dict) -> None:
    fields = ["round", "mean_within", "mean_between", "loss", "trained"]
    exists = path.is_file()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: _cell(row.get(key)) for key in fields})


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)
