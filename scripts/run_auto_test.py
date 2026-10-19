"""Run the scripted four-way feature-mapping test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hitl_features.auto_test import AutoTestConfig, run_auto_test


def _repo_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Scripted HITL feature-map evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.config).read_text())
    config = AutoTestConfig(
        feature_path=_repo_path(payload["feature_path"]),
        labels_csv=_repo_path(payload["labels_csv"]),
        label_column=payload["label_column"],
        positive=payload["positive"],
        negatives=list(payload["negatives"]),
        index_column=payload.get("index_column", "index"),
        pool_size=args.pool_size or int(payload.get("pool_size", 5000)),
        round_size=int(payload.get("round_size", 12)),
        rounds=args.rounds or int(payload.get("rounds", 8)),
        out_dim=int(payload.get("out_dim", 32)),
        seed=int(payload.get("seed", 0)),
        epochs=args.epochs or int(payload.get("epochs", 30)),
        eval_fraction=float(payload.get("eval_fraction", 0.3)),
        classifier_epochs=int(payload.get("classifier_epochs", 40)),
        device=args.device or payload.get("device"),
        output_dir=args.output_dir,
        hidden_dim=int(payload.get("hidden_dim", 256)),
        img_dir=_repo_path(payload["img_dir"]) if payload.get("img_dir") else "",
    )
    report = run_auto_test(config)
    print(f"Wrote {args.output_dir}/report.json")
    print(
        "recommended",
        report["n_recommended"],
        "positives",
        report["recommended_positives"],
    )
    for name, metrics in report["four_way"].items():
        print(
            name,
            metrics.get("roc_auc"),
            metrics.get("average_precision"),
            metrics.get("note", ""),
        )


if __name__ == "__main__":
    main()
