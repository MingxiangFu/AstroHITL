"""Run the expert main/secondary hierarchy test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hitl_features.expert_test import ExpertTestConfig, run_expert_hierarchy


def main() -> None:
    parser = argparse.ArgumentParser(description="Expert hierarchy feature-map evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--round-size", type=int)
    parser.add_argument("--classifier-epochs", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.config).read_text())
    config = ExpertTestConfig(
        feature_path=payload["feature_path"],
        labels_csv=payload["labels_csv"],
        index_column=payload.get("index_column", "index"),
        main_column=payload.get("main_column", "main_label_expert"),
        secondary_column=payload.get("secondary_column", "secondary_label_expert"),
        pool_size=args.pool_size or int(payload.get("pool_size", 400)),
        round_size=args.round_size or int(payload.get("round_size", 16)),
        rounds=args.rounds or int(payload.get("rounds", 4)),
        out_dim=int(payload.get("out_dim", 32)),
        seed=int(payload.get("seed", 0)),
        epochs=args.epochs or int(payload.get("epochs", 8)),
        eval_fraction=float(payload.get("eval_fraction", 0.3)),
        classifier_epochs=args.classifier_epochs or int(payload.get("classifier_epochs", 20)),
        device=args.device or payload.get("device"),
        output_dir=args.output_dir,
        hidden_dim=int(payload.get("hidden_dim", 128)),
    )
    report = run_expert_hierarchy(config)
    print(f"Wrote {args.output_dir}/report.json")
    print("recommended", report["n_recommended"], "mains", report["recommended_main_counts"])
    for name, metrics in report["main_classification"].items():
        print(
            "main",
            name,
            "acc",
            metrics.get("accuracy"),
            "macro_auc",
            metrics.get("macro_roc_auc"),
            metrics.get("note", ""),
        )
    raw_agree = report["main_score_agreement"]["raw"].get("mean_pairwise_agreement")
    mapped_agree = report["main_score_agreement"]["mapped"].get("mean_pairwise_agreement")
    print("main score agreement raw", raw_agree, "mapped", mapped_agree)
    for main, subgraph in report["subgraphs"].items():
        print("subgraph", main, "n", subgraph.get("n_recommended"), subgraph.get("note", ""))


if __name__ == "__main__":
    main()
