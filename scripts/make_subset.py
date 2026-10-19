"""Build a portable 10k catalog subset. Feature rows are stored uncompressed."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hitl_features.features import catalog_indices, load_feature_rows
from hitl_features.subset import SUBSET_SEED, SUBSET_SIZE, choose_subset_indices, write_feature_table, write_stored_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the portable 10k subset")
    parser.add_argument(
        "--features",
        default="<path-to-features>",
    )
    parser.add_argument(
        "--labels",
        default="<path-to-labels>",
    )
    parser.add_argument(
        "--images",
        default="<path-to-images>",
    )
    parser.add_argument("--output", default=str(ROOT / "subset"))
    parser.add_argument("--size", type=int, default=SUBSET_SIZE)
    parser.add_argument("--seed", type=int, default=SUBSET_SEED)
    args = parser.parse_args()
    required = _label_indexes(args.labels)
    catalog = catalog_indices(args.features)
    chosen = choose_subset_indices(catalog, required, size=args.size, seed=args.seed)
    print(f"subset {len(chosen)} rows, required labels kept {len(set(required) & set(chosen))}, seed {args.seed}")
    matrix = load_feature_rows(args.features, chosen)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_feature_table(output / "features.h5", matrix, chosen)
    image_root = Path(args.images)
    members = []
    missing = []
    for index in chosen:
        source = image_root / f"{index}.jpg"
        if not source.is_file():
            missing.append(index)
        else:
            members.append((f"{index}.jpg", source))
    if missing:
        raise SystemExit(f"missing {len(missing)} images, first {missing[:5]}")
    write_stored_zip(output / "images.zip", members)
    shutil.copyfile(args.labels, output / "expert_testset.csv")
    for name in ("features.h5", "images.zip", "expert_testset.csv"):
        path = output / name
        print(f"{name} {path.stat().st_size} bytes")


def _label_indexes(path: str) -> list[int]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return [int(float(row["index"])) for row in reader if str(row.get("index") or "").strip()]


if __name__ == "__main__":
    main()
