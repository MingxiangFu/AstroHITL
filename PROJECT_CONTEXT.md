# HITL custom features — project context

> **中文摘要**：UMAP 计算和失败都会显示在中间面板；右栏默认是特征训练，映射特征要等映射训练后才能选。

## Goal

> **中文说明**：先做传统分类，再让人定义类别，最后用人标注的类别图训练特征映射。

Traditional classification, then human-in-the-loop custom classification, then human-in-the-loop custom features. v1 trains one contrastive map from a cumulative exclusive category graph and compares raw versus mapped features.

## Layout

> **中文说明**：代码在 hitl_features，网页在 web，两份一万行特征在 datasets，图像在 subset。

- `hitl_features/graph.py` — categories, nesting, exclusive assignment, sibling strengths in [0, 2]. A new name that matches a sibling is rejected. Import of that name still adds samples. The same string may sit under another parent, including a child named like its parent. The stored path joins those names with `|`. `|` is rejected in a typed name.
- `hitl_features/loss.py`, `mapper.py`, `train.py`, `modelsizes.py` — L2-normalized MLP and strength-weighted supervised contrastive loss. Temperature is 0.1. Training continues the same module until the map width changes, which discards that module. The classifier and ranker share a width menu capped at 11,180,103 parameters.
- `hitl_features/recommend.py` — random, diversity, unknown, certain, uncertain, and the algorithmic default. With two or more classes, certain and uncertain rank cosine kNN probabilities (k=15): highest max class probability, or highest normalized entropy then a lower max probability. With one class every probability is 1 and entropy is 0, so certain uses the smallest mean cosine distance to that class and uncertain the largest. An empty set means every category at the current level. Inside a category the pool is only samples already in that category with no subclass. Recycle-bin samples are not references.
- `hitl_features/pool_predict.py` — after mapper training, cosine kNN (k=15) on the current pool writes `pool_predictions` into `session.json`. An unlabeled sample gets a root category. A sample on a parent with children gets a child category. No mapper means no prediction and no extra catalog read. Certainty recommend refreshes this list when a mapper exists, and ranks with the live probabilities rather than the stored hard label.
- `hitl_features/preview.py`, `score_board.py`, `figures.py` — category previews (2 recent, 3 representative), 5 equal-width score bins on a seeded 40-image sample, and auto-test ROC, PR, and contact-sheet figures. The contact sheet draws `{catalog_index}.jpg` from `img_dir` when that file exists. A relative image directory is resolved from the process working directory.
- `hitl_features/expert_test.py` — main graph from `main_label_expert`, subgraphs from `secondary_label_expert`, and pairwise agreement with `mapped_{class}_combined_score`.
- `hitl_features/four_way.py`, `auto_test.py` — binary recommended-versus-random comparison. Kept, not the main cell. A finished run also writes `roc.png`, `pr.png`, and `contact_sheet.png`.
- `hitl_features/labels.py`, `subset.py` — a chosen index column and one or more label columns, in level order, merge into the open graph and train. A name that already exists under that parent is reused. A file that already uses `index, label_1, label_2, ...` still imports directly. Export writes each placed sample with its ancestors and skips the recycle bin. The portable subset keeps original catalog indexes.
- `hitl_features/web/app.py` — local Flask UI on `127.0.0.1:5002`, board layout only. Left: 推荐, 当前类别, and 回收站 on one row, then the recommender and page buttons. The image index sits under the square. Small mode is 3 columns and at least 3 rows; large mode is 1 column and at least 1 row. Center: category cards, or raw and mapped UMAPs of the candidate pool (gray) with labeled samples in category color. While a UMAP fit runs, the panes say so; a generation or read failure is written in the panel. Right opens on 特征训练. The edge table opens on 平均距离. 分类/排序 keeps raw features available and leaves mapped features disabled until the map has been trained. The head-size count follows the selected input width. Recommend, train, import, export, and UMAP lock those starters and show a progress bar or an indeterminate bar on the matching column. CSV import still asks for columns, then merges into the open graph. `F` toggles freeze with the same focus rule as `Z`. Thumbnail paths are resolved before `send_file`.
- `datasets/fama_base/features.h5` — FAMA-base 10k, no compression filter; rows match `BGS_ALL_FAMA_Astro_base.h5`. This is the web form's default feature path.
- `datasets/ae/features.h5` — AE 10k on the same `index` values and the same images. See `datasets/README.md`.
- `subset/` — shared images: uncompressed `images.zip` (unpack to `subset/images/{index}.jpg`) and `expert_testset.csv`. All 3,960 expert indexes are included in the 10k tables; seed 0 fills the rest. The full 451,400-row catalog is not stored here.
- `configs/expert_hierarchy.yaml` — hierarchy cell. `feature_path` is `<path-to-features>`; labels are `subset/expert_testset.csv`.
- `configs/mergers_dev.yaml` — older single-class cell. Paths outside this repo are `<path-to-features>`, `<path-to-labels>`, and `<path-to-images>`.

Full catalogs stay outside this repository. This repo stores code, the 10k demo, and small run outputs. `runs/` and `web_sessions/` are gitignored. Checkpoints use `torch.save`. Feature matrices saved for a session use uncompressed `numpy.save`.

## Commands

> **中文说明**：演示请新建 conda 环境；测试这次仍用 base。

```bash
conda run -n base python -m pytest
conda run -n base python scripts/run_web.py
conda run -n base python scripts/run_expert_test.py --config configs/expert_hierarchy.yaml --output-dir runs/expert_hierarchy_smoke
```

Tests passed: 89 passed and 2 skipped, conda `base`, on 2026-10-20. The two skips are the full catalog checks, which now look for `<path-to-features>` and `<path-to-images>`. The hierarchy smoke (pool 400, 4 rounds, 64 images, CUDA) wrote `runs/expert_hierarchy_smoke/report.json`. Leaf distance stayed smaller within a category than between categories. Main-class separation did not: child edges use strength 1, so samples that share a main label but not a secondary are pushed apart. Mapped main-class macro AUC was 0.590 versus 0.538 on raw features; accuracy was not higher. Catalog-score agreement moved from 0.517 to 0.538. The web server was not restarted for the shared feature-table move.

## Repository

> **中文说明**：`v1` 已推到 GitHub 上的私有仓库。

Branch `v1`. GitHub repository: `MingxiangFu/hitl_feature_lab`.

Updates: `update/2026-10-15-v1.md`, `update/2026-10-15-expert-hierarchy-test.md`, `update/2026-10-15-drag-drop.md`, `update/2026-10-15-annotation-workflow.md`, `update/2026-10-15-graph-tasks-subset.md`, `update/2026-10-15-subset-media.md`, `update/2026-10-16-contact-sheet.md`, `update/2026-10-16-pool-predictions.md`, `update/2026-10-16-board-ui.md`, `update/2026-10-17-board-refine.md`, `update/2026-10-18-board-layout.md`, `update/2026-10-18-import-recycle.md`, `update/2026-10-19-release-ui.md`, `update/2026-10-19-names-umap.md`, `update/2026-10-19-umap-readme.md`, `update/2026-10-20-shared-feature-tables.md`, `update/2026-10-20-public-prep.md`, `update/2026-10-20-portable-paths.md`.
