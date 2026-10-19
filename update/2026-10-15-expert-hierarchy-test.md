# Expert hierarchy auto test

> **中文说明**：自动测试改为用专家主次标签建层级图，并已跑完一轮小规模全流程。前端代码没有改。

## Design

`expert_testset.csv` labels every row with `main_label_expert` and `secondary_label_expert`. The script builds one graph:

- A top-level category for each main label that the recommender actually places.
- A child under that category for the sample's secondary label.
- One continued feature map for the whole graph.

`mergers` and `pair_group` appear only as secondary labels, so they are tested inside subgraphs, not as top-level classes.

After training, the report has three checks:

1. **Main classification.** Four arms on `main_label_expert`: recommended or same-size random samples, raw or mapped features. Metrics are accuracy and macro one-vs-rest ROC AUC.
2. **Subgraphs.** Every main label that landed at least two secondary labels is classified the same way, using `secondary_label_expert` and only the samples in that branch.
3. **Catalog probabilities.** For each class, pairwise agreement between the classifier probability and `mapped_{class}_combined_score`. This is separate from the hard-label accuracy.

The file `expert_testset_locked.csv` is still refused. The full `expert_testset.csv` is the file requested for this test, and it includes rows the older scaling split had held out.

## Simple run

Command, conda `base`, CUDA:

```bash
conda run -n base python scripts/run_expert_test.py \
  --config configs/expert_hierarchy.yaml \
  --output-dir runs/expert_hierarchy_smoke \
  --device cuda
```

Pool 400, 4 rounds, 16 images per round, 8 mapper epochs, 20 classifier epochs. Output: `runs/expert_hierarchy_smoke/report.json`. 64 images were placed. None of the catalog indices fell outside the feature file. Eval set: 1,188 rows.

Leaf cosine distance after round 4: within 0.054, between 0.174. Main-label distance went the other way: within 0.237, between 0.172. Child edges default to strength 1, so two samples that share a main label and differ in secondary are repelled. That spreads the parent class.

Main-class holdout, recommended samples: raw accuracy 0.331 and macro AUC 0.538; mapped accuracy 0.306 and macro AUC 0.590. Random-sample macro AUC was 0.530 raw and 0.558 mapped. Catalog-score agreement was 0.517 raw and 0.538 mapped.

Subgraphs were fit for smooth (17), unbarred spirals (15), irregular (13), edge-on with bulge (9), barred spirals (7), and poor quality (2). The poor-quality branch has one image per secondary and is not interpretable. No branch shows a stable mapped-over-raw gain at this size.

Unit tests: 38 passed.

## Not changed

The Flask UI was left as it is while that interface is being tried by hand.
