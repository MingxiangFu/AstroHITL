# Scope labels, category borders, and UMAP hits

> **中文说明**：最明确推荐的标题改为本次排序所用的类别和分数，类别卡片用类别色描边，UMAP 已标注点可以查看图像。

## Certain captions

最明确 inside one class ranks by mean cosine distance to that class alone. The old caption was the pool kNN, which votes among every labeled root with k=15. That vote is refreshed after the round, so it is not a stale cache. It answers a different question: a sample closest to A can still have a majority of neighbors from another, larger class. The recommend column now shows the scope class and the score used to order the row. A higher score is placed first. With two or more classes, the caption is the winning class of the same probabilities that rank the row.

> **中文说明**：标题不是过期缓存，而是另一套全池投票；现在显示的是本次范围和排序分数。

## Board and UMAP

Category cards use that category's color for the border. The delete control on a classification class is a visible button. Labeled UMAP points sit above the gray pool. Hovering one shows its image and category and hides them when the pointer leaves. Clicking one keeps the card until another labeled point or empty space is clicked. Classification results and recommendation rows are ordered by probability or score, highest first.

> **中文说明**：分类类的删除按钮可见；UMAP 悬停显示图像和类别，点击后保持到另一次点击。

## Verification

`conda run -n base python -m pytest` passed with 92 tests and 2 skips, conda `base`, on 2026-10-20. The skips are the full-catalog checks. The web server was not restarted.

> **中文说明**：92 个测试通过，2 个完整目录检查被跳过，网页服务没有重新启动。
