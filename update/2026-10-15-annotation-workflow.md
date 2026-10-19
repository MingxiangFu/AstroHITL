# Annotation workflow

> **中文说明**：类别计数和指定推荐算上全部子孙，卡片显示最近图与代表图，一轮没放完也能重新推荐，训练后按分数分箱并在自动测试里保存曲线。

## Descendant counts and recommendation

A category card counts every sample assigned to that category or to any descendant. Direct assignment is unchanged: `samples()` and waiting still list only the category itself. Specified-category recommendation rewrites every graph label inside that same descendant set to the chosen category, so a kNN neighbor anywhere under it counts as a hit.

> **中文说明**：父类显示的数量和“指定该类”的近邻，都包含它下面所有子类里的样本。

## Category preview strip

Each category card shows the n most recently added images and the m representative images, with a vertical bar between the two groups. Both groups are drawn from the category plus all descendants. Recent images are the latest assignments by the recorded order, with a higher sample index breaking a tie. Representatives use mapped features only after the contrastive mapper has been trained, and raw features otherwise. Mean cosine similarity is computed inside the class. Members already shown as recent are left out when other members remain. The rest are ranked by that mean similarity, a high-density pool of about `max(3m, 12)` is kept, and greedy farthest-first picks m images that are far from each other. If the class is smaller than n+m, the empty side is omitted.

> **中文说明**：卡片左侧是最近放入的图，右侧是类内又密又彼此拉开的代表图，父类没有直接样本时也会显示子类的图。

## Recommend again before the round is finished

Recommend stays available while some left-hand images are still unplaced. The new batch replaces those unplaced images. Only indexes already assigned in the graph are excluded. Unplaced leftovers are not treated as in the graph, so they can return in a later batch. Confirm still waits until the current left-hand images are placed.

> **中文说明**：没放完也可以换机制再要一批，未放下的图会被换掉，已放进图里的不会再被推荐。

## Score bins and automatic-test figures

After classification or ranking training, the frozen mapped head scores a fixed sample of the bound pool. The sample is seeded, size 40, and sorted, so the same task refresh keeps the same indexes. Ranking uses the scalar score. Classification uses the probability of the predicted class and shows that class name on the thumbnail. The scores are split into 5 equal-width bins, with up to 4 thumbnails in each bin. The automatic test still writes the numeric four-way summary, and also saves `roc.png`, `pr.png`, and `contact_sheet.png`. The two curves put the four arms (recommended/random × raw/mapped) on shared axes. The contact sheet shows high-score and low-score examples from the mapped recommended arm when that arm has scores. PNG pixels are stored uncompressed.

> **中文说明**：分类或排序训练结束后，固定的 40 张图按分数分成 5 档；自动测试另存 ROC、PR 和高低分样例。

## Defaults

Shipped defaults are recent n=2, representative m=3, 5 score bins, 4 thumbnails per bin, and a fixed sample of 40 images (or the whole pool when it is smaller).

> **中文说明**：默认最近 2 张、代表 3 张、5 个分数箱、每箱最多 4 张、固定抽样 40 张。

## Verification

`conda run -n base python -m pytest` — 50 passed, including subtree counts, specified kNN, preview selection, mid-round recommend, score bins, and the saved curves. No new medium or full-catalog GPU run was started. The annotation server on `127.0.0.1:5002` was stopped and was not started again.

> **中文说明**：50 项测试通过，5002 端口上的标注服务已停止且没有重新启动。
