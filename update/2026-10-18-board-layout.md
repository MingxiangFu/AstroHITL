# Board layout pass

> **中文说明**：这一轮把推荐图、类别卡片、边表和分类结果收成一块更高的三栏工作板。

## Image cells

The catalog index sits in the caption under the square, so a square thumbnail has no side bars. A non-square source can still letterbox inside the square. Large mode is one column whose image width is the recommend area width; the row count follows the area height and is at least 1. Small mode is 3 columns, each a third of that width, with at least 3 rows. Empty cells stay when a page has fewer samples, and the area does not shrink. After a successful drop, the opened slot is filled with the next sample that is not already on the page.

> **中文说明**：编号写在方图下面，小图三列、大图一列，放下之后只补上被腾出的那一格。

## Category cards

A card grows to fit its name and the full recent-2 plus representative-3 strip. The card no longer has a delete control on each preview. Removing a sample stays inside the category, on a waiting image.

> **中文说明**：卡片边框会撑开以放下名称和五张预览，预览上不再有删除按钮。

## Edge table

The control above the table shows or hides it. Clicking 类别对 sorts by the pair label in dictionary order and toggles direction. Clicking 数值 sorts by the active edge number and toggles direction; missing numbers stay last. Long tables are paged. Clicking an edge in the graph still highlights its row and moves to that page.

> **中文说明**：边表可以排序、翻页和隐藏，点击图上的边仍会高亮对应的行。

## Distances and freeze

Numeric cells in the per-round training table are centered, as are the headers. `F` / `f` toggles freeze and is ignored while focus is in an input, textarea, or select, the same rule as `Z`.

> **中文说明**：每轮距离表的数字居中，F 与 Z 一样在输入框里不会触发。

## Classification summary

Before training, the user sets train, val, and test fractions. Each may be 0 except train, which must be greater than 0, and the three always sum to 1. Each class is split with stratified sampling at those fractions. The classifier trains on the train split only. The right column then shows a short summary: the three fractions; and for each split, n, raw accuracy, and mapped accuracy, plus raw ROC AUC and mapped ROC AUC when those values exist. An empty split shows n=0. A ranking result shows n, raw pairwise accuracy, and mapped pairwise accuracy.

> **中文说明**：训练、验证、测试比例之和为 1，右侧只显示各份的样本数和准确率。

## Inference samples

The bottom score-bin strip is gone, so the three columns can use the height. After classification, 显示结果 fills the left recommend area with samples that belong to a graph category placed in the task and are not themselves the labeled training images. The caption is the catalog index followed by the predicted score.

> **中文说明**：底部分数箱已去掉；显示结果会把未参与训练的任务图像放回左侧，并在编号后写上预测分数。

## Verification

`conda run -n base python -m pytest` passed: 74 tests, conda `base`, on 2026-10-18. The web server on `127.0.0.1:5002` was not restarted.

> **中文说明**：74 个测试通过，网页服务没有重新启动。
