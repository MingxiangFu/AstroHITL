# Import, recycle bin, and single-class recommend

> **中文说明**：这一轮放开映射训练的空轮限制，并补上回收站、单类推荐和可选列的 CSV 导入。

## Mapper training

Confirm trains the feature map while the left recommend list still has unplaced images. The button is enabled once at least one sample sits in a real category.

> **中文说明**：推荐列表里还有没放下的图时，也可以训练特征映射。

## Recycle bin

Every graph starts with a recycle bin. It has no edges and is not a recommend target or a task class. A control on the left switches from recommendations to every labeled sample in the current view, and each of those samples can move into the bin. Bin samples are left out of mapper training, classifiers, ranking, kNN reference pools, and distance stats.

> **中文说明**：回收站自动存在、没有连线，里面的样本不参与训练和推荐。

## Single-class recommenders

With one class in scope, cosine kNN assigns probability 1 to that class and normalized entropy 0, so 最明确 and 最不明确 previously shared one catalog-index order. That tie is the probability math. In that case 最明确 now ranks by the smallest mean cosine distance to the labeled members, and 最不明确 by the largest. Two or more classes still use max probability and normalized entropy.

> **中文说明**：只有一个类别时，最明确和最不明确改按到该类样本的平均余弦距离排序。

## Inference count and CSV import

The right column has an input for how many inference samples 显示结果 places in the left column. CSV import no longer posts the file immediately. The user picks the file, then chooses the index column and one or more label columns in level order, and import plus training start only after that confirmation.

> **中文说明**：结果张数可以指定；CSV 要先选定索引列和标签列，确认后才导入。

## Verification

`conda run -n base python -m pytest` passed: 78 tests, conda `base`, on 2026-10-18. The web server on `127.0.0.1:5002` was not restarted.

> **中文说明**：78 个测试通过，网页服务没有重新启动。
