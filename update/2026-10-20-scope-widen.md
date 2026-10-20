# Recommend when the scope vote is empty

> **中文说明**：推荐范围的 kNN 没有分到样本时，不再返回空集，而是按范围内分数取前若干倍再推荐。

## Slice

When a recommend scope is set and kNN assigns no candidate-pool sample to those classes, each pool sample is ranked by the highest score among the scoped classes. Another class may still score higher for that sample. The selected recommender then runs on the top `n` times the recommend count. `n` defaults to 10, and the recommend row can change it. If kNN does find in-scope samples, the candidate set stays as before.

> **中文说明**：多个范围类取其中的最高分；倍数默认 10，找到范围内样本时仍走原来的候选集。

## Caption

The row shows the scoped class and the score used to rank it. It does not show the global kNN label.

> **中文说明**：行上显示的是范围类和用于排序的分数，不是全局 kNN 标签。
