# Fast predicted labels for the search pool

> **中文说明**：每次特征映射训练之后，给当前搜索池里还没标完的样本补一个类别猜测，先不改变推荐。

## When it runs

Predictions are rebuilt at the end of mapper training, including a label-file import that trains the mapper. If that training does not produce a mapper, the list is cleared and the feature file is not read again. Resetting the mapper clears the list as well. Assigning an image does not refresh the guesses until the next training.

> **中文说明**：没有训练好的映射时不预测，也不会去扫整份 451400 行特征。

## Rule

Cosine kNN uses the L2-normalized mapped features already loaded for the session pool. k is 15. A sample with no assignment gets a predicted root category from neighbors that already have a root. A sample assigned to a parent, but not to one of that parent's children, gets a predicted child from neighbors under the same parent that already have a child. The vote is a majority, and a tie follows the nearest neighbor. Recommend still searches the same pool and does not filter on these labels.

> **中文说明**：用映射后的余弦近邻投票，k 为 15；未标注的猜根类，已有父类但没有子类的猜子类。

## Storage

Each session keeps `pool_predictions` in its `session.json`. A record is the catalog `index`, the predicted `category_id`, and `level` (`root` or `child`). The next training replaces the whole list.

> **中文说明**：预测写在会话的 session.json 里，下次训练会整表重算。

## Verification

`conda run -n base python -m pytest` passed: 64 tests, conda `base`, on 2026-10-16. The synthetic test checks that k=15 can keep a nearer root even when a larger far group would win a full scan, and that a parent-level sample is sent to the nearer child. The session test trains one epoch on four rows, reloads the stored records, checks that a later recommend still returns the unlabeled index, and checks that reset clears the list. The web server on `127.0.0.1:5002` was not restarted.

> **中文说明**：合成特征测试和会话往返测试已通过，网页服务没有重新启动。
