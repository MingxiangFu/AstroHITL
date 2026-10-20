# Board results, subclass recommend, and UMAP cache

> **中文说明**：标注板可以投影分类和排序结果，子类推荐改为在全池上预测，UMAP 的原始坐标只计算一次。

## Recommend scope

A subclass in the recommend well is predicted over the full candidate pool, including samples that are not already labeled with the parent. The recommender then runs on that predicted set. If the mapper is not trained, or that subclass has no labeled examples, the request reports that instead of using a wider category. 当前类别 keeps the older screen and searches only samples already labeled in the open category. That action leaves the left column in 当前类别. The checkbox there defaults to samples with no subclass.

> **中文说明**：子类推荐在整个候选池上预测；当前类别仍只在该类已标注样本中筛选，默认不显示已有子类的样本。

## Results

Feature training lists every category tag, and those tags can be dragged into the other wells. A classification class can be renamed or deleted. 运行预测 sits beside 显示分类结果 and uses the same count. It scores the active classifier or ranker on the full pool, bins by score, and samples into the left panel. A saved ranking result can be shown the same way as a classification result. The active result button stays highlighted with its name visible, and each result can be renamed or deleted. A multiclass image shows the predicted class and its score, with the class color on the border. Recommended samples use the predicted category color on that border.

> **中文说明**：运行预测与显示分类结果共用张数；多分类结果显示类别、分数和类别色边。

## UMAP

Points are drawn smaller. Raw-feature coordinates are fit the first time UMAP opens for the current pool. Later opens only recolor labeled points. Mapped coordinates are fit again when the mapper changes. Replacing the candidate pool refits both.

> **中文说明**：原始特征的 UMAP 坐标按候选池只算一次，映射坐标在映射模型变化时重算。

## Verification

`conda run -n base python -m pytest` passed with 91 tests and 2 skips, conda `base`, on 2026-10-20. The skips are the full-catalog checks. The web server was not restarted.

> **中文说明**：91 个测试通过，2 个完整目录检查被跳过，网页服务没有重新启动。
