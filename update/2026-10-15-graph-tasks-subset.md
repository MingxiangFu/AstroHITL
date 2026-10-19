# Graph layout, tasks, labels, and a portable subset

> **中文说明**：类别卡片可以拖开，同一张图上能保留多个任务，也可以用标签文件建图，并附带一份 1 万行子集。

## Card positions

A category card can be dragged to any point on the graph. The position is a fraction of the canvas and is stored with the graph, so a reload draws the card in the same place. Relation lines use those centers and move with the card while it is dragged. The category name is still the handle that drops onto a task class.

> **中文说明**：拖动卡片会记住位置，连线跟着卡片走。

## Several tasks

Each classification or ranking run is stored as its own task, with its own score bins. A new task starts from an empty class list, so a category used in an earlier task can be used again. After training, the center column shows that task. Selecting an older task shows its bins again. Images in a score bin can be dragged onto the graph. If the image is already annotated, it keeps an orange border and stays in the result view. A recommended image on the left still leaves the left list once it is dropped on the graph.

> **中文说明**：分类和排序都可以多次训练；分数图里的图片能拖进类别图，已在图中的会留在结果里并换成橙色边框。

## Label file

The user uploads a CSV, then chooses the index column and the label column. Rows whose catalog index is in the feature file become one root category per label. Indexes outside the file are skipped. The mapper is trained on that graph, and the session stays editable.

> **中文说明**：选定索引列和标签列后，程序按标签建图并训练映射，之后仍可继续标注。

## Remove a sample

Remove drops the sample's category assignment and puts it on the root waiting list. It does not offer a shortcut that moves the sample straight into another category.

> **中文说明**：移出样本只是取消标注，并把它放回根上的待标注列表。

## Portable subset

`subset/features.h5` has 10,000 rows and no HDF5 compression filter (30,802,048 bytes). All 3,960 indexes in `expert_testset.csv` are included. Seed 0 fills the other 6,040 from the catalog. The original catalog index is stored in the `index` dataset and is not renumbered. Thumbnails in `subset/images.zip` (34,483,644 bytes, uncompressed) are named `{catalog_index}.jpg`. The label file is `subset/expert_testset.csv` (6,243,417 bytes). Both large files are under GitHub's 100MB limit, so they are ordinary git blobs, not Git LFS, and they were not split.

> **中文说明**：子集保留了专家表里的全部原编号，再用种子 0 补到 1 万行；特征和图片都没有压缩。

## Other machine

Unpack with `unzip subset/images.zip -d subset/images`. Install from `requirements.txt` if needed. Point the web form at `subset/features.h5` and `subset/images`. Run `scripts/run_web.py` and `scripts/run_auto_test.py --config configs/subset_auto.yaml`. See `README.md`.

> **中文说明**：另一台机器解压图片后，用子集特征和图片目录打开网页，并按配置跑自动测试。

## Verification

`conda run -n base python -m pytest` passed on conda `base`. No new medium or full-catalog GPU run was started. Nothing is listening on `127.0.0.1:5002`, and the web server was not started again.

> **中文说明**：测试已通过，5002 端口空着，网页服务没有重新启动。
