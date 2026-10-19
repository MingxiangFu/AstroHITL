# Release UI

> **中文说明**：这一轮整理发布界面：两行模式、导入并入当前图、右侧训练区、进行中的进度，以及带参数量的模型宽度。

## Left column

The left toolbar is two rows. The first row chooses 推荐, 当前类别, or 回收站, and the active mode stays highlighted. 回收站 lists the samples in the recycle bin. The second row is the recommender, the recommend button, and the page buttons. In 回收站 that recommender row stays disabled.

> **中文说明**：左侧第一行切换推荐、当前类别和回收站，第二行才是推荐控件。

## Import

CSV import still asks for the index column and the label columns before it runs. The rows are added to the open graph. A label that already exists under the same parent receives the sample. Otherwise the name becomes a new category. Categories that are not in the file stay, including the recycle bin.

> **中文说明**：导入会并入当前类别图，同级同名则归入该类，否则新建。

## Right column

The edge table and the feature-training record sit in the right column, so showing or hiding the table does not move the graph. That column switches between 特征训练 and 分类/排序, with the same highlighted mode control as the left column.

> **中文说明**：边表和映射训练记录改到右栏，开关边表不再挪动中间的图。

## Busy state

Recommend, map training, classification, ranking, import, and export disable the controls that would start another of those jobs. Recommend shows an indeterminate bar on the left column. Training shows a bar on the right column, filled by epoch. Import closes the column dialog first, then fills a bar on the graph from rows and then epochs. Export uses an indeterminate bar.

> **中文说明**：推荐、训练、导入、导出进行时会锁住重复操作，并用进度条或不定动画标出对应区域。

## Model sizes

The feature map offers hidden widths 64, 256, 512, 1024, and 2048. 256 is the previous small map. The classifier and ranker share hidden pairs 64×32, 256×128, 512×256, 1024×512, 2048×1024, and 3072×1536, plus the largest pair on a step of 64 that stays at or under 11,180,103 parameters. Each option shows the exact weight count for the open session. On the portable subset (768-d features, map width 32, two classes) the map counts are 51,296, 205,088, 410,144, 820,256, and 1,640,480. The raw classifier counts are 51,362, 230,018, 525,570, 1,313,282, 3,675,138, 7,085,570, and 10,929,858 (3,968×1,984). Changing the map width drops the old module and requires a new training run.

> **中文说明**：映射和分类/排序都能选宽度，选项上是当前会话的准确参数量；更换映射宽度后必须重新训练。

## Verification

`conda run -n base python -m pytest` passed: 83 tests, conda `base`, on 2026-10-19. The web server on `127.0.0.1:5002` was not restarted.

> **中文说明**：83 个测试通过，网页服务没有重新启动。
