# Category names, feature source, and UMAP

> **中文说明**：同级类别不能重名，分类和排序可选原始或映射特征，中间面板可以看两张 UMAP。

## Names before this change

`create_category` and `rename` accepted any non-empty name and did not look at siblings. Two categories under the same parent could share a display name, and a child could share its parent's name. Identity was the category id. `path()` returned category ids, not a stored name path, and `|` was not reserved. CSV import was the exception: `merge_hierarchy` and `merge_labels` reused an existing non-recycle sibling with the same name and assigned the sample to it. The recycle bin is a root category named Recycle, and import skips it, so a label `Recycle` created a separate user category.

> **中文说明**：改之前手建或改名不查同级重名，只有 CSV 导入会把样本并进已有的同名类别。

## Sibling names

A new name that matches a sibling under the same parent is rejected, including a rename. The recycle bin is left out of that check, so a user category may still be named Recycle. Incremental CSV import of an existing name still adds samples to that sibling. It does not create a second category and it is not a rejection. The same string may exist under different parents, including a child with the same name as its parent.

> **中文说明**：同一父类下的新名字不能和兄弟重复；导入已有名字仍是追加样本，不是拒绝。

## Stored paths

The backend path is the ancestor names joined by `|`, for example `barred|barred`. `|` is reserved and rejected in a category name and in a task-class name typed by the user. The page refuses `|` before it posts, and the server rejects it as well. Paths do not use `\` or `/`.

> **中文说明**：后台路径用 `|` 连接各级名称，用户输入里的 `|` 会被拒绝。

## Training features

Classification and ranking train one model. The choice is raw features or mapped features, and the default is mapped. The control sits beside the model-size menu. Each size option shows the parameter count for the selected input width: mapped uses the session output dimension, raw uses the catalog feature dimension. A hidden pair that would pass 11,180,103 parameters on the new width falls back to 64×32. The training request sends the selected source, and inference and the score board use that same space. Only `mapped.pt` or `raw.pt` is saved. The auto and expert four-way arms still train both spaces on their own.

> **中文说明**：分类和排序默认用映射特征，模型大小旁的参数量跟着所选输入维度变。

## UMAP

The center panel switches from the category graph to two UMAPs, raw and mapped, side by side. Both draw the session candidate pool in high-transparency gray, then draw labeled samples on top in their category color. Recycled samples stay in the gray pool only. The fit has no epoch fraction, so the graph column shows the existing indeterminate bar and the same starter controls stay disabled. Coordinates are cached until the pool or the map changes; a later request only recolors.

> **中文说明**：中间可以换成并排的原始和映射 UMAP，灰底是候选池，彩色点是已标注样本。

## Verification

`conda run -n base python -m pytest` passed: 89 tests, conda `base`, on 2026-10-19. The web server on `127.0.0.1:5002` was not restarted.

> **中文说明**：89 个测试通过，网页服务没有重新启动。
