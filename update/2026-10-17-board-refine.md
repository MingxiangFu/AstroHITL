# Board refinements

> **中文说明**：工作板补上可见的类别标签、一对多的任务类、3×3 小图、边的高亮，以及中英文切换。

## Magnifier

The magnifier stays off until the toolbar button or `Z` / `z`. The key is ignored while focus is in an input, textarea, or select. Hover magnifies the left-strip images, the score-bin images, and the representative images inside graph category cards.

> **中文说明**：放大镜默认关闭，快捷键改为 Z，类别卡片里的代表图也能放大。

## Recommend column

上一页 and 下一页 stay on one line with the mechanism select, at the same height, and the buttons shrink instead of wrapping. 大图 / 小图 sits in the recommend column. 小图 is a 3×3 grid; empty cells remain when the page has fewer than nine images, so the column height does not change with the image count or the size toggle.

> **中文说明**：翻页按钮和推荐方式同一行，小图固定为九宫格。

## Wells and task classes

A category dropped into the recommend well is a visible tag: name, sample count, color, and ×. Classification and ranking use empty task classes created in the right column. The same frozen drag drops a graph category into one of those classes. One class can hold several graph categories. A drop does not create a task class by itself.

> **中文说明**：先新建空的任务类，再把图上的类别拖进去，一个任务类可以包含多个图类别。

## Edges and language

Clicking a graph edge highlights that edge’s row under the graph. Clicking it again clears the highlight, and clicking another edge moves it. One button switches the page chrome between Chinese and English and stores the choice in the browser. Category names, prediction names, saved task names, score values, and server error text stay as stored data.

> **中文说明**：点击连线会高亮图下的数值行；页面文案可以在中英文之间切换并记住选择。

## Verification

`conda run -n base python -m pytest` passed: 71 tests, conda `base`, on 2026-10-17. The web server on `127.0.0.1:5002` was not restarted.

> **中文说明**：71 个测试通过，网页服务没有重新启动。
