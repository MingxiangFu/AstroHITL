# UMAP status, demo data, and board defaults

> **中文说明**：UMAP 空白时现在能看出是在计算还是失败了；演示数据说明和右栏默认值也一起改了。

## Blank UMAP

On the trained session `web_sessions/46392e65` (6,994 pool rows, mapper already trained), both UMAPs finish in about 43 seconds and return finite coordinates. Labeled points have category colors. The panes were empty for that whole wait: the page clears the SVG as soon as UMAP is selected and only draws when the response arrives. The fit has no epoch fraction, so the only signal was the thin indeterminate bar. A failed fit, including `Expected all tensors to be on the same device` when a saved CUDA map was loaded on CPU while the session device stayed CUDA, was written to the top status line. A body that could not be parsed failed the same way. The center plot stayed blank in every case.

The panes now say that UMAP is computing, and they keep any previous points. A generation error is the server message in the panel. A response that cannot be read says that the result could not be read. Before the mapped matrix is built, the mapper is moved onto the session device.

> **中文说明**：空白是因为坐标要等两次 UMAP 都算完才画上，失败也只写在顶栏。

## Demo data

`subset/features.h5` is the FAMA-base 10k subset. Its first catalog ids 26, 55, and 86 match the same rows in `BGS_ALL_FAMA_Astro_base.h5`. The full catalog has 451,400 rows and was not added.

> **中文说明**：FAMA-base 的 1 万行仍是 subset，完整目录没有放进仓库。

## Run instructions

The README tells a new user to create the conda env `hitl-feature-lab` and install `requirements.txt` there, not into `base`. Unpacking `subset/images.zip` is part of the demo. FAMA-base images are `subset/images/{index}.jpg`. A user may substitute another uncompressed feature table and `{index}` images. `umap-learn` is now a requirement. The new env imported `umap`, Flask, h5py, and torch. `base` was not modified. The web server was not restarted.

> **中文说明**：说明改为新建 conda 环境，并写明哪套图片对应哪份特征。

## Defaults

The right column opens on 特征训练. The edge table opens on 平均距离. Mapped features for classification and ranking stay disabled until the mapper has been trained. Raw stays available. The parameter count still follows the input width of the choice that is actually selected.

> **中文说明**：右栏默认是特征训练，边表默认是平均距离，没训练映射之前不能选映射特征。

## Verification

`conda run -n base python -m pytest` passed: 91 tests, conda `base`, on 2026-10-19. The web server on `127.0.0.1:5002` was not restarted.

> **中文说明**：91 个测试通过，网页服务没有重新启动。
