# Automatic-test contact sheets show galaxy thumbnails

> **中文说明**：自动测试的高低分样例图以前只写编号和分数，现在会画出对应的星系缩略图。

## Cause

`contact_sheet.png` always painted a blue tile with the catalog index and the score. It never called `resolve_image_path`, and `configs/subset_auto.yaml` had no image directory. This is not the earlier media 404: Flask `send_file` joined a relative path onto `hitl_features/web`, but the contact sheet is written by matplotlib from the automatic test and does not go through that route. A freshly unpacked `subset/images/13621.jpg` was on disk and still did not appear.

> **中文说明**：原因是样例图根本没有去打开 JPEG，不是网页把相对路径拼错目录。

## Fix

The sheet looks up `{catalog_index}.jpg` in `img_dir`. A relative directory is resolved from the process working directory, then the file is opened by its absolute path. The index and score stay in the panel title. A missing file still uses the placeholder. `scripts/run_auto_test.py` resolves `img_dir` from the repository root, the same way it resolves the feature file. `configs/subset_auto.yaml` points at `subset/images`. `configs/mergers_dev.yaml` uses `<path-to-images>` for a thumbnail tree that is not in this repository. A one-column sheet keeps both the high-score and low-score rows.

> **中文说明**：配置里的图片目录会按仓库根目录转成绝对路径，再按原编号读取 JPEG。

## Verification

The regression test unpacks `13621.jpg` from `subset/images.zip` into `subset/images` and checks that the sheet shares quantized colors with that JPEG, more than a placeholder sheet does. `conda run -n base python -m pytest` passed: 62 tests, conda `base`, on 2026-10-16. The web server on `127.0.0.1:5002` was not restarted.

> **中文说明**：回归测试确认图里有真实图片像素；全部测试通过，网页服务没有重新启动。
