# Subset thumbnails served from the working directory

> **中文说明**：另一台机器解压子集后，推荐能成功，但缩略图仍按网页包目录去找，所以返回找不到。

## Cause

Session create and recommend open `subset/features.h5` from the directory where the server was started, so those requests succeed after `unzip subset/images.zip -d subset/images`. `resolve_image_path` then finds `subset/images/{catalog_index}.jpg` as a relative path. Flask `send_file` joins that relative path onto the package directory `hitl_features/web`, so the file is not the jpeg that was unpacked. Catalog indexes `13621` and `63830` are in the archive under those flat names. A missing file was logged as 404 on the other machine.

> **中文说明**：特征文件按启动目录打开，图片路径却被拼到 `hitl_features/web` 下面。

## Fix

`/media/<session>/<catalog_index>` resolves the thumbnail against the process working directory before `send_file`. An index that is not in the unpacked directory still returns 404. The route does not read `images.zip` itself; the README unpack step is unchanged.

> **中文说明**：返回图片前先把相对路径变成启动目录下的绝对路径，压缩包本身不在请求时打开。

## Verification

A regression test unpacks `13621.jpg` and `63830.jpg` from `subset/images.zip` into a fresh `subset/images`, starts a session with relative `subset/features.h5` and `subset/images`, and checks that both media responses are JPEG (`ff d8`). `conda run -n base python -m pytest` passed: 60 tests, conda `base`, on 2026-10-15. The web server on `127.0.0.1:5002` was not restarted.

> **中文说明**：回归测试确认这两个编号返回 JPEG，全部测试通过，网页服务没有重新启动。
