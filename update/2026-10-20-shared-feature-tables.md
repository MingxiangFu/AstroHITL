# Shared 10k feature tables

> **中文说明**：两份一万行特征表现在按名称分开存放，并共用同一组星表索引和同一套图像。

## Tables

`subset/features.h5` moved to `datasets/fama_base/features.h5`. `datasets/ae/features.h5` was copied from the AE 10k table. Both files are 30,802,048 bytes, with uncompressed `mae_features` (10,000 × 768, float32) and `index`. The `index` arrays are equal; the first ids are 26, 55, and 86. Images remain `subset/images.zip`, unpacked to `subset/images/{index}.jpg`. The full 451,400-row catalogs were not added. No file in this change exceeds GitHub's 100MB limit.

> **中文说明**：FAMA-base 表是从 subset 移过去的，AE 表的索引与它完全相同。

## Defaults

The web form's default feature path is `datasets/fama_base/features.h5`. The image directory default stays `subset/images`. `configs/subset_auto.yaml` uses the new feature path. The shipped-subset test checks both tables, their shared indexes, and that `subset/features.h5` is gone.

> **中文说明**：网页默认打开 FAMA-base 特征表，图像目录仍是 subset/images。

## Verification

`conda run -n base python -m pytest` passed: 91 tests, conda `base`, on 2026-10-20. The web server on `127.0.0.1:5002` was not restarted.

> **中文说明**：91 个测试通过，网页服务没有重新启动。
