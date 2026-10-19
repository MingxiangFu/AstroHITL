# Shared 10k feature tables

> **中文说明**：仓库中的两份一万行特征表共用同一组星表索引，并使用 subset 中的同一套图像。

`datasets/fama_base/features.h5` is the FAMA-base 10k table. `datasets/ae/features.h5` is the AE 10k table. Both store uncompressed `mae_features` (10,000 × 768, float32) and `index`. The `index` values are the same catalog ids, in the same order.

Images stay in `subset/images.zip`. Unpack that archive to `subset/images`. A row's image is `subset/images/{index}.jpg` for either feature file. These tables are not the full 451,400-row catalogs.
