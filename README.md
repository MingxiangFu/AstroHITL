# HITL Feature Lab

> **中文说明**：本工具在本地浏览器中依据人工标注的类别图训练特征映射，并在原始特征与映射特征上训练分类器或排序器。

HITL Feature Lab is a local web application for labeling images on a category graph and training one feature map from those labels. The same board trains a classifier or a ranker on the original features and on the mapped features.

## Run

> **中文说明**：使用新建的 conda 环境安装依赖，将图像包解压到 subset/images，再以 FAMA-base 特征表和该图像目录启动本机标注服务。

Create a new conda environment. Do not install these packages into `base`.

```bash
conda create -n hitl-feature-lab python=3.10 -y
conda run -n hitl-feature-lab python -m pip install -r requirements.txt
```

Unpacking an image archive is part of the demo. From the repository root:

```bash
mkdir -p subset/images
unzip subset/images.zip -d subset/images
```

`subset/images/{index}.jpg` pairs with both feature tables. `datasets/fama_base/features.h5` is the FAMA-base 10k subset: 10,000 uncompressed `mae_features` rows taken from `BGS_ALL_FAMA_Astro_base.h5`, with the same catalog ids. `datasets/ae/features.h5` uses those same ids. Neither file is the full 451,400-row catalog.

A user may substitute their own files in the same format: an uncompressed HDF5 feature table (`mae_features`, `features`, or `embeddings`, plus `index`) and an image directory of `{index}.jpg` or `{index}.png`.

Start the board from the repository root:

```bash
conda run -n hitl-feature-lab python scripts/run_web.py
```

Open `http://127.0.0.1:5002`. The server listens on that address only. The form defaults are `datasets/fama_base/features.h5` and `subset/images`. Both paths are resolved from the directory where the server was started.

Each session, its graph, and its feature map are stored under `web_sessions/`.

## Tutorial

> **中文说明**：标注板的分步说明在 docs/tutorial.md，配图在 docs/tutorial_imgs。

`docs/tutorial.md` is the board tutorial, with figures in `docs/tutorial_imgs/`. It covers setup, named projects, the category graph, recommenders, the feature map, classification and ranking, and UMAP.

## Data

> **中文说明**：FAMA-base 与 AE 两份一万行特征表共用同一组星表索引及 subset 中的同一套图像，CSV 导入则把标注合并进当前已打开的类别图。

| Catalog | Features | Images |
| --- | --- | --- |
| FAMA-base 10k | `datasets/fama_base/features.h5` | `subset/images/{index}.jpg`, unpacked from `subset/images.zip` |
| AE 10k | `datasets/ae/features.h5` | `subset/images/{index}.jpg`, unpacked from `subset/images.zip` |

`datasets/fama_base/features.h5` and `datasets/ae/features.h5` each store `mae_features` (10,000 × 768, float32) and `index` with no HDF5 compression filter. Their `index` values are the same catalog ids in the same order, so both tables use the images in `subset/images.zip` (ZIP_STORED). The FAMA-base rows match the same ids in the FAMA-base catalog. These files are not the full 451,400-row catalogs. CSV import asks for the index column and the label columns, then adds those labels to the graph that is already open. A label name that already exists at that level receives the sample. A new name becomes a new category.

## Notes

> **中文说明**：PROJECT_CONTEXT.md 与 update/ 目录由编程代理维护，不是手写文档。

`PROJECT_CONTEXT.md` and the `update/` directory are maintained by a coding agent. They are not written by hand. The directory name is `update/`.

## Paper

> **中文说明**：配套的代码论文仍在准备中，读者目前可通过 fumx@bao.ac.cn 联系作者。

The code paper is still in preparation. For now, readers can contact the author at fumx@bao.ac.cn.

## License

> **中文说明**：本仓库以 MIT 许可证发布，2026 年的版权持有人为 MingxiangFu。

This repository is released under the MIT License. The copyright holder is MingxiangFu, 2026. See `LICENSE`.
