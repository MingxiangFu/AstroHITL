# Portable paths and MIT license

> **中文说明**：跟踪文件中的本机路径已改为仓库内路径或占位符，并加入了 MIT 许可证。

## Paths

Feature and image inputs that are not in this repository now use `<path-to-features>`, `<path-to-labels>`, or `<path-to-images>`. The demo feature table remains `datasets/fama_base/features.h5`. Demo images remain `subset/images`, unpacked from `subset/images.zip`. Hierarchy labels that match the shipped table use `subset/expert_testset.csv`. The GitHub repository identity remains `MingxiangFu/hitl_feature_lab`.

> **中文说明**：仓库里没有的外部数据改用占位符，演示用的特征和图像路径保持不变。

## License

`LICENSE` is the MIT License, copyright holder MingxiangFu, year 2026. `README.md` has a License section.

> **中文说明**：许可证为 MIT，版权持有人是 MingxiangFu，年份是 2026。

## Verification

`conda run -n base python -m pytest` passed with 89 tests and 2 skips, conda `base`, on 2026-10-20. The skips are the full-catalog checks. The web server was not restarted.

> **中文说明**：89 个测试通过，2 个完整目录检查被跳过，网页服务没有重新启动。
