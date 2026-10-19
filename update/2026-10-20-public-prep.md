# Public-repository preparation

> **中文说明**：公开前检查了已跟踪文件中的密钥，并在 README 中说明了代理文档和论文联系方式。

## Secret review

Tracked text has no keys, tokens, passwords, private keys, or credential files. History has none of those patterns either. `127.0.0.1` is the local bind address. Scientific tables were not changed.

> **中文说明**：没有发现密钥，科学数据表也没有改动。

## README and ignore rules

`README.md` says `PROJECT_CONTEXT.md` and `update/` are maintained by a coding agent, and that the code paper is still in preparation. Readers can write to fumx@bao.ac.cn. `.gitignore` now also ignores env files, private-key names, notebook checkpoints, and `datasets/jwst_10k/`. `web_sessions/`, `runs/`, `*.pt`, and `*.npy` were already ignored. The tracked tree does not contain `web_sessions`, the JWST image zip, or the full catalogs. `datasets/fama_base/features.h5`, `datasets/ae/features.h5`, and `subset/images.zip` stay tracked.

> **中文说明**：README 写明了代理维护的 update/ 目录和暂用的联系邮箱，三份演示数据仍在仓库中。
