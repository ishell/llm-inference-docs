---
source: https://vllm.ai/blog/2025-01-10-dev-experience
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# 安装与开发：nightly 按 commit 钉死，uv 比 pip 快两个数量级（他们那台机）

英文对照：[en/vllm/blog/architecture/dev-experience.md](../../../../en/vllm/blog/architecture/dev-experience.md)  
原文：https://vllm.ai/blog/2025-01-10-dev-experience  
当时公开最新 v0.6.6.post1。数字是演示。

`pip install vllm` / `uv pip install vllm`。nightly：`--extra-index-url https://wheels.vllm.ai/nightly`（pip 要 `--pre`）。Python 改代码：`VLLM_USE_PRECOMPILED=1 pip install -e .` 不编 CUDA。钉 commit：uv 把 extra-index 当更高优先级；pip 会和 PyPI 混成「最新」，所以要用完整 wheel URL。他们 8th-gen CPU 缓存命中：pip ~75s，uv ~0.38s（约 **200×**）。生产靠 commit hash bisect，不是只跟 PyPI 标签。
