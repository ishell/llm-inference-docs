---
source: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/perf-analyzer-README.html
lang: zh
fetched: 2026-08-31
---

# Triton Performance Analyzer

测 Triton 上传统模型性能的 CLI。**LLM 请用 AIPerf / GenAI-Perf**；这是非生成式那条线。

**负载：** concurrency / request-rate / 自定义间隔  
**测量：** 时间窗直到稳态，或按请求数窗口。

支持 sequence / ensemble / decoupled。输入可自动生成。

流程：Triton 容器 → simple 模型仓库 → `tritonserver` → SDK 容器 → `perf_analyzer -m simple`。远程加 `-u host:8000`。
