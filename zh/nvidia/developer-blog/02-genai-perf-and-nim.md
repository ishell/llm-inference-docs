---
source: https://developer.nvidia.com/blog/llm-performance-benchmarking-measuring-nvidia-nim-performance-with-genai-perf/
lang: zh
fetched: 2026-08-30
---

# 系列第 2 篇：用 GenAI-Perf 测 NIM（中文摘译）

原文：https://developer.nvidia.com/blog/llm-performance-benchmarking-measuring-nvidia-nim-performance-with-genai-perf/  
英文笔记：`en/nvidia/developer-blog/02-genai-perf-and-nim.md`

**注意：** GenAI-Perf 已停更，新项目请用 **AIPerf**。操作流程几乎一样，NIM 手册第 4 章已是 AIPerf 版，优先看：

`zh/nvidia/nim-benchmarking/04-quickstart.md`

本篇在系列里的位置：第 1 篇讲指标，本篇把 Llama 3.1 8B Instruct 用 NIM 拉起来，用 GenAI-Perf 扫场景。

## 要点

- NIM 是预打包容器，后端可以是 TensorRT-LLM 或 vLLM。NIM 官网上的性能数字就是用 GenAI-Perf 打的。
- 工具和 NIM **同机**跑，除非你要把网络算进延迟。
- 扫的场景示例：Translation 200/200、Classification 200/5、Summary 1000/200、Codegen 200/1000；并发 1, 2, 5, 10, 50, 100, 250。
- `--measurement-interval 30000`（毫秒）是每个测量窗口。大模型高并发要加大，例如 100000 ms。
- 结果在 `artifacts/`，主文件 `*_genai_perf.csv`。画 TTFT–RPS 曲线，读延迟预算或目标并发。
- 示例图里 `concurrency=50` 开始延迟陡增、吞吐几乎不涨。
- LoRA：`-m adapter1 adapter2 adapter3 --model-selection-strategy random|round_robin`。

部署 NIM 需要 NGC API key，把模型缓存目录 mount 进容器。细节命令见英文笔记或官方 NIM 文档。
