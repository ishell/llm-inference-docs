---
source: https://developer.nvidia.com/blog/measuring-generative-ai-model-performance-using-nvidia-genai-perf-and-an-openai-compatible-api/
lang: zh
fetched: 2026-08-31
---

# 用 GenAI-Perf 打 OpenAI 兼容 API（导读）

**现状：** GenAI-Perf **已停更**。新项目用 **AIPerf**（`../tools/aiperf.md`）。「客户端打 OpenAI 兼容接口」这个思路没变。

当时是 Triton 的生成式客户端基准：测 LLM 专用指标，可用 OpenOrca / CNN_dailymail，凡是 OpenAI 兼容的服务（NIM、Triton、TRT-LLM、vLLM）都能用同一套客户端对比。

文中支持的端点：**Chat**、**Chat Completions**、**Embeddings**。

流程：Triton SDK 容器 → 起服务（例：`vllm/vllm-openai`）→ `genai-perf -m … --service-kind openai --endpoint-type chat|completions|embeddings`。结果打表，并写到 `/artifacts` 的 CSV/JSON 和图。

扫 `--request-rate` 看 ITL、request latency、吞吐怎么变。
