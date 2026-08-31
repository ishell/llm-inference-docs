---
source: https://developer.nvidia.com/blog/measuring-generative-ai-model-performance-using-nvidia-genai-perf-and-an-openai-compatible-api/
lang: en
fetched: 2026-08-31
---

# Measuring Generative AI Model Performance Using NVIDIA GenAI-Perf and an OpenAI-Compatible API

**Status:** GenAI-Perf is **phased out**. Use **AIPerf** (`../tools/aiperf.md`) for new work. The OpenAI-compatible client idea is unchanged.

GenAI-Perf was Triton's generative-AI client benchmark. It measures LLM-specific metrics, can use OpenOrca / CNN_dailymail, and talks to any OpenAI-compatible API so NIM, Triton, TensorRT-LLM, and vLLM can be compared with the same client.

Supported endpoints in that post: **Chat**, **Chat Completions**, **Embeddings**.

Typical flow: run the Triton SDK container → start a server (example: `vllm/vllm-openai`) → `genai-perf -m … --service-kind openai --endpoint-type chat|completions|embeddings`. Results print as a table and land in `/artifacts` as CSV/JSON plus plots.

Sweep `--request-rate` to see ITL, request latency, and throughput move. Tool is open source (now succeeded by AIPerf).
