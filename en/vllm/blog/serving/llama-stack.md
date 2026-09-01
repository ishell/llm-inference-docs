---
source: https://vllm.ai/blog/2025-01-27-intro-to-llama-stack-with-vllm
lang: en
fetched: 2026-09-01
---

# Llama Stack × vLLM: inference is a swappable Provider, not a second engine

Chinese: `../../zh/vllm/blog/serving/llama-stack.md`  
Red Hat + Meta. Demo: Llama-3.2-1B CPU container.

Two: `remote::vllm` (OpenAI-compatible `/v1`) and inline (same process as Stack). Safety, agents, vectors stay other Stack providers. K8s: vLLM Service DNS `vllm-server.default.svc…:8000/v1`; Stack only fills the URL. Tutorial is 2025-01 `llama stack build` YAML — APIs drift. The point is one app API across the lifecycle, engine underneath swappable.
