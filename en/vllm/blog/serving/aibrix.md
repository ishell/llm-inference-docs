---
source: https://vllm.ai/blog/2025-02-21-aibrix-release
lang: en
fetched: 2026-08-31
---

# Introducing AIBrix

2025-02-21. https://github.com/vllm-project/aibrix — ByteDance Kubernetes control plane, in production there since ~2024. Co-design engine + system.

Launch list: dense LoRA, LLM gateway/routing, app-tailored autoscaler, unified runtime sidecar, distributed inference, distributed KV, heterogeneous GPUs with SLO, GPU failure detection.

**vs production-stack (their FAQ):** AIBrix = large-scale / cloud-native, already production 6+ months. production-stack = LMCache/UChicago, from-scratch building blocks, KV-centric (transfer, blending, routing) for long-context / prefill-heavy; plans to reuse AIBrix parts. **vs KServe/KubeAI:** more vLLM-native (load, scale, LoRA).
