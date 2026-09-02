---
source: https://vllm.ai/blog/2025-02-21-aibrix-release
lang: en
fetched: 2026-08-31
---

# Introducing AIBrix

2025-02-21. https://github.com/vllm-project/aibrix — ByteDance Kubernetes control plane, started early 2024, already on multiple internal businesses. Co-design: engine + system, cloud-native on K8s, not “a Deployment that happens to run vLLM.” Quotes from Clayton Coleman and Robert Nishihara are on the page.

Launch blocks: high-density LoRA; LLM gateway/routing; app-tailored autoscaler; unified runtime sidecar (metrics, model pull); distributed inference; distributed KV; mixed GPUs under SLO; GPU failure detection.

Roadmap already named P/D aggregation, request migration, cross-instance KV reuse; request-level QoS / priority / fairness; roofline profiling for SLO. Later posts (Router, Mooncake, Elastic EP) pick up pieces of that list.

**vs production-stack (FAQ):** AIBrix = large-scale / cloud-native, production 6+ months. production-stack = LMCache/UChicago, from-scratch blocks, KV-centric (transfer, blending, routing) for long-context / prefill-heavy; planned to reuse AIBrix parts. **vs KServe/KubeAI:** more vLLM-native (fast load, scale, LoRA) because it only has to serve one engine.

Control planes can be swapped. KV affinity does not go away — that is the [router](router.md) post.

Local figures (copyright remains with the original site; study copies):

![aibrix diagram](../../../../assets/vllm/blog/serving/aibrix/01-aibrix-diagram.png)
