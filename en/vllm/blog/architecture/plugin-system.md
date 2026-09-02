---
source: https://vllm.ai/blog/2025-11-20-vllm-plugin-system
lang: en
fetched: 2026-09-01
---

# vLLM plugin system

2025-11-20. Study note. Two-week releases + hundreds of PRs/week make a long-lived fork a full-time job. Monkey patches replace whole classes for a ten-line change and break on upgrade.

Fourth path: vanilla vLLM + your package via Python entry points. Proprietary / experimental / off-review-cycle work stays in the plugin. Hardware-specific door: [hardware-plugin.md](hardware-plugin.md). AFD already uses `vllm.general_plugins`.

Fits: custom scheduler, KV behavior, hardware, execution. Does not fit: rewriting the engine heart while tracking main weekly. Cousins: Sleep Mode admin APIs, KVConnector, WeightTransferEngine — leave a door, do not fork.

Local figures (copyright remains with the original site; study copies):

![vllm plugin system arch](../../../../assets/vllm/blog/architecture/plugin-system/01-vllm-plugin-system-arch.png)
