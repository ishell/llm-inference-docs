---
source: https://vllm.ai/blog/2025-06-30-minimax-m1
lang: en
fetched: 2026-09-01
---

# MiniMax-M1: Lightning Attention + MoE; the Docker sample still pinned V0

Chinese: `../../zh/vllm/blog/serving/minimax-m1.md`  
456B total, ~45.9B active. M3: [minimax-m3](minimax-m3.md).

They quote ~**25%** FLOPs vs DeepSeek R1 at 100k generated tokens. `--quantization experts_int8`. The Docker snippet sets `VLLM_USE_V1=0` — **historical**; hybrid allocator later landed in V1. Lightning Attention via Triton. PagedAttention: they claim fragmentation <4% vs traditional 60–80%. Architecture + then-deploy, not M3 MSA.

Local figures (copyright remains with the original site; study copies):

![benchmark](../../../../assets/vllm/blog/serving/minimax-m1/01-benchmark.png)

![moe](../../../../assets/vllm/blog/serving/minimax-m1/02-moe.png)

![lightning attention](../../../../assets/vllm/blog/serving/minimax-m1/03-lightning_attention.png)
