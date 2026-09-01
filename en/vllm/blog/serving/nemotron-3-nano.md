---
source: https://vllm.ai/blog/2025-12-15-run-nvidia-nemotron-3-nano
lang: en
fetched: 2026-09-01
---

# Nemotron 3 Nano: 30B/3B hybrid MoE, Thinking Budget, NVFP4 later

Chinese: `../../zh/vllm/blog/serving/nemotron-3-nano.md`  
1M context. 2026-01-28 addendum: NVFP4 + QAD, B200 vs FP8-H100 they quote **4×**.

vs Nano 2: FFN→sparse MoE, most attention→Mamba-2. They quote up to ~**4×** token throughput. Then `reasoning-parser deepseek_r1` (later Nemotron 3 often `nemotron_v3`). `VLLM_ATTENTION_BACKEND=FLASHINFER`. Day-0 how-to, not kernel depth. Larger: [nemotron-3-super](nemotron-3-super.md) / [nemotron-3-ultra](nemotron-3-ultra.md).
