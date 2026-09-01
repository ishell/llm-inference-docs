---
source: https://vllm.ai/blog/2026-03-11-nemotron-3-super
lang: en
fetched: 2026-09-01
---

# Nemotron 3 Super: 120B/12B, 1M context, Thinking Budget

Chinese: `../../zh/vllm/blog/serving/nemotron-3-super.md`  
v0.17.1. 4×H100 BF16 example. Spark numbers: [dgx-spark](dgx-spark.md).

Hybrid MoE + Mamba, MTP, Latent MoE (4 experts at ~1 expert cost). BF16/FP8/NVFP4. They claim Blackwell NVFP4 ~**4×** vs H100 FP8 at matched accuracy — cookbook, not a generation plate. `--kv-cache-dtype fp8` `--reasoning-parser nemotron_v3` `--tool-call-parser qwen3_coder`. vs prior Super: up to ~**5×** throughput, ~**2×** accuracy in their charts. Marketing figures on the original page.
