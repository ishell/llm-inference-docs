---
source: https://vllm.ai/blog/2026-06-04-nemotron-3-ultra-vllm
lang: en
fetched: 2026-09-01
---

# Nemotron 3 Ultra: 550B/55B hybrid MoE; training rollouts also vLLM

Chinese: `../../zh/vllm/blog/serving/nemotron-3-ultra.md`  
v0.22.0 image. 8×B200 example. Cookbook is the real recipe.

Hybrid Transformer-Mamba MoE, 1M context. BF16: 8×GB200/B200/GB300/B300 or 16×H100 / 8×H200. NVFP4: 4×Blackwell or 8×H100. `VLLM_USE_FLASHINFER_MOE_FP4=1`. Their 8×B200 NVFP4: TP8, `--kv-cache-dtype fp8`, MTP `num_speculative_tokens 5`, `--mamba-backend triton` `--mamba-ssm-cache-dtype float32`, `--reasoning-parser nemotron_v3`, `--tool-call-parser qwen3_coder`. NeMo RL / Gym uses vLLM for multi-node rollout. Marketing numbers (30% cost, leading TPS) live on the original figures — not a signed SLA.

Local figures (copyright remains with the original site; study copies):

![hero](../../../../assets/vllm/blog/serving/nemotron-3-ultra/01-hero.png)

![figure1](../../../../assets/vllm/blog/serving/nemotron-3-ultra/02-figure1.svg)

![figure2](../../../../assets/vllm/blog/serving/nemotron-3-ultra/03-figure2.svg)

![figure3](../../../../assets/vllm/blog/serving/nemotron-3-ultra/04-figure3.svg)
