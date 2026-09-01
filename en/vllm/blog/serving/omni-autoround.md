---
source: https://vllm.ai/blog/2026-06-02-vllm-omni-autoround
lang: en
fetched: 2026-09-01
---

# Omni × AutoRound: W4A16, quantize once, serve

Chinese: `../../zh/vllm/blog/serving/omni-autoround.md`  
LLM Compressor sibling is a separate post.

Reads `quantization_config.quant_method = "auto-round"` — no extra `--quantization`. Qwen3-Omni-30B-A3B: 66 GB → 25 GB (~**62%**). OmniBench W4A16 slightly above BF16; TIIF drift ~**1.3%**. Intel B60: FLUX BF16 needs TP4; W4A16 fits one card; leftover GPUs run CFG Parallel, guided gen ~**1.55–1.67×**. Wan2.2 / GLM-Image / FLUX live; BAGEL / Ovis had checkpoints, runtime still landing. Quantize offline; the hot path only infers.
