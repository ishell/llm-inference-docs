---
source: https://vllm.ai/blog/2025-12-09-intel-autoround-llmc
lang: en
fetched: 2026-09-01
---

# AutoRound × LLM Compressor: W4A16 into compressed-tensors, serve in vLLM

Chinese: `../../zh/vllm/blog/architecture/autoround-llmc.md`  
Omni sibling: [omni-autoround](../serving/omni-autoround.md).

Three trainables per quantized tensor: rounding offset `V`, clip `α`/`β`. Sequential decoder blocks, signed GD on reconstruction. Light: often **128** calib samples, ~**200** iters, not thousands. Then `AutoRoundModifier` emitted W4A16 for Llama/Qwen dense. Zero extra inference overhead. XPU still needed `--enforce-eager` and a vLLM PR. GSM8K 5-shot Qwen3-8B W4A16 **0.911** (demo, nondeterministic). Later: FP8/MXFP4/NVFP4, mixed-bit, MoE. Quantize on a workstation, serve on Arc B60 — not the same box.
