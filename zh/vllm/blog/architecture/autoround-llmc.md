---
source: https://vllm.ai/blog/2025-12-09-intel-autoround-llmc
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# AutoRound × LLM Compressor：W4A16 进 compressed-tensors，vLLM 直接 serve

英文对照：[en/vllm/blog/architecture/autoround-llmc.md](../../../../en/vllm/blog/architecture/autoround-llmc.md)  
原文：https://vllm.ai/blog/2025-12-09-intel-autoround-llmc  
Omni 侧见 [omni-autoround](../serving/omni-autoround.md)。

每量化张量三个可训量：rounding 偏移 `V`，clip 的 `α`/`β`。按 decoder 层顺序，signed GD 最小化块重建误差。轻量：常 **128** 条校准、约 **200** iters，不是上千。当时 `AutoRoundModifier` 出 W4A16，Llama/Qwen dense。推理路径零额外开销。XPU 上当时还要 `--enforce-eager`，并跟 vLLM PR。GSM8K 5-shot 他们报 Qwen3-8B W4A16 **0.911**（演示，非确定性）。后续才是 FP8/MXFP4/NVFP4、混 bit、MoE。量化在工作站、serve 在 Arc B60 可以不是同一台机器。
