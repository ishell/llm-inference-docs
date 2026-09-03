---
source: https://vllm.ai/blog/2025-04-05-llama4
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Llama 4：Scout 16E / Maverick 128E，iRoPE 1:3，v0.8.3+

英文对照：[en/vllm/blog/serving/llama4.md](../../../../en/vllm/blog/serving/llama4.md)  
原文：https://vllm.ai/blog/2025-04-05-llama4  
每 token 只激活 **1** expert（17B active）。

8×H100：Scout `--max-model-len 1000000`（他们建议 `attn_temperature_tuning: true`）；Maverick-FP8 约 **430K**。8×H200：Scout 3.6M，Maverick 1M。多图：`--limit-mm-per-prompt image=10`（默认 1）。`--kv-cache-dtype fp8` 可把窗口再翻倍量级，他们说评测几乎不掉。Scout 10M 要多机 TP/PP。iRoPE：无 RoPE 全局 attention 与分块局部 RoPE 1:3。Maverick MMLU-Pro 官方 80.5，H100 FP8 **80.4**。`VLLM_DISABLE_COMPILE_CACHE=1` 是当时的开工旗。

本地图（原文版权仍归原站；学习对照用）：

![perf](../../../../assets/vllm/blog/serving/llama4/01-perf.png)
