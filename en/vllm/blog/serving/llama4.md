---
source: https://vllm.ai/blog/2025-04-05-llama4
lang: en
fetched: 2026-09-01
---

# Llama 4: Scout 16E / Maverick 128E, iRoPE 1:3, v0.8.3+

Chinese: [zh/vllm/blog/serving/llama4.md](../../../../zh/vllm/blog/serving/llama4.md)  
**One** expert per token (17B active).

8×H100: Scout `--max-model-len 1000000` (they suggest `attn_temperature_tuning: true`); Maverick-FP8 ~**430K**. 8×H200: Scout 3.6M, Maverick 1M. Multi-image: `--limit-mm-per-prompt image=10` (default 1). `--kv-cache-dtype fp8` can roughly double the window; they saw little eval drop. Scout 10M needs multi-node TP/PP. iRoPE: global no-RoPE vs chunked local RoPE at 1:3. Maverick MMLU-Pro reported 80.5, H100 FP8 **80.4**. `VLLM_DISABLE_COMPILE_CACHE=1` was a then-launch flag.

Local figures (copyright remains with the original site; study copies):

![perf](../../../../assets/vllm/blog/serving/llama4/01-perf.png)
