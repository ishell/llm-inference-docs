---
source: https://vllm.ai/blog/2026-04-22-fp8-kvcache
lang: en
fetched: 2026-08-31
---

# The State of FP8 KV-Cache and Attention Quantization

2026-04-22. `--kv-cache-dtype fp8` quantizes KV **and** QK/ScoreV (e4m3). Accuracy numbers use **uncalibrated per-tensor scale=1.0** (lower bound).

```bash
vllm serve meta-llama/Llama-3.1-8B --kv-cache-dtype fp8
vllm serve gpt-oss-20b --kv-cache-dtype fp8 --kv-cache-dtype-skip-layers sliding_window
```


Local figures (copyright remains with the original site; study copies):

![fig1 niah before after plot](../../../../assets/vllm/blog/performance/fp8-kvcache/01-fig1_niah_before_after_plot.png)

![fig2 llama 8b](../../../../assets/vllm/blog/performance/fp8-kvcache/02-fig2_llama_8b.png)

![fig3 gptoss 20b](../../../../assets/vllm/blog/performance/fp8-kvcache/03-fig3_gptoss_20b.png)

![fig4 gemma](../../../../assets/vllm/blog/performance/fp8-kvcache/04-fig4_gemma.png)

![fig5 llama b200](../../../../assets/vllm/blog/performance/fp8-kvcache/05-fig5_llama_b200.png)

![fig6 gptoss b200](../../../../assets/vllm/blog/performance/fp8-kvcache/06-fig6_gptoss_b200.png)

![fig7 Qwen3 30B A3B Thinking 2507 reasoning combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/07-fig7_Qwen3-30B-A3B-Thinking-2507_reasoning_combined_plot.png)

![fig8 Qwen3.5 27B reasoning combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/08-fig8_Qwen3.5-27B_reasoning_combined_plot.png)

![fig9 Llama 3.3 70B Instruct openai mrcr 2 needles combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/09-fig9_Llama-3.3-70B-Instruct_openai_mrcr_2_needles_combined_plot.png)

![fig10 Qwen3 30B A3B Instruct 2507 openai mrcr 2 needles combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/10-fig10_Qwen3-30B-A3B-Instruct-2507_openai_mrcr_2_needles_combined_plot.png)

![fig11 Qwen3.5 27B openai mrcr 4 needles combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/11-fig11_Qwen3.5-27B_openai_mrcr_4_needles_combined_plot.png)

![fig12 Qwen3 30B A3B Instruct 2507 openai mrcr 2 needles combined B200 plot](../../../../assets/vllm/blog/performance/fp8-kvcache/12-fig12_Qwen3-30B-A3B-Instruct-2507_openai_mrcr_2_needles_combined_B200_plot.png)

![fig13 Qwen3 30B A3B Thinking 2507 reasoning combined B200 plot](../../../../assets/vllm/blog/performance/fp8-kvcache/13-fig13_Qwen3-30B-A3B-Thinking-2507_reasoning_combined_B200_plot.png)

![fig14 Kimi K2.5 openai mrcr 4 needles H200 plot](../../../../assets/vllm/blog/performance/fp8-kvcache/14-fig14_Kimi-K2.5_openai_mrcr_4_needles_H200_plot.png)

## Fixes

Hopper FA3 long-context accumulation: 128k NIAH **91% → 13%**. Two-level FP32 accumulation → **89%**. Register pressure hurts prefill; `head_dim=256` still slower than BF16 on TTFT.

Hybrid models: small sliding-window layers do not amortize FP8. Skip them (`--kv-cache-dtype-skip-layers sliding_window`). Also: per-head scales, fused query quant, decode tiles.

## ITL slope (concurrency 1, H100)

`ITL = slope × input_len + intercept`. Break-even = context where FP8 ITL < BF16.

Llama-3.1-8B: slope **54%** of BF16, break-even ~**7k**. gpt-oss-20b skip-SW: **71%** / ~7.7k (was 96% / 741k before fixes).

Load (c=8, ~20k/2k): Llama output tok/s **+14.9%**; gpt-oss skip-SW **+4.8%**.

`head_dim=256`: decode wins (68% slope), TTFT quadratic ~**1.6×**. B200 FlashInfer: no two-level accum; Llama break-even ~4k.

## Accuracy

Reasoning: ~1–2 points. MRCR AUC ~97–98% (Llama 70B), ~94–98% (Qwen3 MoE @256k), full AUC at 1M (Qwen3.5-27B).

**Calibrate** if you see a *systematic* drop (Kimi-K2.5 + FlashMLA), not noisy buckets.

## Skip FP8 KV if

contexts <~7k; `head_dim=256` and TTFT matters; uncalibrated accuracy <95% on your set; many tiny sliding-window layers (skip those layers instead).
