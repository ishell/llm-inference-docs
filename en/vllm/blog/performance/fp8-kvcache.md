---
source: https://vllm.ai/blog/2026-04-22-fp8-kvcache
lang: en
fetched: 2026-09-04
---

# The State of FP8 KV-Cache and Attention Quantization

Chinese: [zh/vllm/blog/performance/fp8-kvcache.md](../../../../zh/vllm/blog/performance/fp8-kvcache.md)

2026-04-22. `--kv-cache-dtype fp8` quantizes the KV cache **and** runs the whole attention computation (QK and ScoreV matmuls) in FP8. The format used throughout the post is **e4m3**. Accuracy numbers are **uncalibrated per-tensor scale=1.0** — the simplest knob, and a lower bound. Calibration and per-head scales only get better.

At 128k+ contexts the KV cache often dominates GPU memory for full-attention decoders, and every Decode step rereads a large fraction of it. Halving that storage can mean more concurrency or a longer window on the same card — if accuracy holds.

```bash
# FP8 KV-cache for all layers
vllm serve meta-llama/Llama-3.1-8B --kv-cache-dtype fp8

# Hybrid-attention models: skip sliding-window layers (recommended)
vllm serve gpt-oss-20b --kv-cache-dtype fp8 --kv-cache-dtype-skip-layers sliding_window
```

They validated decoder-only and MoE models on Hopper (FA3, vLLM's flash-attention fork) and Blackwell (FlashInfer). On the paths in the post, Decode cost and KV memory drop while accuracy stays near baseline. Main caveats: hybrid models with **small** sliding-window layers (skip those layers), and `head_dim=256` (Prefill can still regress). For `head_dim` 64 and 128, FP8 can speed **both** Prefill and Decode. Best-case memory-bound Decode: per-token KV cost down to **54%** of BF16.

Local figures (copyright remains with the original site; study copies):

![fig1 niah before after plot](../../../../assets/vllm/blog/performance/fp8-kvcache/01-fig1_niah_before_after_plot.png)

Figure 1: 128k needle-in-a-haystack on Hopper before/after the FA3 FP8 fixes. Two-level accumulation restores long-context accuracy near BF16; the optimized FP8 path still keeps the Decode-speed win.

## The problems they found

`--kv-cache-dtype fp8` had been in vLLM for a while. Stress tests turned up two classes of bug.

**Accuracy.** Hopper FA3 FP8 lost accumulation precision at long context. On 128k NIAH, accuracy fell from **91%** (BF16) to **13%**. Cause: Tensor Core “FP32” accumulation is imprecise when the contraction dimension is large (same hardware issue as DeepSeek-V3 training, Technical Report Figure 7(b)). In Decode/Prefill that contraction dim is context length: `Softmax(AttnScore) * V`.

**Performance.** On hybrid models with sliding-window layers (gpt-oss-20b), the FP8 ITL slope was **96%** of BF16 — almost no Decode speedup despite half the memory. Break-even was **741,565** tokens, past any practical window.

## Kernel and vLLM improvements

**Two-level accumulation.** Write partial sums into a *real* FP32 register ([SageAttention2](https://arxiv.org/abs/2411.10958); [flash-attention#104](https://github.com/vllm-project/flash-attention/pull/104)). NIAH comes back to **89%**. Cost: register pressure, Prefill slowdown. Tiling configs ([flash-attention#125](https://github.com/vllm-project/flash-attention/pull/125)) recover some of it for `head_dim` 64/128; **above 128, Prefill still trails BF16**.

**Skipping layers.** `--kv-cache-dtype-skip-layers` ([vllm#33695](https://github.com/vllm-project/vllm/pull/33695)) lets a hybrid numeric format exist. GPT-OSS sliding-window layers attend a fixed window (e.g. 128 tokens); FP8 overhead never amortizes. Keep those layers BF16. Same flag can skip quantization-sensitive layers.

**Per-head scales.** FA3 already takes an array of scales, one per KV head. Wiring it needed general static-quant group shapes ([vllm#30833](https://github.com/vllm-project/vllm/pull/30833)) and `reshape_and_cache_flash` taking an array instead of one scalar ([vllm#30141](https://github.com/vllm-project/vllm/pull/30141)).

**Query quantization fusion.** Move query quant out of the attention backend into a torch op that `torch.compile` can fuse ([vllm#24914](https://github.com/vllm-project/vllm/pull/24914)). Kills a fixed per-token overhead.

**FA3 FP8 tile sizes.** Prefill tiles for `head_dim=64` and `128` to cut register spills from two-level accum ([flash-attention#125](https://github.com/vllm-project/flash-attention/pull/125)). Separate Decode tiles that flatten the ITL slope ([flash-attention#96](https://github.com/vllm-project/flash-attention/pull/96), [#91](https://github.com/vllm-project/flash-attention/pull/91)).

Hopper and Blackwell have 2× FP8 FLOPs vs BF16, so Prefill *should* also win. In practice that is not free, which is the rest of the post.

## Performance: single request (concurrency 1)

ITL grows linearly with input length because every generated token attends the full KV cache. Fit:

`ITL = slope × input_len + intercept`

Prefill is quadratic:

`TTFT = a × input_len² + b × input_len + c`

**Break-even** = context length where FP8 ITL drops below BF16 ITL. Slope is the per-token attention tax.

Setup: 1×H100, FA3 via the vLLM fork (native FP8 KV + online softmax rescaling). `vllm bench serve`, concurrency 1, 128 output tokens, input lengths 256 → 125k.

![fig2 llama 8b](../../../../assets/vllm/blog/performance/fp8-kvcache/02-fig2_llama_8b.png)

Figure 2: Llama-3.1-8B, H100. FP8 nearly halves the Decode ITL slope with almost no intercept penalty; break-even ~7k; TTFT similar.

| | BF16 | FP8 |
|---|---|---|
| ITL slope (ms/token) | `4.37e-05` | `2.37e-05` |
| intercept (ms) | 6.44 | 6.58 |
| slope vs BF16 | 100% | **54%** |
| intercept gap | — | +0.14 ms |
| break-even | — | **~7k** |

Even with two-level accumulation on, long-context TTFT is *slightly* better than BF16.

![fig3 gptoss 20b](../../../../assets/vllm/blog/performance/fp8-kvcache/03-fig3_gptoss_20b.png)

Figure 3: gpt-oss-20b, H100. Hybrid global + sliding-window (window 128). Skip-SW is the winner: those layers have bounded KV, so they pay quant tax without a long-context return.

| | BF16 | FP8 (all layers) | FP8 skip-SW |
|---|---|---|---|
| ITL slope (ms/token) | `8.94e-06` | `7.14e-06` | `6.34e-06` |
| intercept (ms) | ~4.03–4.07 | same cluster | same cluster |
| slope vs BF16 | 100% | 80% | **71%** |
| break-even | — | 22,109 | **7,659** |

Recommendation in the post: use skip-SW for hybrid models.

### ITL slope table (before vs after)

_Table 3 from the post. “Before” is v0.10.2; “after” is v0.19.1._

| Model | Version | FP8 variant | Break-even (tokens) | FP8 slope (% of BF16) |
|---|---|---|---|---|
| Llama-3.1-8B | v0.10.2 | FP8 | 24,889 | 63% |
| Llama-3.1-8B | v0.19.1 | FP8 | **7,010** | **54%** |
| gpt-oss-20b | v0.10.2 | FP8 | 741,565 | 96% |
| gpt-oss-20b | v0.19.1 | FP8 | 22,109 | 80% |
| gpt-oss-20b | v0.19.1 | FP8 skip-SW | **7,659** | **71%** |

Takeaway: Decode-heavy long-context serving on H100 — FP8 is already clearly worth it for Llama-class models, and for hybrid models once small sliding-window layers are skipped.

## Throughput under load

Concurrency 1 isolates per-token attention. Under load: **150 requests, concurrency 8**, ~20k input / ~2k output (±15%).

_Table 4: Llama-3.1-8B. FP8: **+14.9%** output tok/s, **−13.0%** total duration, **−14.8%** median ITL._

| Config | Median TTFT (ms) | Median ITL (ms) | Total duration (s) | Output tok/s |
|---|---|---|---|---|
| BF16 | 763.6 | 15.18 | 672.6 | 450.3 |
| FP8 | 742.8 | 12.93 | 585.2 | **517.5** |

_Table 5: gpt-oss-20b. Skip-SW: **+4.8%** output tok/s, **−4.6%** duration, **−4.8%** median ITL._

| Config | Median TTFT (ms) | Median ITL (ms) | Total duration (s) | Output tok/s |
|---|---|---|---|---|
| BF16 | 468.9 | 8.09 | 364.2 | 831.6 |
| FP8 | 451.7 | 7.90 | 355.1 | 853.0 |
| FP8 skip-SW | 456.4 | 7.70 | 347.4 | **871.8** |

The 54% ITL-slope cut at c=1 becomes +14.9% output throughput at c=8 for Llama — faster tokens **and** 2× KV memory, so the scheduler packs more. gpt-oss gains less because sliding-window layers cap the memory win; skip-SW recovers the most. At higher concurrency or longer context, BF16 OOMs or evicts first; that is when the house-size advantage actually shows.

## `head_dim=256` (Prefill regression)

Two-level accumulation is **on by default** after flash-attention#104, so users do not silently eat a 91%→13% NIAH cliff. At large head dims that default makes TTFT worse than BF16.

![fig4 gemma](../../../../assets/vllm/blog/performance/fp8-kvcache/04-fig4_gemma.png)

Figure 4: gemma-4-E2B on H100, `head_dim=256`. Three of four layers are sliding-window **512** (4× gpt-oss’s 128, so quantizing SW layers *does* amortize). Decode wins; Prefill loses.

| | BF16 | FP8 |
|---|---|---|
| ITL slope (ms/token) | `5.30e-05` | `3.60e-05` (**68%**) |
| TTFT quadratic coeff (ms/token²) | `6.93e-07` | `1.12e-06` (**~1.6×**) |

Two ways out named in the post:

1. **Disable two-level accumulation** — recovers Prefill, including extra Prefill wins at head dim 64/128. **You must accuracy-test the workload.**
2. Accumulate only every N steps: open PR [flash-attention#122](https://github.com/vllm-project/flash-attention/pull/122). Functional; recovers Prefill speedups.

## FlashInfer on Blackwell (B200)

Accumulation is a Hopper FA3 problem. B200 + FlashInfer does not need two-level accum. These B200 runs compare BF16 vs FP8 only — **layer skipping was not yet supported on B200** when they ran.

![fig5 llama b200](../../../../assets/vllm/blog/performance/fp8-kvcache/05-fig5_llama_b200.png)

Figure 5: Llama-3.1-8B, B200 FlashInfer. Slope again ~54% of BF16; intercept almost unchanged; break-even **~4k**.

| | BF16 | FP8 |
|---|---|---|
| ITL slope (ms/token) | `1.80e-05` | `9.72e-06` |
| intercept (ms) | 3.93 | 3.96 |

![fig6 gptoss b200](../../../../assets/vllm/blog/performance/fp8-kvcache/06-fig6_gptoss_b200.png)

Figure 6: gpt-oss-20b, B200 FlashInfer. Slope cut is stronger than H100, but intercept still needs a longer context before it pays.

| | BF16 | FP8 |
|---|---|---|
| ITL slope (ms/token) | `3.56e-06` | `2.06e-06` |
| intercept (ms) | 3.15 | 3.17 |
| break-even | — | **~13k** |

## Accuracy

Models: Llama-3.3-70B-Instruct, Qwen3-30B-A3B-Instruct-2507, Qwen3-30B-A3B-Thinking-2507, Qwen3.5-27B.

- Long-context (Prefill-heavy): `openai/mrcr`, up to 1M. Average pass@1 per length bucket over **5** repetitions; **AUC** across lengths ([Context Arena](https://contextarena.ai/)).
- Reasoning (Decode-heavy): AIME25, GPQA:Diamond, MATH500, LiveCodeBench-v6. pass@1: **10** reps for AIME25 and LiveCodeBench-v6; **5** for GPQA and MATH500.
- Sampling: each model’s default **non-greedy** params.

**All of these numbers use per-tensor uncalibrated scale=1.0.** Reproducible with `--kv-cache-dtype fp8`. Lower bound. They also support calibration via [`vllm-project/LLM-Compressor`](https://github.com/vllm-project/llm-compressor) and per-head scales ([vllm#30141](https://github.com/vllm-project/vllm/pull/30141)); see vLLM’s quantized KV-cache examples.

### Reasoning

![fig7 Qwen3 30B A3B Thinking 2507 reasoning combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/07-fig7_Qwen3-30B-A3B-Thinking-2507_reasoning_combined_plot.png)

Figure 7: Qwen3-30B-A3B-Thinking-2507, BF16-model and FP8-W&A settings. FP8 KV + FP8 attention: **~1–2 points**. Lowest recovery **97%** (GPQA:Diamond, BF16 model). Short Prefill, long Decode (tens of thousands of tokens).

![fig8 Qwen3.5 27B reasoning combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/08-fig8_Qwen3.5-27B_reasoning_combined_plot.png)

Figure 8: Qwen3.5-27B decoder-only. At most **0.7 points**. Lowest recovery **99%** (AIME25, BF16 model).

### Long-context MRCR

![fig9 Llama 3.3 70B Instruct openai mrcr 2 needles combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/09-fig9_Llama-3.3-70B-Instruct_openai_mrcr_2_needles_combined_plot.png)

Figure 9: Llama-3.3-70B-Instruct, 8k → 128k (model max). **97–98%** of baseline AUC@128k in both BF16-model and FP8-W&A settings.

![fig10 Qwen3 30B A3B Instruct 2507 openai mrcr 2 needles combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/10-fig10_Qwen3-30B-A3B-Instruct-2507_openai_mrcr_2_needles_combined_plot.png)

Figure 10: Qwen3-30B-A3B-Instruct-2507 MoE, up to 256k. Longest buckets gap more than Llama. AUC recovery **~94%** (BF16 model) / **~98%** (FP8 model). Bucket variance is partly the baseline (32k > 8k/16k; 128k > 64k).

![fig11 Qwen3.5 27B openai mrcr 4 needles combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/11-fig11_Qwen3.5-27B_openai_mrcr_4_needles_combined_plot.png)

Figure 11: Qwen3.5-27B up to **1M**. Aggregate AUC@1M **matches** baseline in both settings; longest buckets still noisy.

### FlashInfer on B200 (accuracy)

No two-stage accum needed. Same recipes as Hopper: Qwen3-30B-A3B-Instruct-2507 on MRCR; Qwen3-30B-A3B-Thinking-2507 on reasoning.

![fig12 Qwen3 30B A3B Instruct 2507 openai mrcr 2 needles combined B200 plot](../../../../assets/vllm/blog/performance/fp8-kvcache/12-fig12_Qwen3-30B-A3B-Instruct-2507_openai_mrcr_2_needles_combined_B200_plot.png)

Figure 12: MRCR AUC recovery **~93%** (BF16 model) / **~96%** (FP8 model). Competitive, not as tight as the best Hopper/FA3 cases.

![fig13 Qwen3 30B A3B Thinking 2507 reasoning combined B200 plot](../../../../assets/vllm/blog/performance/fp8-kvcache/13-fig13_Qwen3-30B-A3B-Thinking-2507_reasoning_combined_B200_plot.png)

Figure 13: reasoning — average differences **roughly a point or less**.

Post’s conclusion: FP8 KV is ready as the **default starting point** for many long-context, Decode-heavy, memory-bound deployments. Exceptions: Prefill-dominated `head_dim=256`; hybrid models whose small sliding-window layers should stay BF16; backends/models with a persistent uncalibrated drop (calibrate).

### When to calibrate

Not every model likes scale=1.0. Kimi-K2.5 uses **FlashMLA** (not FA3/FlashInfer) on H200.

![fig14 Kimi K2.5 openai mrcr 4 needles H200 plot](../../../../assets/vllm/blog/performance/fp8-kvcache/14-fig14_Kimi-K2.5_openai_mrcr_4_needles_H200_plot.png)

Figure 14: uncalibrated FP8 KV + FP8 attention. Aggregate AUC drop is modest and error bands overlap, but the shift is **systematic across buckets**, not noisy outliers. Start uncalibrated; calibrate when you see that shape on the real workload. Especially relevant for non-FA3/FlashInfer backends.

## When to avoid FP8 KV-cache

Stay on BF16 (or skip the offending layers) if:

- **Contexts ≲ ~7k.** FP8’s intercept gap can make BF16 slightly faster on ITL.
- **`head_dim=256` and TTFT / Prefill matters.** Two-level accum ~**1.6×** TTFT quadratic. Disabling it recovers speed only with careful accuracy checks.
- **Uncalibrated accuracy <95% on your set.** Kimi-K2.5 + FlashMLA is the worked example; calibrate on target data.
- **Many tiny sliding-window layers.** Prefer `--kv-cache-dtype-skip-layers sliding_window` rather than all-layer FP8.

Same tax as TensorRT-LLM’s quantization chapter: the house gets smaller; you still sign for quality.
