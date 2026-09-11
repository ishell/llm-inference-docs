---
source: https://vllm.ai/blog/2026-04-22-fp8-kvcache
lang: en
fetched: 2026-09-05
---

# The State of FP8 KV-Cache and Attention Quantization

Chinese: [zh/vllm/blog/performance/fp8-kvcache.md](../../../../zh/vllm/blog/performance/fp8-kvcache.md)  
Original: https://vllm.ai/blog/2026-04-22-fp8-kvcache  
2026-04-22. Jonas Kübler* (AWS), Eldar Kurtić* (Red Hat AI), Lucas Wilkinson, Matthew Bonanni, Michael Goin, Alexandre Marques (Red Hat AI), Kailash Budhathoki (AWS); * equal contribution. Study extract, not an official reprint.

Long-context serving is increasingly memory-bound: for full-attention decoders the KV cache often dominates GPU memory at 128k+, and every Decode step rereads a large fraction of it. `--kv-cache-dtype fp8` quantizes the KV cache **and** runs the whole attention computation (QK and ScoreV matmuls) in FP8. The format used throughout is **e4m3**. Halving that storage can mean more concurrency or a longer window on the same card — if accuracy holds.

The flag had been in vLLM for a while. They stress-tested prefill-heavy and decode-heavy workloads across decoder-only and MoE models, Hopper and Blackwell, found and fixed critical accuracy and performance bugs in FA3 (Figure 1). On the validated paths: near-baseline accuracy, lower decode cost and KV memory. Main caveats: hybrid-attention models with **small** sliding-window layers (skip those layers), and `head_dim = 256` (Prefill can still regress). For `head_dim` 64 and 128, FP8 can speed **both** Prefill and Decode. Best-case memory-bound Decode: per-token KV cost down to **54%** of BF16. Large head dims (256) still cut ITL; default Prefill is still slightly worse than BF16.

```bash
# FP8 KV-cache for all layers
vllm serve meta-llama/Llama-3.1-8B --kv-cache-dtype fp8

# Hybrid-attention models: skip sliding-window layers (recommended)
vllm serve gpt-oss-20b --kv-cache-dtype fp8 --kv-cache-dtype-skip-layers sliding_window
```

Accuracy numbers are **uncalibrated per-tensor scale=1.0** — the simplest knob, and a lower bound. Calibration and per-head scales only get better.

Local figures (copyright remains with the original site; study copies):

![fig1 niah before after plot](../../../../assets/vllm/blog/performance/fp8-kvcache/01-fig1_niah_before_after_plot.png)

*Figure 1 (original caption): Needle-in-a-haystack at 128k on Hopper before and after the FP8 Flash Attention 3 fixes. The accumulation fix restores long-context accuracy from a severe FP8 regression back near the BF16 baseline, while the optimized FP8 path still preserves the decode-speed advantage.*

Original TOC (this note follows the same path): Problems → Kernel and vLLM improvements → Single request → Throughput under load → Large head dims → FlashInfer on B200 → Accuracy (reasoning / long-context / B200 / when to calibrate) → When to avoid.

## The problems they found

`--kv-cache-dtype fp8` had been in vLLM for a while. Stress tests turned up two classes of bug.

**Accuracy.** Hopper FA3 FP8 lost accumulation precision at long context. On 128k NIAH, accuracy fell from **91%** (BF16) to **13%**. Cause: the two-level accumulation section below.

**Performance.** On hybrid models with sliding-window layers (gpt-oss-20b), the FP8 ITL slope was **96%** of BF16 — almost no Decode speedup despite half the memory. Break-even exceeded **700k** tokens (Table 3: **741,565**), past any practical window.

The next section is what they shipped.

## Kernel and vLLM improvements

Shipped during the investigation: more flexible quant schemes, accuracy fixes, performance.

**Two-level accumulation.** Hopper FP8 Tensor Cores are documented as accumulating into FP32 registers, but intermediate accumulation loses precision when the contraction dimension is large — a known hardware issue also seen in DeepSeek-V3 training ([Technical Report](https://arxiv.org/abs/2412.19437) Figure 7(b)). At **100K or more**, numerical errors get drastic. In inference that contraction dim is context length: `Softmax(AttnScore) * V`. Empirically: NIAH 91% → 13%.

Mitigation: two-level accumulation ([SageAttention2](https://arxiv.org/abs/2411.10958)) that writes partial sums into a *real* FP32 register ([flash-attention#104](https://github.com/vllm-project/flash-attention/pull/104)). NIAH comes back to **89%**. Cost: register pressure, Prefill slowdown. Tiling ([flash-attention#125](https://github.com/vllm-project/flash-attention/pull/125)) recovers some of it for `head_dim` 64/128; **above 128, Prefill still trails BF16**.

**Skipping layers.** vLLM used to allow only one numeric format for all Attention layers. `--kv-cache-dtype-skip-layers` ([vllm#33695](https://github.com/vllm-project/vllm/pull/33695)) allows hybrids. GPT-OSS sliding-window layers attend a fixed window (e.g. 128 tokens); FP8 overhead never amortizes. Keep those layers BF16 (numbers below). Same flag can skip quantization-sensitive layers.

**Per-head scales.** FA3 already takes an array of scales, one per KV head. Wiring it needed general static-quant group shapes ([vllm#30833](https://github.com/vllm-project/vllm/pull/30833)) and `reshape_and_cache_flash` taking an array instead of one scalar ([vllm#30141](https://github.com/vllm-project/vllm/pull/30141)).

**Query quantization fusion.** Move query quant out of the attention backend into a torch op that `torch.compile` can fuse ([vllm#24914](https://github.com/vllm-project/vllm/pull/24914)). Kills a fixed per-token overhead.

**FA3 FP8 tile sizes.** Prefill tiles for `head_dim=64` and `128` to cut register spills from two-level accum ([flash-attention#125](https://github.com/vllm-project/flash-attention/pull/125)). Separate Decode tiles that flatten the ITL slope ([flash-attention#96](https://github.com/vllm-project/flash-attention/pull/96), [#91](https://github.com/vllm-project/flash-attention/pull/91)).

Hopper and Blackwell have 2× FP8 FLOPs vs BF16, so Prefill *should* also win. In practice that is not free — the rest of the post.

## Performance: single request (concurrency 1)

Every generated token attends the full KV cache, so ITL grows linearly with input length. FP8 halves bytes per cached token and should halve the ITL slope. Prefill is quadratic; 2× FP8 FLOPs should help there too. The following sections show those gains are not always out of the box.

Concurrency 1 isolates attention: ITL and TTFT fully separate. Fit:

`ITL = slope × input_len + intercept`

`TTFT = a × input_len² + b × input_len + c`

Slope is the per-token attention tax. **Break-even** = context length where FP8 ITL drops below BF16. The quadratic TTFT model is compute-bound prefill (self-attention over the input).

Setup: 1×H100, [FlashAttention-3](https://openreview.net/forum?id=tVConYid20) via the [vLLM fork](https://github.com/vllm-project/flash-attention) (native FP8 KV + online softmax rescaling). `vllm bench serve`, concurrency 1, 128 output tokens, input lengths 256 → 125k.

![fig2 llama 8b](../../../../assets/vllm/blog/performance/fp8-kvcache/02-fig2_llama_8b.png)

*Figure 2 (original caption): Single-request H100 benchmark for Llama-3.1-8B. FP8 nearly halves the decode ITL slope relative to BF16 with almost no intercept penalty, bringing the decode break-even point down to about 7k tokens while preserving similar TTFT.*

Fitted ITL slope `4.37e-05` → `2.37e-05` ms/token; intercept `6.44` → `6.58` ms. Slope ratio **54%** of BF16 (near optimal); intercept gap **0.14 ms**. Break-even ~**7k**. Even with two-level accumulation on, long-context TTFT is *slightly* better than BF16.

| | BF16 | FP8 |
|---|---|---|
| ITL slope (ms/token) | `4.37e-05` | `2.37e-05` |
| intercept (ms) | 6.44 | 6.58 |
| slope vs BF16 | 100% | **54%** |
| intercept gap | — | +0.14 ms |
| break-even | — | **~7k** |

![fig3 gptoss 20b](../../../../assets/vllm/blog/performance/fp8-kvcache/03-fig3_gptoss_20b.png)

*Figure 3 (original caption): Single-request H100 benchmark for gpt-oss-20b. Skipping sliding-window layers is the best FP8 variant because those layers have bounded KV-cache footprints, so they pay quantization overhead without getting much long-context benefit.*

gpt-oss-20b: 20B hybrid, global + sliding-window (**window 128**). Sliding-window KV is bounded, so quantizing those layers yields diminishing returns. `--kv-cache-dtype-skip-layers sliding_window` keeps them BF16 and quantizes only global layers.

Fitted slope: BF16 `8.94e-06` → full FP8 `7.14e-06` → skip-SW `6.34e-06` ms/token. Intercepts `4.03`–`4.07` ms. Slope vs BF16: 80% (full FP8), **71%** (skip-SW). Before the improvements, BF16 and FP8 slopes were nearly identical.

Skip-SW is the winner: bounded SW layers stay BF16 (quant adds constant overhead, no long-context memory win) → lowest slope, tiny intercept penalty. Recommendation in the post: use this variant for hybrids.

| | BF16 | FP8 (all layers) | FP8 skip-SW |
|---|---|---|---|
| ITL slope (ms/token) | `8.94e-06` | `7.14e-06` | `6.34e-06` |
| intercept (ms) | ~4.03–4.07 | same cluster | same cluster |
| slope vs BF16 | 100% | 80% | **71%** |
| break-even | — | 22,109 | **7,659** |

Practical takeaway: for long-context decode-heavy serving where KV traffic dominates, on H100 FP8 is already clearly beneficial for Llama-class models and for hybrid models once small sliding-window layers are skipped.

### ITL slope table (before vs after)

*Table 3 from the post: summary across both analyzed models and KV-cache variants. “Before” is v0.10.2; “after” is v0.19.1.*

| Model | Version | FP8 variant | Break-even (tokens) | FP8 slope (% of BF16) |
|---|---|---|---|---|
| Llama-3.1-8B | v0.10.2 | FP8 | 24,889 | 63% |
| Llama-3.1-8B | v0.19.1 | FP8 | **7,010** | **54%** |
| gpt-oss-20b | v0.10.2 | FP8 | 741,565 | 96% |
| gpt-oss-20b | v0.19.1 | FP8 | 22,109 | 80% |
| gpt-oss-20b | v0.19.1 | FP8 skip-SW | **7,659** | **71%** |

## Throughput under load

Concurrency 1 isolates per-token attention. Under load: **150 requests, concurrency 8**, ~20k input / ~2k output (±15%). Tables 4 and 5.

*Table 4: Llama-3.1-8B. FP8: **+14.9%** output tok/s, **−13.0%** total duration, **−14.8%** median ITL.*

| Config | Median TTFT (ms) | Median ITL (ms) | Total duration (s) | Output tok/s |
|---|---|---|---|---|
| BF16 | 763.6 | 15.18 | 672.6 | 450.3 |
| FP8 | 742.8 | 12.93 | 585.2 | **517.5** |

*Table 5: gpt-oss-20b. Skip-SW: **+4.8%** output tok/s, **−4.6%** duration, **−4.8%** median ITL.*

| Config | Median TTFT (ms) | Median ITL (ms) | Total duration (s) | Output tok/s |
|---|---|---|---|---|
| BF16 | 468.9 | 8.09 | 364.2 | 831.6 |
| FP8 | 451.7 | 7.90 | 355.1 | 853.0 |
| FP8 skip-SW | 456.4 | 7.70 | 347.4 | **871.8** |

Single-request ITL improvements become real serving gains. Llama’s 54% ITL-slope cut at c=1 becomes +14.9% output throughput at c=8 — faster tokens **and** 2× KV memory, so the scheduler packs more. gpt-oss gains less because sliding-window layers cap the memory win; skip-SW recovers the most.

These runs are moderately heavy (c=8, ~20k inputs). At higher concurrency or longer context, BF16 OOMs or evicts first; that is when the house-size advantage actually shows.

## `head_dim=256` (Prefill regression)

Two-level accumulation is **on by default** after flash-attention#104, so users do not silently eat a 91%→13% NIAH cliff. At large head dims that default makes TTFT worse than BF16.

![fig4 gemma](../../../../assets/vllm/blog/performance/fp8-kvcache/04-fig4_gemma.png)

*Figure 4 (original caption): gemma-4-E2B on H100 (`head_dim = 256`). FP8 improves decode ITL, but prefill becomes slower because two-level accumulation raises register pressure enough to outweigh the FP8 arithmetic advantage.*

gemma-4-E2B: `head_dim=256`. Three of four layers are sliding-window **512** (4× gpt-oss’s 128).

ITL slope `5.30e-05` → `3.60e-05` ms/token (**68%**). TTFT quadratic `6.93e-07` → `1.12e-06` ms/token² (**~1.6×**). Decode wins across the measured range. Window 512 is large enough to amortize FP8, so quantizing SW layers *does* pay — a constant offset vs skipping them. Prefill is significantly *slower* at long contexts from register pressure at `head_dim=256`.

| | BF16 | FP8 |
|---|---|---|
| ITL slope (ms/token) | `5.30e-05` | `3.60e-05` (**68%**) |
| TTFT quadratic coeff (ms/token²) | `6.93e-07` | `1.12e-06` (**~1.6×**) |

Two ways out named in the post:

1. **Disable two-level accumulation** — recovers Prefill, including extra Prefill wins at head dim 64/128. **You must accuracy-test the workload.**
2. Accumulate only every N steps: open PR [flash-attention#122](https://github.com/vllm-project/flash-attention/pull/122). Functional; recovers Prefill speedups.

Option 1 in particular would also give larger prefill speedups at head dims 64 and 128.

## FlashInfer on Blackwell (B200)

Most performance work targeted H100 and FA3. B200 + FlashInfer is included for completeness. Accumulation is a Hopper FA3 problem. B200 does not need two-level accum.

![fig5 llama b200](../../../../assets/vllm/blog/performance/fp8-kvcache/05-fig5_llama_b200.png)

*Figure 5 (original caption): Llama-3.1-8B on B200 with FlashInfer. FP8 again reduces the decode ITL slope to about 54% of BF16 with an almost negligible intercept penalty, so decode breaks even at roughly 4k tokens.*

Slope `1.80e-05` → `9.72e-06` ms/token; intercept `3.93` → `3.96` ms.

| | BF16 | FP8 |
|---|---|---|
| ITL slope (ms/token) | `1.80e-05` | `9.72e-06` |
| intercept (ms) | 3.93 | 3.96 |
| break-even | — | **~4k** |

![fig6 gptoss b200](../../../../assets/vllm/blog/performance/fp8-kvcache/06-fig6_gptoss_b200.png)

*Figure 6 (original caption): gpt-oss-20b on B200 with FlashInfer. FP8 lowers the decode ITL slope more strongly than on H100, but the model still needs longer contexts before the smaller slope outweighs the fixed overhead.*

Slope `3.56e-06` → `2.06e-06` ms/token; intercept `3.15` → `3.17` ms; fitted break-even **~13k**.

| | BF16 | FP8 |
|---|---|---|
| ITL slope (ms/token) | `3.56e-06` | `2.06e-06` |
| intercept (ms) | 3.15 | 3.17 |
| break-even | — | **~13k** |

Unlike H100, these B200 runs compare BF16 vs FP8 only — **layer skipping was not yet supported on B200** when they ran.

## Accuracy

Models: Llama-3.3-70B-Instruct, Qwen3-30B-A3B-Instruct-2507, Qwen3-30B-A3B-Thinking-2507, Qwen3.5-27B.

- Long-context (Prefill-heavy): `openai/mrcr`, up to 1M. Average pass@1 per length bucket over **5** repetitions; **AUC** across lengths ([Context Arena](https://contextarena.ai/)).
- Reasoning (Decode-heavy): AIME25, GPQA:Diamond, MATH500, LiveCodeBench-v6. pass@1: **10** reps for AIME25 and LiveCodeBench-v6; **5** for GPQA and MATH500.
- Sampling: each model’s default **non-greedy** params.

**All of these numbers use per-tensor uncalibrated scale=1.0.** Simplest possible config — no calibration data, no per-head tuning — worst-case accuracy. Two reasons: (1) trivially reproducible via `--kv-cache-dtype fp8`; (2) lower bound — calibrated scales only get better. They also support calibration via [`vllm-project/LLM-Compressor`](https://github.com/vllm-project/llm-compressor) and per-head scales ([vllm#30141](https://github.com/vllm-project/vllm/pull/30141)); see [vLLM’s quantized KV-cache examples](https://github.com/vllm-project/vllm/blob/4f436782afd0b21d6754ea6bc4b80639f737bbc1/docs/features/quantization/quantized_kvcache.md#3-recommended-calibration-using-a-dataset-with-llm-compressor).

### Reasoning

Short Prefill, long Decode (tens of thousands of tokens). Tests whether FP8 KV + FP8 attention degrades reasoning across extended generation chains.

![fig7 Qwen3 30B A3B Thinking 2507 reasoning combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/07-fig7_Qwen3-30B-A3B-Thinking-2507_reasoning_combined_plot.png)

*Figure 7 (original caption): Reasoning benchmarks for Qwen3-30B-A3B-Thinking-2507. In both the BF16-model and FP8-model settings, enabling FP8 KV-cache plus FP8 attention changes average accuracy by only about 1–2 points across these decode-heavy tasks.*

At most **1–2 points**. Lowest recovery **97%** (GPQA:Diamond, BF16 model).

![fig8 Qwen3.5 27B reasoning combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/08-fig8_Qwen3.5-27B_reasoning_combined_plot.png)

*Figure 8 (original caption): Reasoning benchmarks for Qwen3.5-27B. FP8 KV-cache plus FP8 attention is nearly lossless here, with sub-point differences across the aggregate scores in both BF16-model and FP8-model settings.*

At most **0.7 points**. Lowest recovery **99%** (AIME25, BF16 model).

### Long-context MRCR

Heavy prefill, short decode. Validates up to **1M**-token prompts.

![fig9 Llama 3.3 70B Instruct openai mrcr 2 needles combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/09-fig9_Llama-3.3-70B-Instruct_openai_mrcr_2_needles_combined_plot.png)

*Figure 9 (original caption): MRCR results for Llama-3.3-70B-Instruct up to 128k context. The FP8 KV-cache plus FP8 attention curves track the baseline closely in both BF16-model and FP8-model settings, recovering about 97–98% of the baseline AUC.*

**97–98%** of baseline AUC@128k in both BF16-model and FP8-W&A settings.

![fig10 Qwen3 30B A3B Instruct 2507 openai mrcr 2 needles combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/10-fig10_Qwen3-30B-A3B-Instruct-2507_openai_mrcr_2_needles_combined_plot.png)

*Figure 10 (original caption): MRCR results for Qwen3-30B-A3B-Instruct-2507 up to 256k context. FP8 KV-cache plus FP8 attention remains close to baseline overall, but the longest buckets show a clearer gap here than for Llama; AUC recovery is about 94% in the BF16-model setting and about 98% in the FP8-model setting.*

Higher bucket variance on both sides; part of it is the baseline itself (32k > 8k/16k; 128k > 64k). AUC@256k still close: **~94%** (BF16 model) / **~98%** (FP8 model).

![fig11 Qwen3.5 27B openai mrcr 4 needles combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/11-fig11_Qwen3.5-27B_openai_mrcr_4_needles_combined_plot.png)

*Figure 11 (original caption): MRCR results for Qwen3.5-27B up to 1M context. FP8 KV-cache plus FP8 attention matches the baseline aggregate AUC in both model settings, although the longest context buckets still show visible variance.*

Even at 1M on a strong baseline, aggregate AUC@1M **fully recovers**.

### FlashInfer on B200 (accuracy)

Unlike Hopper, which needed two-stage accumulation for FA3 precision, Blackwell uses the default FlashInfer kernel and skips those interventions. Same recipes as Hopper: Qwen3-30B-A3B-Instruct-2507 (BF16/FP8) on `openai/mrcr`; Qwen3-30B-A3B-Thinking-2507 (BF16/FP8) on reasoning.

![fig12 Qwen3 30B A3B Instruct 2507 openai mrcr 2 needles combined B200 plot](../../../../assets/vllm/blog/performance/fp8-kvcache/12-fig12_Qwen3-30B-A3B-Instruct-2507_openai_mrcr_2_needles_combined_B200_plot.png)

*Figure 12 (original caption): MRCR results for Qwen3-30B-A3B-Instruct-2507 with FlashInfer. FP8 KV-cache plus FP8 attention remains competitive to baseline: AUC recovery is about 93% in the BF16-model setting and about 96% in the FP8-model setting.*

![fig13 Qwen3 30B A3B Thinking 2507 reasoning combined B200 plot](../../../../assets/vllm/blog/performance/fp8-kvcache/13-fig13_Qwen3-30B-A3B-Thinking-2507_reasoning_combined_B200_plot.png)

*Figure 13 (original caption): Reasoning benchmarks for Qwen3-30B-A3B-Thinking-2507 with FlashInfer. The FP8 KV-cache plus FP8 attention configuration stays close to baseline, with average differences of roughly a point or less across the two model settings.*

On B200 + FlashInfer, FP8 KV + FP8 attention stays competitive while keeping the smaller KV cache and lower decode cost. Strong, though not uniformly as tight as the best Hopper/FA3 cases.

### Final remarks

FP8 KV is ready as the **default starting point** for many long-context, Decode-heavy, memory-bound deployments. Exceptions: Prefill-dominated `head_dim=256`; hybrid models whose small sliding-window layers should stay BF16; backends/models with a persistent uncalibrated drop (calibrate).

Primary focus here is uncalibrated scale. For niche accuracy recovery they also shipped: (1) scale calibration on a user dataset via [`LLM-Compressor`](https://github.com/vllm-project/llm-compressor); (2) per-attention-head scales ([vllm#30141](https://github.com/vllm-project/vllm/pull/30141)). Examples: the vLLM docs link above.

### When to calibrate

Not every model likes scale=1.0. Kimi-K2.5 uses **FlashMLA** (not FA3/FlashInfer) on H200.

![fig14 Kimi K2.5 openai mrcr 4 needles H200 plot](../../../../assets/vllm/blog/performance/fp8-kvcache/14-fig14_Kimi-K2.5_openai_mrcr_4_needles_H200_plot.png)

*Figure 14 (original caption): MRCR results for Kimi-K2.5 with FlashMLA and uncalibrated FP8 KV-cache plus FP8 attention. The drop is modest in aggregate AUC but consistently negative across context lengths, which makes this a good example of when calibration is worth doing.*

A consistent downward shift across buckets. Aggregate AUC drop is modest and error bands overlap, but the degradation is **systematic**, not noisy outliers. Start uncalibrated; calibrate when you see that persistent downward shift on the real workload. Especially relevant for non-FA3/FlashInfer backends (e.g. FlashMLA), where FP8 kernel behavior may differ from the validated paths.

## When to avoid FP8 KV-cache

Stay on BF16 (or skip the offending layers) if:

- **Contexts ≲ ~7k.** FP8’s intercept gap can make BF16 slightly faster on ITL.
- **`head_dim=256` and TTFT / Prefill matters.** Two-level accum ~**1.6×** TTFT quadratic. Disabling it recovers speed only with careful accuracy checks.
- **Uncalibrated accuracy <95% on your set.** Kimi-K2.5 + FlashMLA is the worked example; calibrate on target data.
- **Many tiny sliding-window layers.** Prefer `--kv-cache-dtype-skip-layers sliding_window` rather than all-layer FP8.

Same tax as TensorRT-LLM’s quantization chapter: the house gets smaller; you still sign for quality. Neighbors: [turboquant.md](turboquant.md), [torch-compile.md](../architecture/torch-compile.md).
