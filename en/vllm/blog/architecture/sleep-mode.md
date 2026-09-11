---
source: https://vllm.ai/blog/2025-10-26-sleep-mode
lang: en
fetched: 2026-09-05
---

# Zero-Reload Model Switching with vLLM Sleep Mode

Chinese: [zh/vllm/blog/architecture/sleep-mode.md](../../../../zh/vllm/blog/architecture/sleep-mode.md)  
Source: https://vllm.ai/blog/2025-10-26-sleep-mode

2025-10-26. Embedded LLM. Study extract, not an official reprint. Numbers: **vLLM 0.11.0**, `cudagraph_mode: FULL_AND_PIECEWISE`, A100 and A4000. Original page is Plotly-heavy; this note skips the widgets and keeps seconds from the typed tables plus the chart data series.

Two models that each fit on the GPU, but not together: keep both loaded (**2×** VRAM) or reload on demand (**30–100+ s** per switch). Sleep Mode hibernates in seconds and wakes fast — on-demand efficiency with persistent-serving speed.

Local figures (copyright remains with the original site; study copies):

![sleepmode](../../../../assets/vllm/blog/architecture/sleep-mode/01-sleepmode.png)

**Caption (original).** vLLM Sleep Mode.

## Introduction

**The multi-model serving problem:** two LLMs each fit, both at once do not. Traditional options:

1. **Keep both loaded** → 2× GPU memory (expensive, often impossible)
2. **Reload on demand** → 30–100+ seconds per switch (slow, wasteful)

Sleep Mode is a third way: hibernate in seconds, wake fast.

### Two Sleep Levels for Different Needs

- **Level 1:** offload weights to CPU RAM (fastest wake)
- **Level 2:** discard weights entirely (nearly as fast wake, minimal RAM)

Both claimed **18–200×** vs full reload. Works with Tensor Parallelism (TP), Pipeline Parallelism (PP), and Expert Parallelism (EP).

### Why Sleep Mode Beats Fast Weight Loaders

Even instant weight loading still pays cold-start costs Sleep Mode avoids:

| Cost | Description | Fast Weight Loaders | Sleep Mode |
|------|-------------|---------------------|------------|
| 1. VRAM load time | Copying weights to GPU | Optimized | Preserved |
| 2. Memory allocator setup | CUDA allocator initialization | Every time | Preserved |
| 3. CUDA graph capture | Record execution graphs | Every time | Preserved |
| 4. GPU kernel JIT compilation | DeepGEMM, FlashInfer, TorchInductor | Every time | Preserved after initial warmup |
| 5. Cache warm-up | First-request overhead | Every time | Quick re-warm |

Keeping the process alive preserves infrastructure (#2–4). That is why **Sleep Mode inference is 61–88% faster** than cold starts.

This post covers:

- Benchmarks from 0.6B to 235B, A4000 to A100
- Technical deep-dives on the gains
- Ablations on warm-up and FP8
- Decision guide for sleep level

## Quick Start: Using Sleep Mode

### Online Serving API

Two servers with Sleep Mode:

```bash
# Terminal 1: Start Phi-3-vision
export VLLM_SERVER_DEV_MODE=1
vllm serve microsoft/Phi-3-vision-128k-instruct --enable-sleep-mode --port 8001

# Terminal 2: Start Qwen3-0.6B
export VLLM_SERVER_DEV_MODE=1
vllm serve Qwen/Qwen3-0.6B --enable-sleep-mode --port 8002
```

### Sleep and Wake Models

```bash
# Put Phi-3-vision to sleep (Level 2 - minimal RAM usage)
curl -X POST 'localhost:8001/sleep?level=2'

# Put Qwen3-0.6B to sleep (Level 2)
curl -X POST 'localhost:8002/sleep?level=2'

# Wake up Phi-3-vision for inference
curl -X POST 'localhost:8001/wake_up'
curl -X POST 'localhost:8001/collective_rpc' \
  -H 'Content-Type: application/json' \
  -d '{"method":"reload_weights"}'

# IMPORTANT: Reset prefix cache after waking (Level 2 only)
curl -X POST 'localhost:8001/reset_prefix_cache'

# Now run inference on Phi-3-vision...
# (your inference requests here)

# Put back to sleep when done
curl -X POST 'localhost:8001/sleep?level=2'

# Wake up Qwen3-0.6B
curl -X POST 'localhost:8002/wake_up'
# (Level 1 doesn't need reload_weights or reset_prefix_cache)

# Run inference on Qwen3-0.6B...
```

> **NOTE.** Level 2 must call `reload_weights` and `reset_prefix_cache` after waking. Level 1 does not.

> **WARNING. Security:** `/sleep`, `/wake_up`, `/collective_rpc`, and `/reset_prefix_cache` require `VLLM_SERVER_DEV_MODE=1` and should only be exposed on trusted networks. These admin endpoints can disrupt service — training clusters or backend apps, not the public internet.

## Performance Overview

### Sleep Mode L1 vs No Sleep Mode Performance

Interactive original: **total time for 5 model switches** — infer A, switch to B, infer B, repeat (A→B→A→B→A→B).

**With Sleep Mode:** sleep/wake between switches; infrastructure stays.  
**Without:** each switch is a full vLLM restart and reload.

**Model A:** Qwen3-235B-A22B-Instruct-2507-FP8 (TP=4). **Model B:** Qwen3-Coder-30B-A3B-Instruct (TP=1). GPU: A100. vLLM 0.11.0. Sleep Level 1. Compilation: `cudagraph_mode: FULL_AND_PIECEWISE`.

Event seconds from the chart series (including initial loads):

| Stage | WITH Sleep L1 (s) | WITHOUT Sleep (s) |
|------|-------------------|-------------------|
| A Load (+ warmup 2.38) | 97.61 + 2.38 | 97.9 / 97.4 / 97.71 (three loads) |
| B Load (+ warmup 2.42) | 47.63 + 2.42 | 47.33 / 47.47 / 47.46 |
| A Wake | 5.66, 5.29, 5.27 | — (full Load each time) |
| B Wake | 2.89, 2.86, 2.85 | — |
| A Prompt | 1.8, 1.7, 0.92 | 3.8, 3.7, 3.72 |
| B Prompt | 1.0, 0.93, 0.54 | 3.7, 2.9, 2.45 |
| A Sleep | 6.01, 5.78, 5.89 | — |
| B Sleep | 2.78, 2.78 | — |
| Event sum (approx.) | **~205 s** | **~456 s** |

## Inference Performance Boost

Beyond faster switching, first inference after wake is faster: the model is already warmed up.

Original definition: **inference time = prefill + decode (first request after wake/load)**. Different questions to avoid caching; output capped at **100** tokens. Error bars are min/max across runs. Same A100 / vLLM 0.11.0 / Level 1 / `FULL_AND_PIECEWISE`.

Chart series (three runs):

| Model | Wake (s) | Cold (s) | Wake mean | Cold mean |
|------|----------|----------|-----------|-----------|
| Qwen3-235B-A22B (TP=4) | 1.8, 1.7, 0.92 | 3.8, 3.7, 3.72 | ~1.47 | ~3.74 |
| Qwen3-Coder-30B (TP=1) | 1.0, 0.93, 0.54 | 3.7, 2.9, 2.45 | ~0.82 | ~3.02 |

Prose: wake inference **61–88%** faster than cold start. Cited pair: wake **0.92 s** vs cold **3.72 s** (first inference **4–7×** slower).

#### Why Sleep Mode Improves Inference Speed

The 61–88% is **not** faster memcpy. It is preserved infrastructure.

**What Sleep Mode preserves:**

| Component | Preserved? | Cold Start Must Pay |
|-----------|-----------|---------------------|
| Memory allocator (`CuMemAllocator`) | Yes | Reinitialize every time |
| CUDA graphs | Yes | Re-capture every time |
| Process state (Python, CUDA context) | Yes | Restart every time |
| GPU kernel JIT cache | Yes (after initial warmup) | Recompile every time |

**The critical difference:**

- **Without Sleep Mode:** process dies on unload → **you cannot pre-warm**. Restart Python and CUDA context, reinit allocator, recapture graphs, re-JIT (DeepGEMM, FlashInfer, TorchInductor). **Result:** first inference **4–7×** slower (0.92 s wake vs 3.72 s cold).
- **With Sleep Mode:** process stays → **pre-warm pays off**. Allocator, graphs, process state, JIT kernels preserved after initial warmup. **Result:** first inference stays ~1 s, skipping a 3–4 s cold penalty.

> **NOTE.** Timing varies a lot by model size, GPU generation, and config. See [Impact of Warm-Up](#impact-of-warm-up-on-sleep-mode): **5–7×** slower without warm-up.

## Model Switching Performance

Waking a sleeping model is **18–20×** faster than loading a fresh vLLM instance.

Same A100 / Level 1 / `FULL_AND_PIECEWISE`. Chart series (three runs):

| Model | Wake (s) | Cold load (s) | Wake mean | Cold mean | Approx. speedup |
|------|----------|---------------|-----------|-----------|-----------------|
| Qwen3-235B-A22B (TP=4) | 5.66, 5.29, 5.27 | 97.9, 97.4, 97.71 | ~5.41 | ~97.7 | **~18×** |
| Qwen3-Coder-30B (TP=1) | 2.89, 2.86, 2.85 | 47.33, 47.47, 47.46 | ~2.87 | ~47.4 | **~17×** |

## Hardware Scalability: A4000 GPU Results

Same workload on an **A4000** with smaller models: gains hold across GPU class and size.

**Model A:** Qwen3-0.6B. **Model B:** Phi-3-vision-128k-instruct. GPU: A4000 (TP=1). vLLM 0.11.0. Sleep Level 1. `cudagraph_mode: FULL_AND_PIECEWISE`.

Chart series (including initial loads):

| Stage | WITH Sleep L1 (s) | WITHOUT Sleep (s) |
|------|-------------------|-------------------|
| A Load (+ warmup 2.49) | 21.01 + 2.49 | 21.04 / 20.98 / 20.98 |
| B Load (+ warmup 7.37) | 46.01 + 7.37 | 46.01 / 46.02 / 46.02 |
| A Wake | 0.11, 0.10, 0.10 | — |
| B Wake | 0.80, 0.80, 0.80 | — |
| A Prompt | 0.44, 0.43, 0.43 | 2.64, 2.50, 2.63 |
| B Prompt | 2.04, 1.73, 1.61 | 9.78, 9.01, 9.79 |
| Prose total (5 switches) | **85 s** | **226 s** (**~62%**) |

### A4000: Inference Performance

Inference = prefill + decode (first request after wake/load); different questions; 100-token cap.

| Model | Wake (s) | Cold (s) | Prose improvement |
|------|----------|----------|-------------------|
| Qwen3-0.6B | 0.44, 0.43, 0.43 | 2.64, 2.50, 2.63 | **83%** faster |
| Phi-3-vision-128k (4B) | 2.04, 1.73, 1.61 | 9.78, 9.01, 9.79 | **81%** faster |

### A4000: Model Switching Performance

| Model | Wake (s) | Cold (s) | Approx. speedup |
|------|----------|----------|-----------------|
| Qwen3-0.6B | 0.11, 0.10, 0.10 | 21.04, 20.98, 20.98 | **~200×** |
| Phi-3-vision-128k (4B) | 0.80, 0.80, 0.80 | 46.01, 46.02, 46.02 | **~58×** |

**Key observations on A4000:**

- **Inference:** 83% faster for Qwen3-0.6B, 81% for Phi-3-vision
- **Switching:** wake **~0.1–0.8 s**, **58–203×** vs cold start
- **Total time savings 62%** (85 s vs 226 s for 5 switches)
- Near-instant for small models (~0.1 s wake)
- Sleep Mode works across GPU classes and model sizes

## Sleep Levels: Choosing the Right Mode

Two levels, different tradeoffs:

**Level 1 (default):** offload weights to CPU, discard KV cache

- **Fastest wake** (~0.1–0.8 s small models, ~3–6 s large)
- **Needs enough CPU RAM** for weights
- **Best for:** ample RAM, frequent switching

**Level 2:** discard weights and KV; CPU keeps only small buffers (RoPE scaling tensors, etc.)

- **Slower wake** (~0.8–2.6 s small models) — reload weights from disk
- **Minimal CPU RAM** — tiny buffers only
- **Best for:** tight RAM, or many models that will not all fit

### Performance Comparison: Level 1 vs Level 2 vs No Sleep

**Model A:** Qwen3-0.6B. **Model B:** Phi-3-vision-128k-instruct. GPU: A100 (TP=1). vLLM 0.11.0. `FULL_AND_PIECEWISE`.

**Performance summary:**

| Mode | Total Time | Wake Time (A/B) | CPU RAM | Best For |
|------|------------|-----------------|---------|----------|
| **No Sleep** | 357.1 s | N/A (full reload) | Minimal | Single model, no switching |
| **Level 1** | 112.6 s | 0.26 s / 0.82 s | High (~GB per model) | Frequent switching, ample RAM |
| **Level 2** | 124.6 s | 0.85 s / 2.58 s | Minimal (~MB per model) | Tight RAM, cost optimization |

Chart series wakes: L1 A 0.25 / 0.28 / 0.25 s, B 0.82 / 0.82 / 0.83 s; L2 A 0.91 / 0.78 / 0.85 s, B 2.55 / 2.62 / 2.58 s.

**Key insights:**

- **Level 1 is fastest** (68% faster than no sleep) but needs significant CPU RAM
- **Level 2 is nearly as fast** (65% faster than no sleep) with minimal RAM
- **Level 2 wake is ~3× slower than Level 1** (0.85 s vs 0.26 s for Qwen3-0.6B) because of weight reload
- Both sleep modes are large wins vs no sleep

#### Why Level 2 is Still Faster Than No Sleep Mode

Level 2 **also reloads weights from SSD** (like No Sleep). Why **23–45×** faster overall?

**Weight loading is only one of five costs.**

| Cost | Level 2 | No Sleep Mode |
|------|---------|---------------|
| 1. Weight load (SSD → VRAM) | Pay | Pay |
| 2. Process initialization | **Skipped** | Pay |
| 3. Memory allocator setup | **Skipped** | Pay |
| 4. CUDA graph capture | **Skipped** | Pay |
| 5. GPU kernel JIT compilation | **Preserved (already compiled)** | Full compile + warm-up |

**Level 2:** same SSD weight reload; **everything else stays** — process, allocator instance, CUDA graphs, compiled JIT kernels. No recompilation after initial warmup. **~2.6 s average per switch.**

**No Sleep:** same disk hit; **everything else rebuilt** — process restart + allocator + graph recapture; JIT is full compile + explicit `kernel_warmup()` + dummy runs. **~48 s average per switch.**

Five switches:

- **Level 2:** 124.6 s total (~2.6 s per switch)
- **No Sleep:** 357.1 s total (~48 s per switch)

Both reload from SSD. Level 2 is still **2.9×** overall because it keeps the expensive infrastructure.

### Level 2: Inference Performance

A100, TP=1, Sleep Level 2, `FULL_AND_PIECEWISE`. Inference = prefill + decode (first request); different questions; 100-token cap.

Chart series:

| Model | Wake (s) | Cold (s) |
|------|----------|----------|
| Qwen3-0.6B | 0.68, 0.46, 0.44 | 4.66, 3.80, 2.56 |
| Phi-3-vision-128k | 0.78, 0.77, 0.72 | 6.55, 6.21, 6.15 |

### Level 2: Model Switching Performance

Chart series:

| Model | Wake (s) | Cold (s) |
|------|----------|----------|
| Qwen3-0.6B | 0.91, 0.78, 0.85 | 38.53, 37.21, 38.15 |
| Phi-3-vision-128k | 2.55, 2.62, 2.58 | 58.52, 57.65, 58.20 |

**Key observations:**

| Metric | No Sleep | Level 2 | Improvement |
|--------|----------|---------|-------------|
| **Total Time (5 switches)** | 357.1 s | 124.6 s | **65%** faster |
| **Qwen3-0.6B Switch Time** | 37.6 s avg | 0.85 s avg | **45×** |
| **Phi-3-vision Switch Time** | 58.1 s avg | 2.58 s avg | **23×** |
| **Qwen3-0.6B Inference** | 3.67 s avg | 0.53 s avg | **86%** faster |
| **Phi-3-vision Inference** | 6.30 s avg | 0.76 s avg | **88%** faster |
| **Wake Time vs Level 1** | — | 3–10× slower | Trade CPU RAM for speed |

**When to use Level 2:**

- **Limited CPU RAM:** cannot hold all weights
- **Cost optimization:** cheaper cloud boxes with less RAM
- **Many models:** CPU memory is the constraint
- **Still large gains:** 23–45× vs no sleep even with weight reload

**Level 1 vs Level 2:**

- Level 1: ~0.1–0.8 s wake, ~10–100GB+ CPU RAM per model
- Level 2: ~0.8–2.6 s wake, ~MB CPU RAM per model
- Both vs full reload ~20–100 s

## Ablation Studies

### Impact of Warm-Up on Sleep Mode

Does skipping warm-up hurt? Warm-up pre-compiles CUDA graphs on initial load (several seconds). With vs without.

**Model A:** Qwen3-0.6B. **Model B:** Phi-3-vision-128k-instruct. A100, TP=1, Level 1, `FULL_AND_PIECEWISE`.

**Key findings:**

| Metric | With Warm-Up | Without Warm-Up | Difference |
|--------|--------------|-----------------|------------|
| **Initial Load Time** | 108.7 s (includes 8.4 s warm-up) | 101.1 s (no warm-up) | 7.6 s saved initially |
| **First Inference (A)** | 0.45 s | 2.59 s | **5.8×** slower without |
| **First Inference (B)** | 0.93 s | 6.61 s | **7.1×** slower without |
| **Subsequent Inferences** | 0.43 s avg | 0.41 s avg | None |
| **Total Time (5 switches)** | 119.5 s | 119.0 s | Nearly identical |

Chart series: with warm-up A Load 37.65 + Warm Up 2.39, B Load 62.69 + Warm Up 6.0; first A Prompt 0.45, B Prompt 0.93. Without: A Load 37.91, B Load 63.16; first A Prompt 2.59, B Prompt 6.61; later prompts ~0.41 / 0.70.

**Insights:**

- **Warm-up compiles once, every wake benefits:** JIT and CUDA graphs paid on load, preserved across sleep/wake
- **Without warm-up, every wake pays compilation:** the 5–7× hit is on the **first inference after every wake**, not once
- **Compiled kernels survive sleep/wake:** after 8.4 s warmup, later first inferences are 0.45 s and 0.93 s
- **Minimal warmup is enough:** a single **1-token** inference triggers full JIT + graph capture
- **Trade initial load for consistent latency:** 8.4 s once, amortized across switches
- **Recommendation: always warm up** in production if first-inference consistency matters

Totals look similar because 8.4 s is amortized; the *shape* of latency is not.

### Impact of Quantization on Sleep Mode

Does FP8 change Sleep Mode? Same workload on A100, BF16 vs FP8.

Same small pair, TP=1, Level 1, `FULL_AND_PIECEWISE`.

### Ablation: Inference Performance (BF16 vs FP8)

Chart series (three prompts):

| Model | BF16 (s) | FP8 (s) |
|------|----------|---------|
| Qwen3-0.6B | 0.41, 0.40, 0.41 | 0.43, 0.43, 0.45 |
| Phi-3-vision-128k | 0.90, 0.74, 0.80 | 0.69, 0.59, 0.44 |

### Ablation: Model Switching (BF16 vs FP8)

Chart series (three wakes):

| Model | BF16 (s) | FP8 (s) |
|------|----------|---------|
| Qwen3-0.6B | 0.28, 0.27, 0.27 | 0.18, 0.19, 0.16 |
| Phi-3-vision-128k | 0.89, 0.93, 0.88 | 0.79, 0.77, 0.78 |

**Key findings:**

| Metric | BF16 | FP8 | Improvement |
|--------|------|-----|-------------|
| **Total Time (5 switches)** | 108.2 s | 113.6 s | −5% (slightly slower) |
| **Qwen3-0.6B Wake Time** | 0.27 s avg | 0.18 s avg | **33%** faster |
| **Phi-3-vision Wake Time** | 0.90 s avg | 0.78 s avg | **13%** faster |
| **Qwen3-0.6B Inference** | 0.41 s avg | 0.44 s avg | −7% (slightly slower) |
| **Phi-3-vision Inference** | 0.81 s avg | 0.57 s avg | **30%** faster |
| **Initial Load Time** | 90.5 s | 96.9 s | −7% (longer warmup) |

**Insights:**

- **FP8 wakes faster** (13–33%) — less memory movement
- **Larger-model inference benefits more** (30% for Phi-3-vision); tiny models barely move
- **Initial load longer** — quantization during warmup
- After load, FP8 switching is smoother
- Frequent switching can still prefer FP8’s cheaper wakes over a longer first load

## Decision Guide: Which Sleep Level to Use?

### Use Sleep Level 1 When:

- Enough CPU RAM to hold all model weights
- Fastest possible wake (0.1–6 s)
- Switching every few seconds/minutes
- Inference latency consistency is critical

### Use Sleep Level 2 When:

- CPU RAM cannot hold all weights
- Optimizing cloud cost (cheaper, less RAM)
- Many models to manage (10+)

### Skip Sleep Mode When:

- Single model (no switching)
- Switches extremely rare (once per day/week)
- Both models already fit in GPU memory together

## Conclusion

Sleep Mode turns 30–100 second reloads into sub-second switches:

- **18–200×** faster model switching (size and hardware)
- **61–88%** faster inference for warmed models vs cold starts
- **65–68%** total time savings on complete workloads
- **Works at every scale:** 0.6B to 235B, small and large GPUs

Multi-model serving is the future. Sleep Mode makes it practical now.

## Acknowledgements

Vensen Mu, Jeff Aw, Jun Kang Chow, Tun Jian Tan, Pin Siang Tan, Amir Balwel, Ye Hur Cheong, Zhiyao Cen, and Kaichao You for Sleep Mode and this post.

[torch.compile](torch-compile.md) made startup expensive. Sleep refuses to kill the process. Dense LoRA is the other multi-model path; this one swaps **whole weights**.
