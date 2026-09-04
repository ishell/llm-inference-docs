---
source: https://vllm.ai/blog/2025-10-26-sleep-mode
lang: en
fetched: 2026-09-04
---

# Zero-Reload Model Switching with vLLM Sleep Mode

Chinese: [zh/vllm/blog/architecture/sleep-mode.md](../../../../zh/vllm/blog/architecture/sleep-mode.md)

2025-10-26. Demo: **vLLM 0.11.0**. Two models that each fit on the GPU, but not together: keep both loaded (**2×** VRAM) or reload on demand (**30–100+ s** per switch). Sleep Mode hibernates the model and **keeps the process**. Study note; interactive charts on the original page are not reproduced here.

Local figures (copyright remains with the original site; study copies):

![sleepmode](../../../../assets/vllm/blog/architecture/sleep-mode/01-sleepmode.png)

## Two sleep levels

- **Level 1 (default):** offload **weights → CPU RAM**, discard KV. Fastest wake. Needs enough RAM.
- **Level 2:** discard weights too; CPU keeps only small buffers (RoPE scaling tensors, …). Wake reloads weights from disk. Minimal RAM.

Both claimed **18–200×** vs full reload. Works with **TP / PP / EP**.

### Why this beats a “fast weight loader”

Even instant memcpy still pays cold-start tax. The post’s comparison:

| Cost | Fast weight loaders | Sleep Mode |
| --- | --- | --- |
| 1. VRAM load (copy weights to GPU) | Optimized | Preserved |
| 2. CUDA allocator init | Every time | Preserved |
| 3. CUDA graph capture | Every time | Preserved |
| 4. GPU kernel JIT (DeepGEMM, FlashInfer, TorchInductor) | Every time | Preserved after initial warmup |
| 5. Cache / first-request warm-up | Every time | Quick re-warm |

Keeping the process alive preserves **#2–4**. That is why first inference after wake is **61–88% faster** than a cold start — infrastructure, not memcpy.

What stays:

| Component | Sleep | Cold start |
| --- | --- | --- |
| Memory allocator (`CuMemAllocator`) | Yes | Reinit |
| CUDA graphs | Yes | Recapture |
| Process state (Python, CUDA context) | Yes | Restart |
| GPU kernel JIT cache | Yes, after initial warmup | Recompile |

Without Sleep, the process dies on unload: you **cannot** pre-warm. First inference **4–7× slower** in the cited example (**0.92 s** wake vs **3.72 s** cold). With Sleep, first inference stays ~**1 s**, skipping a **3–4 s** cold penalty. Timing varies a lot by model, GPU, config — see the warm-up ablation (**5–7×** slower without warm-up).

## Quick start (then-current API)

Admin endpoints need `VLLM_SERVER_DEV_MODE=1`. Trusted networks only: `/sleep`, `/wake_up`, `/collective_rpc`, `/reset_prefix_cache` can disrupt service (training clusters / backend apps, not the public internet).

```bash
# Terminal 1
export VLLM_SERVER_DEV_MODE=1
vllm serve microsoft/Phi-3-vision-128k-instruct --enable-sleep-mode --port 8001

# Terminal 2
export VLLM_SERVER_DEV_MODE=1
vllm serve Qwen/Qwen3-0.6B --enable-sleep-mode --port 8002
```

Level 2 cycle (Phi-3-vision):

```bash
curl -X POST 'localhost:8001/sleep?level=2'
curl -X POST 'localhost:8002/sleep?level=2'

curl -X POST 'localhost:8001/wake_up'
curl -X POST 'localhost:8001/collective_rpc' \
  -H 'Content-Type: application/json' \
  -d '{"method":"reload_weights"}'
curl -X POST 'localhost:8001/reset_prefix_cache'
# inference …
curl -X POST 'localhost:8001/sleep?level=2'

curl -X POST 'localhost:8002/wake_up'
# Level 1: no reload_weights / reset_prefix_cache
```

**Level 2 must** `reload_weights` and `reset_prefix_cache` after wake. **Level 1 does not.**

## Benchmarks (vLLM 0.11.0, `cudagraph_mode: FULL_AND_PIECEWISE`)

Workload shape unless noted: **5 model switches** — A infer → switch to B → B infer → repeat (A→B→A→B→A→B). Inference time = Prefill + Decode of the **first request** after wake/load; different questions to avoid cache; output capped at **100 tokens**. Error bars on the original are min/max across runs.

### A100, large pair, Level 1

**Model A:** Qwen3-235B-A22B-Instruct-2507-FP8 (**TP=4**). **Model B:** Qwen3-Coder-30B-A3B-Instruct (**TP=1**). GPU: A100. Sleep Level 1.

The original page’s interactive totals for the five-switch workload are charts, not a typed table. Prose claims: waking is **18–20×** faster than a fresh vLLM load. Inference after wake vs cold start is the **61–88%** band above.

### A4000, small pair, Level 1

**Model A:** Qwen3-0.6B. **Model B:** Phi-3-vision-128k-instruct. A4000, **TP=1**, Level 1.

- Inference: wake **83%** faster for Qwen3-0.6B, **81%** faster for Phi-3-vision.
- Switching: wake **~0.1–0.8 s**, **58–203×** vs cold start.
- Five switches: **85 s vs 226 s** (**62%** total savings).
- Near-instant for small models (~**0.1 s** wake).

### Level 1 vs Level 2 vs no sleep (A100, same small pair, TP=1)

Wake-time bands from the level write-up:

- L1: ~**0.1–0.8 s** small models, ~**3–6 s** large.
- L2: ~**0.8–2.6 s** small, because of disk reload.

| Mode | Total time (5 switches) | Wake (A / B) | CPU RAM | Best for |
| --- | --- | --- | --- | --- |
| **No Sleep** | **357.1 s** | N/A (full reload) | Minimal | Single model |
| **Level 1** | **112.6 s** | **0.26 s / 0.82 s** | High (~GB per model) | Frequent switching, ample RAM |
| **Level 2** | **124.6 s** | **0.85 s / 2.58 s** | Minimal (~MB per model) | Tight RAM, many models |

- L1 **68%** faster than no sleep; L2 **65%**.
- L2 wake ~**3×** slower than L1 (0.85 s vs 0.26 s on Qwen3-0.6B).

L2 still reloads weights from SSD, same as no-sleep — so why **23–45×** faster switches?

| Cost | Level 2 | No Sleep |
| --- | --- | --- |
| 1. Weight load (SSD → VRAM) | Pay | Pay |
| 2. Process init | Skipped | Pay |
| 3. Allocator setup | Skipped | Pay |
| 4. CUDA graph capture | Skipped | Pay |
| 5. Kernel JIT | Preserved (already compiled) | Full compile + `kernel_warmup()` + dummy runs |

Five switches: L2 **124.6 s** (~**2.6 s**/switch) vs no sleep **357.1 s** (~**48 s**/switch) → **2.9×** overall even though both hit the disk.

Level 2 vs no sleep, same A100 small pair:

| Metric | No Sleep | Level 2 | Improvement |
| --- | --- | --- | --- |
| Total (5 switches) | 357.1 s | 124.6 s | **65%** faster |
| Qwen3-0.6B switch | 37.6 s avg | 0.85 s avg | **45×** |
| Phi-3-vision switch | 58.1 s avg | 2.58 s avg | **23×** |
| Qwen3-0.6B inference | 3.67 s avg | 0.53 s avg | **86%** faster |
| Phi-3-vision inference | 6.30 s avg | 0.76 s avg | **88%** faster |
| Wake vs Level 1 | — | 3–10× slower | RAM vs speed |

L1 vs L2 one-liners: L1 ~**0.1–0.8 s** wake, ~**10–100 GB+** CPU RAM per model; L2 ~**0.8–2.6 s**, ~**MB** RAM; both vs full reload ~**20–100 s**.

## Ablations (A100, TP=1, Level 1, same small pair)

### Warm-up

Warm-up pre-compiles CUDA graphs on initial load. Compare with vs without.

| Metric | With warm-up | Without | Difference |
| --- | --- | --- | --- |
| Initial load | **108.7 s** (includes **8.4 s** warm-up) | **101.1 s** | **7.6 s** saved up front |
| First inference (A) | **0.45 s** | **2.59 s** | **5.8×** slower without |
| First inference (B) | **0.93 s** | **6.61 s** | **7.1×** slower without |
| Subsequent inferences | 0.43 s avg | 0.41 s avg | None |
| Total (5 switches) | **119.5 s** | **119.0 s** | Nearly identical |

Insights they draw: JIT + CUDA graphs paid **once** on load, then preserved across sleep/wake. **Without** warm-up, the **5–7×** hit is on the **first inference after every wake**, not once. A single **1-token** inference is enough to trigger full JIT + graph capture. Recommendation in the post: **always warm up** in production if you care about consistent first-token latency. Totals look similar because the 8.4 s is amortized; the *shape* of latency is not.

### FP8 vs BF16

Same workload, Level 1.

| Metric | BF16 | FP8 | Change |
| --- | --- | --- | --- |
| Total (5 switches) | **108.2 s** | **113.6 s** | **−5%** (slightly slower) |
| Qwen3-0.6B wake | 0.27 s avg | 0.18 s avg | **33%** faster |
| Phi-3-vision wake | 0.90 s avg | 0.78 s avg | **13%** faster |
| Qwen3-0.6B inference | 0.41 s avg | 0.44 s avg | **−7%** |
| Phi-3-vision inference | 0.81 s avg | 0.57 s avg | **30%** faster |
| Initial load | **90.5 s** | **96.9 s** | **−7%** (longer warmup) |

FP8 moves less memory on wake; helps inference more on the larger of the two models; longer initial load (quant during warmup). Frequent switching can still prefer FP8’s cheaper wakes.

## Decision guide (from the post)

**Level 1:** enough CPU RAM for all weights; fastest wake (**0.1–6 s**); switches every few seconds/minutes; inference latency must stay consistent.

**Level 2:** RAM cannot hold all weights; cheaper cloud boxes; **10+** models.

**Skip Sleep:** one model; switches once a day/week; both models already fit in VRAM together.

## Close

Headline recap: **18–200×** switching; **61–88%** faster first inference vs cold; **65–68%** total time on the small-pair A100 workload; range **0.6B–235B**, A4000–A100.

Acknowledgements in the post: Vensen Mu, Jeff Aw, Jun Kang Chow, Tun Jian Tan, Pin Siang Tan, Amir Balwel, Ye Hur Cheong, Zhiyao Cen, Kaichao You.

[torch.compile](torch-compile.md) made startup expensive. Sleep refuses to kill the process. Dense LoRA (AIBrix-style) is the other multi-model path; this one swaps **whole weights**.
