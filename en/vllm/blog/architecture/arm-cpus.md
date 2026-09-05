---
source: https://vllm.ai/blog/2026-07-29-optimizing-vllm-on-arm-cpus
lang: en
fetched: 2026-09-04
---

# Optimizing vLLM on Arm CPUs

Chinese: [zh/vllm/blog/architecture/arm-cpus.md](../../../../zh/vllm/blog/architecture/arm-cpus.md)

2026-07-29. **Arm Team**. Study note; benches vs an October 2025 BF16 baseline on Neoverse, not your SLA. Hardware-out-of-core: [hardware-plugin.md](hardware-plugin.md). PagedAttention itself: [paged-attention.md](paged-attention.md). INT8 / W4A16 cousins: [autoround-llmc.md](autoround-llmc.md). CPU vs Arc XPU: [intel-arc.md](intel-arc.md).

Fits: Arm Neoverse servers that want wheels, chunked prefill / prefix cache, and INT8 W8A8 / W4A8 on vLLM. Does not fit: treating the page’s **2.7–6.2×** as a promise — allocator gains are **excluded** from the heatmaps.

## Overview

CPU serving is cheaper infrastructure, broadly available. As Neoverse servers spread, open-source serving on Arm has to be usable, feature-complete, and fast. Months of upstream work with vLLM, PyTorch, oneDNN, and KleidiAI. This post: enablement first, then the performance stack.

Dense GEMMs were already near hardware efficiency (~**80%** of runtime). The rest was allocator, OpenMP, layout, attention, quantization.

## Enablement

- Pre-built [wheels](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/#arm-aarch64_2:~:text=venv/bin/activate-,Pre%2Dbuilt%20wheels,%C2%B6,-When%20specifying%20the) and [Docker images](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/#arm-aarch64_4:~:text=%C2%B6-,Pre%2Dbuilt%20images,%C2%B6,-Intel/AMD%20x86)
- Fixes for crashes, accuracy, threading, CPU utilization
- Chunked prefill and prefix caching
- INT8 W8A8 and INT8 W4A8
- GPT-OSS, Whisper, Qwen 3.5 / 3.6
- Tighter [PyTorch](https://github.com/pytorch/pytorch) and [UXL](https://github.com/uxlfoundation) integration

## Performance

October 2025 first benches were far below what the GEMM kernels suggested. Standalone BF16 GEMMs were already close to expected efficiency, so kernel-only work would not move the needle. Profiles pointed at allocator behavior, runtime sync, framework overhead, attention, quantized execution.

### Memory allocation

Prefill and Decode repeatedly allocate/free tensors for scheduling, KV, intermediates. Poor reuse of large allocations → page faults. Root cause: PyTorch’s glibc `malloc`. Large blocks were not reused; alloc/free contended as thread counts rose. Early workaround: preload a caching allocator — extra setup, performance tied to runtime config.

Fix: [mimalloc](https://github.com/microsoft/mimalloc) as PyTorch’s **default** allocator on Arm. Caching allocator, scales under multi-threaded pressure; already a PyTorch dependency on non-Arm Linux; strong on TorchBench.

Llama 3.1 8B out-of-the-box: offline throughput ~**2.3×**; low-concurrency serving ~**7×**.

> Allocator gains are **excluded** from all plots in the post — they would dominate the scale.

### Synchronization at high core counts

Beyond a point, more cores did not help and could regress. One profile: **74%** of paged-attention time in OpenMP dynamic scheduling:

```text
97.94% gomp_thread_start
  90.08% paged_attention_v1_impl
    74.07% gomp_iter_dynamic_next
     7.00% reduceValueBlock::lambda(int)
```

`gomp_iter_dynamic_next` is libgomp’s dynamic loop scheduling: atomic fetch-add to hand chunks to workers. The libgomp in PyTorch wheels used load-linked / store-conditional retries:

```c
for (;;) {
    long old = LDXR(p);
    long newv = old + delta;
    int fail = STLXR(p, newv);
    if (fail == 0) {
        DMB_ISH();
        return old;
    }
}
```

High core counts: many workers contend on one atomic → failed stores and retry traffic. The bench box was Neoverse V2, which has [Arm LSE](https://learn.arm.com/learning-paths/servers-and-cloud-computing/lse/example/) (`LDADDAL`). PyTorch’s OpenMP runtime did not use it.

Fix: a libgomp in PyTorch that uses LSE atomics on capable CPUs. Llama 3.1 8B: offline **+9%**; low-concurrency TPOT **−15%**.

### Dense-layer layout overhead

High-performance GEMM wants blocked weight layout. Without prepack, each call transforms framework layout → kernel format. Expensive at low concurrency (not amortized). Fast oneDNN path for dense layers (Compute Library for Arm): pack BF16 weights at **warmup**, reuse at inference.

Llama 3.1 8B: offline **+16%**; low-concurrency TPOT **−60%**.

### Paged attention

CPU paged attention was not Arm-tuned. QK / PV matmuls and softmax exp fell back to reference. Prefill used PyTorch SDPA — so chunked prefill and prefix caching were off on the Arm CPU path.

QK / PV via custom GEMM with [BFMMLA](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/bfloat16-processing-for-neural-networks-on-armv8_2d00-a). Softmax exp: vectorized third-degree polynomial.

Kernel up to ~**4×**; Llama 3.1 8B offline **+12%**. Paged attention for prefill unlocked chunked prefill and prefix caching.

### BF16, after the three cuts

Sync + prepack + paged attention vs the October 2025 BF16 baseline:

![heatmap bf16 optimized vs bf16 baseline](../../../../assets/vllm/blog/architecture/arm-cpus/01-heatmap_bf16_optimized_vs_bf16_baseline.png)

**Figure.** Optimized BF16 serving relative to the October 2025 BF16 baseline (study copy).

### INT8 W8A8

INT8 weights vs BF16: less bandwidth, larger models in the same RAM. On Arm with I8MM, W8A8 maps to [`SMMLA`](https://developer.arm.com/documentation/dui0379/e/arm-and-thumb-instructions/smmla) — **2×** theoretical matmul throughput vs BF16.

W8A8 path: [oneDNN](https://github.com/uxlfoundation/oneDNN) JIT kernels using `SMMLA` on SVE128 and SVE256. Hugging Face checkpoints named: `RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8`, `RedHatAI/whisper-large-v3-quantized.w8a8`.

Vs optimized BF16 (per-token activation quant, channelwise weight quant): up to **+88%** throughput, **−45%** TPOT, **−54%** TTFT, depending on concurrency.

![heatmap int8 vs bf16 optimized](../../../../assets/vllm/blog/architecture/arm-cpus/02-heatmap_int8_vs_bf16_optimized.png)

**Figure.** INT8 W8A8 vs optimized BF16 (study copy).

Learning Path: [INT8 W8A8 on Arm](https://learn.arm.com/learning-paths/servers-and-cloud-computing/vllm-benchmark-quantisation/).

### INT8 W4A8

INT4 weights, still 8-bit activations — more bandwidth relief, especially at low concurrency. Accelerated by [KleidiAI](https://github.com/ARM-software/kleidiai) INT4 micro-kernels.

Vs the W8A8 baseline: up to **+29%** throughput, **−26%** TPOT, **−18%** TTFT. Largest at low concurrency (memory-bound).

![heatmap int4 vs int8](../../../../assets/vllm/blog/architecture/arm-cpus/03-heatmap_int4_vs_int8.png)

**Figure.** INT8 W4A8 vs INT8 W8A8 (study copy).

How to quantize: [llm-compressor INT8 W4A8](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/int8_w4a8/).

## Summary

Vs October 2025 BF16:

- Optimized BF16: up to **2.7×** serving throughput
- INT8 W8A8: up to **4.8×** throughput, **5.7×** TPOT
- INT8 W4A8: up to **6.2×** throughput, **7.8×** TPOT, **2.6×** TTFT

Gains from the full CPU stack: allocation, OpenMP, dense prepack, paged attention, quantization — not GEMM kernels alone.

![bars all vs bf16 baseline](../../../../assets/vllm/blog/architecture/arm-cpus/04-bars_all_vs_bf16_baseline.png)

**Figure.** Serving speedups for optimized BF16, INT8 W8A8, and INT8 W4A8 vs October 2025 BF16 (study copy).

Also claimed: broader features, better out-of-the-box usability, upstream integration, more models — a more production-ready Neoverse stack.

## Acknowledgements

vLLM community. **[Li Jiang](https://github.com/bigPYJ1151)** (Intel) for the CPU backend. **[Sanket Kale](https://github.com/sanketkaleoss)** (Fujitsu) for initial Arm CPU enablement. **[Shreyas](https://github.com/Shreyas-fuj)** (Fujitsu) for SVE256 INT8 kernels in oneDNN.

Arm / PyTorch / Intel / oneDNN are trademarks of their owners. Post copyright 2026 Arm Limited (`open-source-office@arm.com`).
