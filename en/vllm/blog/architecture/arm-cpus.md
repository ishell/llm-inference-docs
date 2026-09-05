---
source: https://vllm.ai/blog/2026-07-29-optimizing-vllm-on-arm-cpus
lang: en
fetched: 2026-09-05
---

# Optimizing vLLM on Arm CPUs

Chinese: [zh/vllm/blog/architecture/arm-cpus.md](../../../../zh/vllm/blog/architecture/arm-cpus.md)  
Source: https://vllm.ai/blog/2026-07-29-optimizing-vllm-on-arm-cpus

2026-07-29. **Arm Team**. Study extract, not an official reprint. Benches vs an October 2025 BF16 baseline on Neoverse, not your SLA. Hardware-out-of-core: [hardware-plugin.md](hardware-plugin.md). PagedAttention itself: [paged-attention.md](paged-attention.md). INT8 / W4A16 cousins: [autoround-llmc.md](autoround-llmc.md). CPU vs Arc XPU: [intel-arc.md](intel-arc.md).

Fits: Arm Neoverse servers that want wheels, chunked prefill / prefix cache, and INT8 W8A8 / W4A8 on vLLM. Does not fit: treating the page’s **2.7–6.2×** as a promise — allocator gains are **excluded** from the heatmaps.

## Introduction

LLM serving on CPUs is a real deployment path: lower cost, simpler infrastructure, broad availability in cloud and enterprise data centers. As Arm® Neoverse™-based servers spread, open-source serving such as vLLM on Arm CPUs has to be usable, feature-complete, and fast.

Months of upstream work with the vLLM, PyTorch, oneDNN, and KleidiAI communities. Result: better usability, broader model and feature support, and performance gains that any Neoverse server running vLLM can take.

This post: enablement and coverage first, then the main optimizations and end-to-end serving numbers.

## Enablement

Alongside performance, they improved usability and feature completeness of vLLM on Arm® CPUs so deployment on Arm® servers is easier.

Key enablement:

- Pre-built [wheels](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/#arm-aarch64_2:~:text=venv/bin/activate-,Pre%2Dbuilt%20wheels,%C2%B6,-When%20specifying%20the) and [Docker images](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/#arm-aarch64_4:~:text=%C2%B6-,Pre%2Dbuilt%20images,%C2%B6,-Intel/AMD%20x86)
- Fixes for crashes, accuracy, threading, CPU utilization
- Chunked prefill and prefix caching
- INT8 W8A8 and INT8 W4A8
- GPT-OSS, Whisper, Qwen 3.5 / 3.6
- Tighter [PyTorch](https://github.com/pytorch/pytorch) and [UXL](https://github.com/uxlfoundation) integration

With enablement in place, they turned to finding and removing performance bottlenecks.

## Performance Improvements

First benches on Arm CPUs in October 2025 were much lower than expected, even though roughly **80%** of model runtime was in dense layers dispatched to highly optimized BF16 GEMMs. Standalone GEMM kernels behind those layers were already close to expected hardware efficiency, so the biggest gains were unlikely to come from GEMM kernels alone.

Profiles pointed at a broader problem: allocator behavior, runtime synchronization, framework overheads, attention kernels, quantized execution.

### Memory Allocation

LLM serving puts heavy pressure on the CPU allocator. During prefill and decode, vLLM repeatedly allocates and releases tensors for scheduling, KV-cache management, and intermediate operator outputs. In the first benches, allocation was a bottleneck: poor reuse of large allocations, many page faults.

Root cause: PyTorch’s glibc `malloc`. Large allocations were not reused well across repeated inference steps; alloc/free contended as thread counts rose. Early workaround: preload a caching allocator — extra manual setup, performance tied to runtime config.

For out-of-the-box speed, they made [mimalloc](https://github.com/microsoft/mimalloc) PyTorch’s **default** allocator on Arm CPUs. Mimalloc is a caching allocator designed to scale under multi-threaded allocation pressure. Chosen because it was strong across a broad TorchBench set and already a PyTorch dependency on non-Arm Linux builds.

Llama 3.1 8B out of the box: offline throughput **2.3×**; about **7×** in low-concurrency serving.

> Allocator gains are **excluded** from every performance plot in the post — they would dominate the scale and hide the other optimizations. Plots therefore show the rest of the stack.

### Synchronization at High Core Counts

After allocation, the next bottleneck appeared when scaling to higher core counts. Beyond a point, more cores did not help and could regress.

They profiled individual layers at high thread counts. One profile: **74%** of paged-attention time in OpenMP dynamic scheduling:

```text
97.94% gomp_thread_start
  90.08% paged_attention_v1_impl
    74.07% gomp_iter_dynamic_next
     7.00% reduceValueBlock::lambda(int)
```

`gomp_iter_dynamic_next` is libgomp’s dynamic loop-scheduling path. The runtime uses an atomic fetch-add to assign loop chunks to workers. The libgomp in PyTorch wheels implemented that atomic update as a load-linked / store-conditional retry loop:

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

High core counts: many workers contend on the same atomic → repeated failed stores and retry traffic.

Tracing to assembly showed a missed hardware opportunity. The bench box was Neoverse™ V2, which supports [Arm Large System Extensions (LSE)](https://learn.arm.com/learning-paths/servers-and-cloud-computing/lse/example/). LSE provides hardware atomics such as `LDADDAL` that replace that inefficient loop. PyTorch’s OpenMP runtime did not use LSE atomics.

Fix: a libgomp in PyTorch that uses LSE atomics on capable CPUs.

Llama 3.1 8B: offline throughput **+9%**; low-concurrency serving TPOT **−15%**.

### Dense-Layer Layout Overhead

After allocator and runtime, dense layers still left performance on the table. High-performance GEMM is sensitive to weight layout: efficient runs need a blocked format matching the kernel’s vectorization and cache access. Without prepacking, each call can pay to transform framework tensor layout into kernel format.

Especially expensive at low concurrency: packing is not amortized over large batches. Fix: a fast oneDNN path for dense layers, accelerated by the Compute Library for Arm Architecture. vLLM packs BF16 weights at model **warmup** into the kernel format, then reuses that packed representation at inference.

Llama 3.1 8B: offline throughput **+16%**; low-concurrency TPOT **−60%**.

### Paged Attention

The CPU paged-attention kernel was not optimized for Arm CPUs. QK and PV matmuls, and the exponential in softmax, fell back to reference implementations. Prefill therefore used PyTorch Scaled Dot-Product Attention — so chunked prefill and prefix caching were off on the Arm CPU path.

QK / PV via custom GEMM using Arm [BFMMLA](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/bfloat16-processing-for-neural-networks-on-armv8_2d00-a) Advanced SIMD. Softmax exponential: a fast vectorized third-degree polynomial approximation.

Paged attention up to ~**4×**; Llama 3.1 8B offline throughput **+12%**. Prefill on Arm CPUs could then use paged attention, unlocking chunked prefill and prefix caching.

### BF16 Performance Improvements

Synchronization, weight prepacking, and paged attention together make a stronger BF16 serving baseline than October 2025.

![heatmap bf16 optimized vs bf16 baseline](../../../../assets/vllm/blog/architecture/arm-cpus/01-heatmap_bf16_optimized_vs_bf16_baseline.png)

**Figure.** Optimized BF16 serving relative to the October 2025 BF16 baseline (study copy).

### INT8 W8A8 (8-bit weights and activations)

LLM inference repeatedly reads large weight matrices in prefill and decode. INT8 weights instead of BF16 cut bandwidth pressure and can fit larger models in the same memory budget.

On Arm CPUs with I8MM, W8A8 also maps to [`SMMLA`](https://developer.arm.com/documentation/dui0379/e/arm-and-thumb-instructions/smmla) — Arm’s signed INT8 matrix multiply-accumulate — **2×** theoretical matmul throughput vs BF16.

They accelerated the W8A8 path with [oneDNN](https://github.com/uxlfoundation/oneDNN) JIT kernels using `SMMLA` on SVE128 and SVE256.

Multiple Hugging Face INT8 W8A8 checkpoints then perform well out of the box, including `RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8` and `RedHatAI/whisper-large-v3-quantized.w8a8`.

Vs the optimized BF16 baseline, W8A8 with per-token activation quantization and channelwise weight quantization: up to **+88%** throughput, **−45%** TPOT, **−54%** TTFT, depending on concurrency.

![heatmap int8 vs bf16 optimized](../../../../assets/vllm/blog/architecture/arm-cpus/02-heatmap_int8_vs_bf16_optimized.png)

**Figure.** INT8 W8A8 serving relative to the optimized BF16 path (study copy).

> INT8 W8A8 on Arm CPUs: [this Arm Learning Path](https://learn.arm.com/learning-paths/servers-and-cloud-computing/vllm-benchmark-quantisation/).

### INT8 W4A8 (4-bit weights, 8-bit activations)

W4A8 pushes the same idea: INT4 weights, still less bandwidth at inference. Especially useful at low concurrency, where there is less batching to amortize reading weights.

Accelerated through [KleidiAI](https://github.com/ARM-software/kleidiai) INT4 micro-kernels.

Vs the W8A8 baseline above, same per-token activation quant and channelwise weight quant: up to **+29%** throughput, **−26%** TPOT, **−18%** TTFT, depending on concurrency.

Largest W4A8 speedups at low concurrency, where inference is mostly memory-bound — as expected.

![heatmap int4 vs int8](../../../../assets/vllm/blog/architecture/arm-cpus/03-heatmap_int4_vs_int8.png)

**Figure.** INT8 W4A8 serving relative to the INT8 W8A8 path (study copy).

> How to quantize to INT8 W4A8 with llm-compressor: [these docs](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/int8_w4a8/).

## Summary

vLLM on Arm CPUs saw large gains in usability, robustness, model and feature coverage, and performance.

Relative to the October 2025 BF16 baseline:

- Optimized BF16: up to **2.7×** serving throughput
- INT8 W8A8: up to **4.8×** throughput and a **5.7×** TPOT speedup
- INT8 W4A8: best — up to **6.2×** throughput, **7.8×** TPOT, **2.6×** TTFT

Gains from the full CPU inference stack: memory allocation, OpenMP synchronization, dense-layer prepacking, paged attention, quantization — not GEMM kernels alone.

![bars all vs bf16 baseline](../../../../assets/vllm/blog/architecture/arm-cpus/04-bars_all_vs_bf16_baseline.png)

**Figure.** Serving speedups for optimized BF16, INT8 W8A8, and INT8 W4A8 vs October 2025 BF16 (study copy).

Beyond measured performance: broader features, better out-of-the-box usability, upstream integration, more models — a more production-ready Neoverse inference stack.

## Acknowledgements

Thanks to the vLLM community for continued support and collaboration.

Special thanks to **[Li Jiang](https://github.com/bigPYJ1151)** (Intel®) for maintaining the vLLM CPU backend and implementing much of the infrastructure this work builds on. Also **[Sanket Kale](https://github.com/sanketkaleoss)** (Fujitsu) for initial Arm CPU enablement in vLLM, and **[Shreyas](https://github.com/Shreyas-fuj)** (Fujitsu) for SVE256 INT8 kernels in oneDNN.

Arm is a registered trademark of Arm Limited (or its subsidiaries or affiliates). PyTorch is a trademark of The Linux Foundation. Intel and oneDNN are trademarks of Intel Corporation or its subsidiaries. Post copyright 2026 Arm Limited and/or its affiliates (`open-source-office@arm.com`).
