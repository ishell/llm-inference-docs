---
source: https://vllm.ai/blog/2026-01-08-kv-offloading-connector
lang: en
fetched: 2026-09-05
---

# Inside vLLM’s New KV Offloading Connector

Chinese: [zh/vllm/blog/serving/kv-offload.md](../../../../zh/vllm/blog/serving/kv-offload.md)  
Source: https://vllm.ai/blog/2026-01-08-kv-offloading-connector

2026-01-08. Or Ozeri, Danny Harnik (vLLM team at IBM Research). Study extract, not an official reprint. Landed in **vLLM 0.11.0**; **0.12.0** jumped after a contiguous physical-block layout. Same **KVConnector** door as [mooncake.md](mooncake.md): local DRAM vs cluster pool. Focus: **CPU DRAM** offload and host↔device copy.

Local figures (copyright remains with the original site; study copies):

![figure1](../../../../assets/vllm/blog/serving/kv-offload/01-figure1.png)

**Figure 1.** Single request TTFT (Llama-3.1-8B-Instruct, NVIDIA H100).

![figure2](../../../../assets/vllm/blog/serving/kv-offload/02-figure2.png)

**Figure 2.** Concurrent requests throughput (Llama-3.1-8B-Instruct, NVIDIA H100, 10000 prefill requests of 512 tokens).

![figure3](../../../../assets/vllm/blog/serving/kv-offload/03-figure3.png)

**Figure 3.** Single GPU → CPU transfer throughput (NVIDIA H100, single transfer of 1000 blocks).

![figure4](../../../../assets/vllm/blog/serving/kv-offload/04-figure4.png)

**Figure 4.** Single CPU → GPU transfer throughput (NVIDIA H100, single transfer of 1000 blocks).

![figure5](../../../../assets/vllm/blog/serving/kv-offload/05-figure5.png)

**Figure 5.** Single request TTFT (Llama-3.2-1B-Instruct, NVIDIA H100).

![figure6](../../../../assets/vllm/blog/serving/kv-offload/06-figure6.png)

**Figure 6.** Concurrent requests throughput (Llama-3.2-1B-Instruct, NVIDIA H100, 10000 prefill requests of 512 tokens).

## Motivation

Serving an LLM is computationally heavy. At the core it computes KV blobs. The first step for a prompt is computing the KV for that prompt — Prefill. Prefill is expensive and wants an accelerator (GPU) to finish quickly.

KV computed for one prompt can be reused for other prompts that share the same prefix, so you skip recalculation. Caching and reusing KV usually buys two things:

- **Lower request latency** (if reading the cache is faster than recomputing KV)
- **Higher per-node throughput** (GPU cores free up, so more concurrent requests fit)

**KV cache offloading is still useful when requests share no prefix.** Under high concurrency the GPU can run out of room for the KV of in-flight requests. The engine may **preempt** a running request and discard its KV. When that request is rescheduled, KV must be recomputed. Offload to a larger tier (CPU DRAM) before preemption avoids that recompute.

### CPU Offloading

This post emphasizes offload to CPU memory (DRAM), for a stack of reasons:

- CPU RAM is widely available.
- Capacity typically exceeds GPU memory, so the KV cache can be larger.
- CPU↔GPU transfers are low-latency and high-throughput. Combined with capacity, CPU offload is **ideal for preemptions**.
- CPU RAM is also a **convenient staging area** toward external storage, especially when storage latency is high.

## The New Offloading Connector

### The vLLM Connector API

vLLM has long had an API to read and write KV, wired into the request lifecycle: the Connector API. Before handling a request, vLLM asks it whether KV can be imported from an external source. After computing new KV, it asks the API to store those values on an external target.

Originally the connector API was **synchronous**: while vLLM loaded or stored KV externally, the engine was blocked and no new batches could run. vLLM **0.9.0** added **asynchronous** load/store. The offloading connector uses that async API.

The **offloading connector** asynchronously offloads and loads KV. It exposes a pluggable backend API so any medium can be the offload target. Adding a backend is mostly writing a transfer function that copies KV between mediums.

It ships with a CPU backend for native CPU offload. The rest of the post discusses CPU only.

### Using the Offloading Connector

For CPU offload, add to `vllm serve`:

```
--kv_offloading_backend native --kv_offloading_size <size_in_GB>
```

This CLI assumes [PR #24498](https://github.com/vllm-project/vllm/pull/24498), hoped for **0.14.0**.

Older releases:

```
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks": <num_cpu_blocks>}}'
```

`num_cpu_blocks` is the number of CPU blocks for the CPU KV cache.

## Benefits of CPU Offloading via the Offloading Connector

Two micro-benchmarks. The first measures **single-request** Prefill TTFT: CPU cache load vs GPU recompute. The second measures **concurrent** throughput under heavier load.

First: one Prefill, CPU load vs GPU KV compute.

Figure 1: loading KV from CPU cuts TTFT by **2–22×**, depending on prompt size. Setup and code at the end of the post.

Offload latency (GPU → CPU) is **not user-facing**: offload is async, and the request can finish without waiting for that transfer. So **the connector barely affects TTFT on cache misses**.

Second: concurrency. Submit **10,000** unique requests (each **512** tokens) and measure throughput vs CPU cache hit rate. Time **excludes** CPU-cache warmup, then convert to token/s. GPU cache is off, so the effect is CPU hits.

Figure 2: throughput rises with CPU KV hit rate. Up to about **9×**, even though TTFT for that prompt size only dropped ~**2×**. The larger win is **throughput maximization**.

### vLLM versions of the Offloading Connector

**0.12.0** was a large jump. Example: Llama-3.1-8B-Instruct on NVIDIA H100 — up to **~4×** lower TTFT, **~5×** higher throughput. Details in the physical-block section below.

Further improvements hoped for **0.14.0** (included in this post’s eval):

- Preempted requests can load back from CPU ([PR #29870](https://github.com/vllm-project/vllm/pull/29870))
- Race between offload and model computation ([PR #31341](https://github.com/vllm-project/vllm/pull/31341))

## Evaluating GPU-CPU Transfer Techniques

The rest of the post is a design deep dive: maximize GPU–CPU throughput while minimizing GPU/CPU-core overhead, so inference throughput can rise.

A backend’s main piece is a **transfer function**. For the CPU backend that function copies GPU memory ↔ CPU memory. It then **supported CUDA-compatible devices** (NVIDIA and AMD).

The implementation uses `cudaMemcpyAsync` — the GPU’s **DMA** (Direct Memory Access). DMA is built for high-throughput device–host copies and barely taxes CPU or GPU cores. That matters because transfers run asynchronously with model compute.

DMA likes **large, physically contiguous** copies. Offload numbers therefore depend on KV layout: models with bigger KV blocks do better.

How fast is DMA versus a custom CUDA kernel?

They wrote [gpu_cpu_benchmark](https://github.com/orozery/playground/tree/kv-offloading-blog-dec-2025/kvcache/gpu_cpu_benchmark) and compared:

- **DMA** via `cudaMemcpyAsync`
- A **custom CUDA kernel** that copies 16-byte words with raw pointers. High parallelism, but more interference with the GPU cores’ main job.

First test: one transfer of **1000** blocks, sizes **4 KB to 16 MB**.

Figures 3 and 4: **DMA is good only for larger blocks**. For small blocks the custom kernel wins throughput, with more noise (higher variance).

Then bi-directional: one read and one write at once. Block size fixed at **2 MB**, vary the size ratio. Both peak when the two directions are roughly equal. One-way both reach about **50 GB/s**; two-way:

- DMA: **83.4 GB/s**
- Custom kernel: **68.5 GB/s**

Two remaining questions:

- **What effective block size does vLLM use?** Depends on the model and config. Next section answers for common models of the day.
- **How does each approach affect GPU model compute?** The connector is meant to run in parallel with the forward pass. The eval shows the throughput impact.

## Changing vLLM’s Memory Layout

This section is the GPU layout change: better KV transfers without hurting compute.

Start from the default KV layout and the fragment size copied on offload — that is the effective physical block for KV transfer in vLLM.

vLLM allocates GPU memory in token blocks, default **16** tokens per block. Physical layout depends on the attention backend (FlashAttention, FlashInfer, …) and the model. Most common models today are uniform: many layers, each with its own KV cache of the same shape. vLLM also supports hybrid models, which were **not yet optimized** for this connector. For uniform models each layer gets its own KV cache, so one logical block fragments into `num_layers` pieces. Some attention backends split further into K and V.

That fragmentation is meaningless for compute and devastating for offload: effective copies shrink. They [upstreamed](https://github.com/vllm-project/vllm/pull/27743) a layout with **one contiguous physical block across all layers**. Effective physical block size grew by about **`2 × num_layers`**, and offloading-connector throughput jumped **an order of magnitude**.

Common models, old (0.11.0) vs new (0.12.0) physical block size (16-token blocks unless noted):

| Model | Old block size | New block size |
| :---- | :---- | :---- |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B (`tensor_parallel_size=2`) | 16 KB | 2 MB |
| deepseek-ai/DeepSeek-V2-Lite-Chat (GPU block size=64) | 72 KB | 1.9 MB |
| meta-llama/Llama-3.1-8B-Instruct | 32 KB | 2 MB |
| meta-llama/Llama-3.2-1B-Instruct | 16 KB | 0.5 MB |
| meta-llama/Llama-3.1-70B-Instruct | 8 KB | 1.25 MB |
| mistralai/Mistral-7B-Instruct-v0.2 | 32 KB | 2 MB |
| mistralai/Mistral-Small-24B-Instruct-2501 | 32 KB | 2.5 MB |
| Qwen/Qwen2.5-3B-Instruct | 8 KB | 0.56 MB |
| Qwen/Qwen3-0.6B | 32 KB | 1.75 MB |
| Qwen/Qwen2.5-7B-Instruct | 16 KB | 0.87 MB |
| Qwen/Qwen3-4B-Instruct-2507 | 32 KB | 2.25 MB |
| Qwen/Qwen2.5-1.5B-Instruct | 8 KB | 0.44 MB |
| Qwen/Qwen3-8B | 28 KB | 1.97 MB |
| Qwen/Qwen3-1.7B | 32 KB | 1.75 MB |
| Qwen/Qwen3-32B (`tensor_parallel_size=2`) | 16 KB | 2 MB |

The new layout is typically **0.5–2 MB**; the old one was a few KB. Combined with the microbench, DMA should be comparable or only slightly behind the custom kernel, depending on the model.

## End-to-end Evaluation of Copy Methods

The two vLLM micro-benchmarks compare two connector variants:

- Upstream: DMA transfer function
- Patched: the custom kernel from the GPU–CPU microbench

They deliberately pick the **worst case for DMA**: a model with a **0.5 MB** physical block.

Figure 5, single-request TTFT: custom kernel **slightly** better — under **1 ms** at a 1K prompt, up to **~15 ms** at 90K. Matches the 0.5 MB microbench. Larger-block models: roughly a tie.

Figure 6, concurrent 10k×512 Prefills: **DMA higher throughput**. About **5.5%** at 0% hit, about **15%** at 80% hit.

Why: the custom kernel fights model compute for GPU cores. At **0%** hit the custom kernel is **~6% worse than no CPU offload at all**. At **100%** hit there is no parallel compute, so the gap shrinks.

They stress this is the worst-case model for DMA. Common models have larger physical blocks and favor DMA more. On **Llama-3.1-8B-Instruct**, DMA gained up to **32%** more throughput than the kernel while matching TTFT.

Summary: the GPU layout change exists so DMA can do KV transfers and raise end-to-end throughput.

## Evaluation Setup and Benchmark Code

Eval setup:

- Single Ubuntu 24.04.1 LTS container
- Kernel 5.14.0-427.81.1.el9_4.x86_64
- Intel Xeon Sapphire Rapids 2.1 GHz (**8** cores limit)
- NVIDIA H100 80GB HBM3
- 500 GB DRAM
- CUDA 12.9
- vLLM commit `2a1776b7ac4fae7c50c694edeafc1b14270e4350`
- Flash Attention backend
- GPU prefix caching **off** (evaluate CPU hits)
- GPU block size 16 tokens
- CPU block size 16 tokens
- De/tokenization **off**

Code: [kv_offload_benchmark.py](https://github.com/orozery/playground/blob/kv-offloading-blog-dec-2025/kvcache/kv_offload_benchmark.py).

### What's Next?

Next milestone: CPU KV cache as an intermediate tier for storage offload.

Correctness and performance remain the top priorities. Try it, share numbers, report issues.

Discussion: `#feat-v1-cpu-offloading` on [vLLM Slack](https://vllm-dev.slack.com/archives/C09AYJFFLKD).
