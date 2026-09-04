---
source: https://vllm.ai/blog/2026-01-08-kv-offloading-connector
lang: en
fetched: 2026-09-04
---

# Inside vLLM’s New KV Offloading Connector

Chinese: [zh/vllm/blog/serving/kv-offload.md](../../../../zh/vllm/blog/serving/kv-offload.md)

2026-01-08. Landed in **vLLM 0.11.0**; **0.12.0** jumped after a contiguous physical-block layout. Same **KVConnector** door as [mooncake.md](mooncake.md): local DRAM vs cluster pool. Focus of the post: **CPU DRAM** offload and host↔device copy. Study note.

Local figures (copyright remains with the original site; study copies):

![figure1](../../../../assets/vllm/blog/serving/kv-offload/01-figure1.png)

![figure2](../../../../assets/vllm/blog/serving/kv-offload/02-figure2.png)

![figure3](../../../../assets/vllm/blog/serving/kv-offload/03-figure3.png)

![figure4](../../../../assets/vllm/blog/serving/kv-offload/04-figure4.png)

![figure5](../../../../assets/vllm/blog/serving/kv-offload/05-figure5.png)

![figure6](../../../../assets/vllm/blog/serving/kv-offload/06-figure6.png)

## Motivation

Prefill computes KV for the prompt — expensive, wants an accelerator. Shared prefixes can reuse that KV: **lower latency** (if load &lt; recompute) and **higher per-node throughput** (GPU cores free for more concurrent work).

Offload is still useful when **no prefix is shared**. GPU KV fills; the engine **preempts**, drops KV, later **RECOMPUTE**s. Staging to a larger tier (CPU DRAM) before preemption avoids that recompute.

## Why CPU

- RAM is widely available
- Capacity typically **> HBM** → larger KV cache
- CPU↔GPU is relatively **low latency / high throughput** vs external storage — good for **preemption**
- Convenient **staging tier** toward disk when storage latency is high

## Connector API

vLLM already queried a Connector before handling a request (import KV) and after computing KV (store). Originally **synchronous**: load/store blocked the engine; no parallel batches. **0.9.0** added **async** load/store. The offloading connector uses that.

Pluggable **backend** API: implement a transfer function that copies KV between mediums. Bundled **CPU backend**. Post discusses CPU only. Transfer function then **CUDA-compatible** (NVIDIA and AMD).

## CLI

Newer flags (the post assumes [PR #24498](https://github.com/vllm-project/vllm/pull/24498), hoped for **0.14.0**):

```bash
--kv_offloading_backend native --kv_offloading_size <size_in_GB>
```

Older releases:

```bash
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks": <num_cpu_blocks>}}'
```

`num_cpu_blocks` is the CPU KV cache size in blocks.

## Benefits (Llama-3.1-8B-Instruct, NVIDIA H100)

**Single Prefill TTFT** (Figure 1): load KV from CPU vs GPU recompute. **2–22×** lower TTFT depending on prompt size. Offload GPU→CPU is async and not on the user-visible path — **miss TTFT almost unaffected**.

**Throughput, 10,000 unique 512-token Prefills**, GPU prefix cache **off** (Figure 2): time excludes CPU-cache warmup; throughput in token/s vs CPU hit rate. Up to about **9×** throughput even though TTFT for that prompt size only dropped ~**2×**. The larger win is **throughput**, not single-request latency.

### Versions

**0.12.0** was a large jump (physical block size — below). Same model/GPU: up to **~4×** TTFT reduction, **~5×** throughput.

Hoped for **0.14.0** (included in this post’s eval):

- Reload preempted requests from CPU — [PR #29870](https://github.com/vllm-project/vllm/pull/29870)
- Race between offload and compute — [PR #31341](https://github.com/vllm-project/vllm/pull/31341)

## GPU↔CPU copy: DMA vs custom kernel

The CPU backend’s transfer is `cudaMemcpyAsync` — **DMA**, little SM/CPU tax, overlaps with forward. DMA likes **large contiguous** copies; layout (block size) matters.

They also micro-benched a **custom CUDA kernel** that copies 16-byte words with raw pointers (lots of parallelism, **fights compute SMs**). Code: [gpu_cpu_benchmark](https://github.com/orozery/playground/tree/kv-offloading-blog-dec-2025/kvcache/gpu_cpu_benchmark).

**Single transfer of 1000 blocks**, sizes **4 KB–16 MB**, H100:

- Figure 3: GPU→CPU. DMA wins on **large** blocks; custom kernel wins on **small**. Kernel noisier (higher variance).
- Figure 4: CPU→GPU. Same shape.

**Bi-directional**, block size fixed **2 MB**, vary read/write ratio. Peak when both directions are roughly equal. One-way both ~**50 GB/s**; two-way:

- DMA **83.4 GB/s**
- Custom kernel **68.5 GB/s**

Decision then depends on vLLM’s **effective physical block size**, and on interference with the forward pass.

## Memory layout change

Default: **16 tokens** per logical block. Physical layout depends on attention backend (FlashAttention, FlashInfer, …) and model. Common **uniform** models: one KV cache per layer, same shape. **Hybrid** models were **not yet optimized** for this connector (then-current).

Per-layer blocks can split further into K and V. Fine for compute; **devastating** for DMA — effective copies of a few **KB**. They [upstreamed](https://github.com/vllm-project/vllm/pull/27743) **one contiguous physical block across layers**. Effective size × about **`2 × num_layers`**. Offload throughput up **an order of magnitude**. New layout typically **0.5–2 MB** vs a few KB.

| Model | Old block | New block |
| --- | --- | --- |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B (`tensor_parallel_size=2`) | 16 KB | 2 MB |
| deepseek-ai/DeepSeek-V2-Lite-Chat (GPU block size=64) | 72 KB | 1.9 MB |
| meta-llama/Llama-3.1-8B-Instruct | 32 KB | **2 MB** |
| meta-llama/Llama-3.2-1B-Instruct | 16 KB | **0.5 MB** |
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

(16-token blocks unless noted.) Combined with the microbench, DMA should be comparable or only slightly behind the kernel, depending on model.

## End-to-end: worst case for DMA (0.5 MB)

Llama-3.2-1B-Instruct, H100 — small physical block on purpose.

Figure 5, single-request TTFT: custom kernel **slightly** better — &lt;**1 ms** at 1K prompt, up to **~15 ms** at 90K. Larger-block models: roughly a tie.

Figure 6, concurrent 10k×512 Prefills: **DMA higher throughput**. ~**5.5%** at 0% hit, ~**15%** at 80% hit. At **0%** hit the custom kernel is **~6% worse than no offload at all** (it fights SMs). At **100%** hit there is no parallel compute, so the gap shrinks.

**Llama-3.1-8B-Instruct:** DMA up to **~32%** more throughput than the kernel, **matched TTFT**. Layout change exists so DMA can do this job.

## Eval setup (post)

- Ubuntu 24.04.1 LTS container; kernel `5.14.0-427.81.1.el9_4.x86_64`
- Intel Xeon Sapphire Rapids 2.1 GHz (**8 cores** limit)
- NVIDIA **H100 80GB HBM3**; **500 GB** DRAM; CUDA **12.9**
- vLLM commit `2a1776b7ac4fae7c50c694edeafc1b14270e4350`
- Flash Attention; GPU prefix caching **off**; GPU and CPU block size **16**; de/tokenization **off**
- Bench: [kv_offload_benchmark.py](https://github.com/orozery/playground/blob/kv-offloading-blog-dec-2025/kvcache/kv_offload_benchmark.py)

## Next (then)

CPU as an intermediate tier toward storage offload. Slack: `#feat-v1-cpu-offloading`.
