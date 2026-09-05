---
source: https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive
lang: en
fetched: 2026-09-05
---

# vLLM Triton Attention Backend Deep Dive

Chinese: [zh/vllm/blog/architecture/triton-attn.md](../../../../zh/vllm/blog/architecture/triton-attn.md)  
Source: https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive

2026-03-04. **vLLM Team at IBM Research**. Study rewrite, not an official reprint. Adapted from a Red Hat [vLLM Office Hours](https://www.youtube.com/watch?v=8QiM-i9ifFo&list=PLbMP1JcGBmSHxp4-lubU5WYmJ9YgAQcf3&index=1) session with **Burkhard Ringlein** (IBM Research). Playlist / future hours: [past topics](https://www.youtube.com/playlist?list=PLbMP1JcGBmSHxp4-lubU5WYmJ9YgAQcf3), [red.ht/office-hours](https://red.ht/office-hours). Authors on the effort: IBM Research, Red Hat, AMD. Attention-backend selection: [optimization](../../optimization/optimization.md). Later ROCm routing: [rocm-attention](rocm-attention.md).

Kernel: [`vllm/v1/attention/ops/triton_unified_attention.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py) (~**800** LOC). FlashAttention 3 is ~**70,000** LOC. Backend wrapper: [`triton_attn.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py). Paper: [*The Anatomy of a Triton Attention Kernel*](https://arxiv.org/abs/2511.11581). Autotuning paper named with Figure 1: [*GPU Performance Portability needs Autotuning*](https://arxiv.org/abs/2505.03780).

**TL;DR from the page:**

- One Triton source on NVIDIA / AMD / Intel; depends only on **PyTorch + Triton**; always shipped, so it is an always-on fallback.
- **Q blocks** fatten query heads and tokens for `tl.dot`. Decode adds **parallel tiled softmax** (the “3D kernel”).
- CUDA graphs want a fixed grid; they moved to **persistent kernels** (then-current: PRs to vLLM still pending).
- Late 2025: Llama 3.1 8B, batch 1, input 500. H100 long Decode reached **100.7%** of FA3; MI300 ~**5.8×** over earlier implementations. Helion preview.

Original sections: Why Triton Helps vLLM → The Triton Attention Backend in vLLM → When the Triton Attention Backend Is Used → Writing a High-Performance Portable Paged Attention Kernel in Triton → Reminder: What the Paged Attention Kernel Does → Optimizing Tile Sizes for tl.dot Using Q Blocks → Adding Parallelization With Parallel Tiled Softmax → CUDA Graphs, Launch Grids, and GPU Execution Waves → From Variable Launch Grids to Persistent Kernels → Benchmarking Results → Preview: Paged Attention in Helion → Conclusion → Acknowledgments.

Over the past year, IBM Research, Red Hat, and AMD upstreamed a Triton attention backend for vLLM: state-of-the-art performance with portability across GPU vendors. The driver is hardware diversity and the cost of maintaining a zoo of specialized kernels.

The post explains why [Triton](https://github.com/triton-lang/triton) fits [vLLM](https://github.com/vllm-project/vllm), what the backend is and when it is used, then the paged-attention kernel: tiles, parallelization, CUDA graphs, benches, and a look at Helion.

## Why Triton Helps vLLM

vLLM wants best-possible inference across platforms, models, and execution strategies: multiple accelerators and generations, many architectures, and workloads that vary in batch size, sequence length, and attention pattern.

One approach is a zoo of highly specialized kernels, each tuned to one model and one GPU. Effective, and it does not scale. Maintaining hundreds of kernels across NVIDIA Hopper and Blackwell, AMD MI300, Intel, and whatever comes next is impractical.

The Triton backend bets on **performance-portable** kernels that adapt to the hardware they run on.

[Triton](https://github.com/triton-lang/triton) is a DSL for writing GPU kernels (matmul, attention, …) in Python; they compile to efficient code on multiple platforms. Tiled programming: low-level enough for hardware-relevant optimizations, high-level enough to stay largely hardware-agnostic.

**Figure 1.** Developers write **logical tiles**. The compiler and autotuner map those tiles onto the device. Tile shapes and execution layouts can differ completely across GPUs; those decisions are automatic, often autotuned (see the autotuning paper above).

![image1](../../../../assets/vllm/blog/architecture/triton-attn/01-image1.png)

**Figure 1.** Triton’s tiled programming model: logical tiles mapped to hardware-specific layouts by the compiler and autotuner.

## The Triton Attention Backend in vLLM

Attention is typically the most performance-critical op. vLLM isolates it behind **attention backends** — a common API, separate from simpler pieces such as linear layers and RMSNorm.

Inside that layer: FlashAttention and FlashInfer on CUDA, ROCm backends, specialized MLA backends (full list: [`vllm/v1/attention/backends`](https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/backends)). The [Triton attention backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py) is **entirely Triton**, **native to vLLM**.

It was introduced for portability and fewer hard dependencies. Same source on NVIDIA, AMD, and Intel. Depends only on **PyTorch + Triton**. Always shipped with vLLM, so it is an **always-on fallback**. Started at IBM Research and Red Hat AI; now maintained more broadly.

## When the Triton Attention Backend Is Used

- **Default on AMD GPUs (ROCm).**
- **Intel XPU float32:** vLLM falls back to Triton because FlashAttention does **not** support fp32 there.
- Models that need **ALiBi sqrt** (StepFun audio models), **sink tokens** and **GPT-OSS** behavior — especially on **pre-Hopper NVIDIA** (A100s).
- **Small head sizes**, **encoder and decoder** attention, **multimodal prefix** attention.
- **Batch invariance.**
- Fallback whenever FlashAttention, FlashInfer, or other deps are missing or fail to import.

## Writing a High-Performance Portable Paged Attention Kernel in Triton

Development started **outside** vLLM, with extensive microbenchmarks. The kernel API was designed to match vLLM; performance tuning was done in isolation before end-to-end integration.

[Microbenchmarks](https://github.com/foundation-model-stack/vllm-triton-backend) covered Prefill-heavy, Decode-heavy, and mixed workloads, and across batch sizes and context lengths.

Figure 2: x-axis total tokens; y-axis latency. Separate subplots for prefill-only, mixed, and decode-only. Different kernel variants win in different regimes; **no single configuration dominates**.

![image2](../../../../assets/vllm/blog/architecture/triton-attn/02-image2.png)

**Figure 2.** Microbenchmark comparison of multiple Triton paged-attention kernel variants across Prefill, Decode, and mixed workloads.

Microbenchmarks expose kernel-level behavior that end-to-end numbers hide.

## Reminder: What the Paged Attention Kernel Does

Paged attention pages the KV cache. For each query in a batch, for each query token, for each query head and corresponding KV head, the kernel traverses the paged KV cache, computes scores, applies value vectors.

Figure 3: query tokens on x, query heads on y, paged-KV traversal as the innermost loop. Causal masking and sliding windows omitted for clarity.

![image3](../../../../assets/vllm/blog/architecture/triton-attn/03-image3.png)

**Figure 3.** Conceptual view of paged attention: query tokens, query heads, and traversal of the paged KV cache.

Low-level kernel optimizations: the authors’ [PyTorch blog, *Enabling vLLM V1 on AMD GPUs with Triton*](https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/). Code: `triton_unified_attention.py` (link above).

## Optimizing Tile Sizes for tl.dot Using Q Blocks

The core op is matmul, `tl.dot`. High performance needs tiles large enough to feed the hardware; simply loading the paged KV cache did **not** get there.

KV-side tile size is capped by **page size**, so they optimize the **query** side. For GQA, process **all query heads that share a KV head** together (cache reuse). Then group **multiple query tokens** into one work item — a **Q block**.

Figure 4: launch grid spans batch size and KV heads. Q blocks decide how many query tokens and heads each kernel instance owns. Autotune picks block sizes per platform.

![image4](../../../../assets/vllm/blog/architecture/triton-attn/04-image4.png)

**Figure 4.** Q blocks combine multiple query heads and query tokens into a single work item to improve `tl.dot` utilization and cache reuse.

## Adding Parallelization With Parallel Tiled Softmax

Bundling query tokens helps **Prefill**. It does nothing for **Decode**, which has a **single** query token. Extra parallelization: **parallel tiled softmax**, the **“3D kernel.”**

KV-cache traversal is split across multiple kernel instances. Each computes partials; a later reduction produces the output. Triton has **no global barrier**, so the reduction is a **second kernel** — parallelism versus launch overhead. **Heuristics** decide when the second launch is worth it.

## CUDA Graphs, Launch Grids, and GPU Execution Waves

CUDA graphs cut launch overhead by recording and replaying a **fixed** graph. Attention grids often depend on batch size and sequence length, which fights that.

GPUs run a fixed number of SMs. Launch more threads than SMs and execution proceeds in **waves**. Figure 5: a second wave underutilizes.

![image5](../../../../assets/vllm/blog/architecture/triton-attn/05-image5.png)

**Figure 5.** GPU execution waves when launched threads exceed available SMs. Example on the page: GPU has **8 SMs**, they want to execute **12 threads**.

Captured in a CUDA graph, that waste is **replayed** even if the effective workload shrinks. Figure 6: extra wasted work when replaying fixed launch grids via CUDA graphs → higher latency.

![image6](../../../../assets/vllm/blog/architecture/triton-attn/06-image6.png)

**Figure 6.** Additional wasted work when replaying fixed launch grids via CUDA graphs.

## From Variable Launch Grids to Persistent Kernels

Figure 7: early paged-attention kernels used **variable launch grids** that scaled with workload size. Flexible, and a poor fit for CUDA graphs.

![image7](../../../../assets/vllm/blog/architecture/triton-attn/07-image7.png)

**Figure 7.** Variable launch grids used in earlier paged-attention kernels.

They designed **persistent kernels**. **Then-current status on the page: PRs to vLLM pending.** A **fixed** number of instances launches, equal to available compute. Each instance reads **metadata from GPU memory** to decide how much work to take. Launch grid stays constant; CUDA graphs can be reused.

![image8](../../../../assets/vllm/blog/architecture/triton-attn/08-image8.png)

**Figure 8.** Persistent kernel: fixed grids, dynamic work assignment.

## Benchmarking Results

Late 2025 end-to-end. Figure 9: **Llama 3.1 8B**, **batch size 1**, **input length 500** tokens. x-axis: output length. NVIDIA **H100** and AMD **MI300**. Normalized to the left-most baseline.

- **H100:** Triton attention reached **100.7%** of FlashAttention 3 for **long Decode** requests.
- **MI300:** about **5.8×** over earlier implementations.
- **Same Triton kernel source** on both platforms.
- LOC reminder: paged attention in Triton ~**800** lines; FlashAttention 3 ~**70,000**.

![image9](../../../../assets/vllm/blog/architecture/triton-attn/09-image9.png)

![image10](../../../../assets/vllm/blog/architecture/triton-attn/10-image10.png)

**Figure 9.** End-to-end latency of Triton paged attention vs FlashAttention 3 on NVIDIA H100 and AMD MI300. Normalized to the left-most baseline.

## Preview: Paged Attention in Helion

[Helion](https://github.com/pytorch/helion) (PyTorch team): a higher-level Triton, or tiled PyTorch. A **simplified** paged-attention kernel was implemented in Helion as an experiment; early results were promising. Write-up: [PyTorch blog, *Portable Paged Attention in Helion*](https://pytorch.org/blog/portable-paged-attention-in-helion/). Code: [draft PR #27293](https://github.com/vllm-project/vllm/pull/27293) on vLLM.

## Conclusion

Models, inference tricks, and hardware keep moving; **performance portability** matters more. The Triton attention backend is the claim that you can reach state-of-the-art attention with **one portable kernel**.

Kernel design + microbenchmarks + system-level work (persistent kernels, CUDA graphs) let it match or beat specialized implementations while staying portable. As of the post it is the **default on AMD**, and it runs on NVIDIA and Intel from the **same source**.

The blog is an overview of the important optimizations; details and more benches are in [*The Anatomy of a Triton Attention Kernel*](https://arxiv.org/abs/2511.11581). Triton is the portable default, not the slow spare — especially where FA3 is absent or too expensive to port.

## Acknowledgments

AI platform team at IBM Research, named on the page: **Burkhard Ringlein**, **Jan van Lunteren**, **Chih-Chieh Yang**, **Sara Kokkila Schumacher**, **Thomas Parnell**, **Mudhakar Srivatsa**, **Raghu Ganti**.
