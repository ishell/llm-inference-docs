---
source: https://vllm.ai/blog/2025-10-09-blackwell-inferencemax
lang: en
fetched: 2026-09-05
---

# SemiAnalysis InferenceMAX: vLLM and NVIDIA Accelerate Blackwell Inference

Chinese: [zh/vllm/blog/performance/blackwell-inferencemax.md](../../../../zh/vllm/blog/performance/blackwell-inferencemax.md)  
Source: https://vllm.ai/blog/2025-10-09-blackwell-inferencemax

2025-10-09. **vLLM Team**. Study extract, not an official reprint. Numbers follow **that day’s** curve — not a permanent plate. Later Pareto on the same family: [gpt-oss-optimizations.md](gpt-oss-optimizations.md). Day-0 gpt-oss: [../serving/gpt-oss.md](../serving/gpt-oss.md). Board: [inferencemax.ai](http://inferencemax.ai); code [InferenceMAX/InferenceMAX](https://github.com/InferenceMAX/InferenceMAX); write-up [SemiAnalysis newsletter](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference).

Fits: reading InferenceMAX’s Pareto on Blackwell **B200/GB200** (gpt-oss 120B, Llama 3.3 70B) and the named FlashInfer / `torch.compile` / `--async-scheduling` work. Does not fit: treating **4×**, **4.3×**, or **3.7×** as your SLA.

## Introduction

Months of close work with NVIDIA on Blackwell (**B200/GB200**) for LLM inference in vLLM. Blackwell brings a new class of efficiency: more memory bandwidth, native **FP4** tensor cores. Performance out of the box is already strong. Extracting more meant refactoring existing kernels and writing new ones for lower-level hardware utilization.

The [SemiAnalysis InferenceMAX](https://github.com/InferenceMAX/InferenceMAX) benches reflect those changes: up to **4× higher throughput at similar latency** vs Hopper on gpt-oss 120B and Llama 3.3 70B.

Multi-month engineering collaboration: **over a hundred pull requests** across vLLM. With NVIDIA they optimized nearly every part of the pipeline — custom kernels (attention, GEMM, MoE) through high-level scheduling and overhead removal. This post breaks those optimizations down and how they turn Blackwell features into production gains.

## Overview of InferenceMax

SemiAnalysis InferenceMAX is a benchmark framework for **automated, recurring** LLM serving tests, with results **updated daily** so software changes show up on a public board. Consistent methods keep comparisons fair and reproducible. The gap between a software update and published numbers is supposed to shrink.

Two representative open-source models then:

- Mixture-of-Experts (MoE): **gpt-oss 120B**
- Dense: **Llama 3.3 70B**

Each model under several prompt/response lengths (ISL = input sequence length, OSL = output sequence length). Three regimes:

| ISL / OSL | Story they tell |
|---|---|
| 1K / 1K | chat, moderate input/output |
| 1K / 8K | reasoning, long outputs |
| 8K / 1K | summarization, long inputs |

## Delivering Performance Across the Pareto Frontier

Blackwell’s compute architecture is a step-change in inference efficiency: latest **HBM3e** (**192 GB** at **8 TB/s** per B200), NVLink **1.8 TB/s** per GPU, 5th-generation tensor cores with built-in **FP4**.

Kernels adapted to those advances. Dramatic gains in throughput (per-GPU) and responsiveness (per-request latency) vs the same vLLM on Hopper.

Workloads vary in sequence length, batch size, and concurrency. A max-throughput config is rarely the min per-user-latency config. Single-point metrics mislead. InferenceMAX uses a **Pareto frontier**: the trade-off between responsiveness and throughput, mapping the envelope across real operating points.

The collaboration’s primary goal: vLLM should use Blackwell’s features across the **full** Pareto frontier — not one working point.

SemiAnalysis results: consistent Blackwell vs Hopper improvements across all interactivity levels for both gpt-oss 120B and Llama 3.3 70B.

Local figures (copyright remains with the original site; study copies):

![gpt oss 120b 1k 1k](../../../../assets/vllm/blog/performance/blackwell-inferencemax/01-gpt-oss-120b-1k-1k.png)

**Figure 1.** SemiAnalysis InferenceMax gpt-oss-120b Pareto, vLLM Blackwell vs Hopper, 1k/1k ISL/OSL across a wide range of interactivity. Up to **4.3×** throughput at similar interactivity.

![llama 70b 1k 8k](../../../../assets/vllm/blog/performance/blackwell-inferencemax/02-llama-70b-1k-8k.png)

**Figure 2.** Same for Llama 3.3 70B, 1k/8k ISL/OSL. Up to **3.7×** throughput.

**These gains were reproducible then** with the InferenceMAX configurations SemiAnalysis published. Optimized software, aimed at extracting the hardware, can land here. Reaching the numbers took a broad set of vLLM optimizations, developed with NVIDIA’s engineers. The most significant of those next.

## vLLM Blackwell Optimizations

Enabling that performance involved work at all levels of the stack. Some optimizations speed raw kernel execution; others cut CPU overhead or use hardware features more fully. Key enhancements named for Blackwell support so far:

**Performance Improvements**

- **Faster kernels via [FlashInfer](https://github.com/flashinfer-ai/flashinfer).** NVIDIA’s FlashInfer library: high-performance kernels including FP8 attention for GQA and MLA, fast FP8 and FP4 GEMMs, MoE kernels, and fused operations. Example: AllReduce + RMSNorm + quantization in a **single** kernel launch, cutting latency. Stack named: CUTLASS, CuTeDSL, cuBLAS, cuDNN, TRTLLM.
- **Smarter graph fusions.** Expanding vLLM’s `torch.compile` graph fusions to patterns such as Attention + Output Quant and AllReduce + RMSNorm + Quant. Fused-kernel performance without manual per-model edits; most importantly, it **generalizes across architectures**.
- **Reduced host overhead with `--async-scheduling`.** Full overlap between model execution and host setup, so the GPU does not idle on synchronization. **The workload is fully pipelined:** while one batch runs on the GPU, the next batch’s data is prepared in parallel.

**Usability Improvements**

- **Automatic quantization and backend selection.** vLLM detects whether a model is quantized and picks the backend; it also picks the attention backend for the GPU. On Blackwell it chooses FlashInfer-based attention (incorporating NVIDIA TensorRT-LLM kernels) when available, else FlashAttention — no manual flags or environment soup.
- **Autotuning for FlashInfer GEMM and MoE.** Ideal kernel depends heavily on batch size and sequence length. Autotuning in the GPU runner: at startup FlashInfer does tactic selection — benchmark and pick kernels — so peak performance holds as ISL/OSL vary.
- **[Quick Start Recipes](https://github.com/vllm-project/recipes).** Alongside code, community quick-start guides for common scenarios. Per model and hardware: launch, tune, validate accuracy, benchmark. Shorter path to numbers.

## Ongoing Work

Each item above was a significant project of its own, in close technical collaboration — and they have not covered them all. The NVIDIA collaboration is ongoing; more improvements sit on the horizon.

Looking ahead: cluster-scale speculative decoding and **Data+Expert Parallel (DEP)** for DeepSeek, Qwen, gpt-oss, and more. NVIDIA’s `gpt-oss-120b-Eagle3-v2` (Eagle speculative decoding): they anticipate ~**2–3×** throughput. DEP uses Blackwell’s **1,800 GB/s** low-latency NVLink GPU-to-GPU interconnect; they expect still more performance and higher concurrency than InferenceMAX showed.

Blackwell performance moves every day from ongoing optimizations between vLLM and NVIDIA. They say they keep finding new ways to push efficiency and scale.

## Acknowledgements

People in the vLLM community named on the page:

- **Red Hat:** Michael Goin, Alexander Matveev, Lucas Wilkinson, Luka Govedič, Wentao Ye, Ilia Markov, Matt Bonanni, Varun Sundar Rabindranath, Bill Nell, Tyler Michael Smith, Robert Shaw
- **NVIDIA:** Po-Han Huang, Pavani Majety, Shu Wang, Elvis Chen, Zihao Ye, Duncan Moss, Kaixi Hou, Siyuan Fu, Benjamin Chislett, Xin Li, Vadim Gimpelson, Minseok Lee, Amir Samani, Elfie Guo, Lee Nau, Kushan Ahmadian, Grace Ho, Pen Chun Li
- **vLLM:** Chen Zhang, Yongye Zhu, Bowen Wang, Kaichao You, Simon Mo, Woosuk Kwon, Zhuohan Li
- **Meta:** Yang Chen, Xiaozhu Meng, Boyuan Feng, Lu Fang

All InferenceMAX results: [http://inferencemax.ai](http://inferencemax.ai). Code: [https://github.com/InferenceMAX/InferenceMAX](https://github.com/InferenceMAX/InferenceMAX). Their explanation: [https://newsletter.semianalysis.com/p/inferencemax-open-source-inference](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference).

Thanks to SemiAnalysis for pushing hardware and open-source software co-design toward fair measurements. Named: Kimbo Chen, Dylan Patel, and others.

They close: more refining and expanding in the weeks and months ahead.
