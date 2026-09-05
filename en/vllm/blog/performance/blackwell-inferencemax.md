---
source: https://vllm.ai/blog/2025-10-09-blackwell-inferencemax
lang: en
fetched: 2026-09-04
---

# SemiAnalysis InferenceMAX: vLLM and NVIDIA Accelerate Blackwell Inference

Chinese: [zh/vllm/blog/performance/blackwell-inferencemax.md](../../../../zh/vllm/blog/performance/blackwell-inferencemax.md)

2025-10-09. **vLLM Team**. Later Pareto on the same family: [gpt-oss-optimizations.md](gpt-oss-optimizations.md). Day-0 gpt-oss: [../serving/gpt-oss.md](../serving/gpt-oss.md). Board: [inferencemax.ai](http://inferencemax.ai); code [InferenceMAX/InferenceMAX](https://github.com/InferenceMAX/InferenceMAX); write-up [SemiAnalysis newsletter](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference). Numbers follow **that day’s** curve — not a permanent plate.

Months of work with NVIDIA on Blackwell **B200/GB200**: more HBM bandwidth, native **FP4** tensor cores. Joint kernel + scheduling work. They quote up to **4× higher throughput at similar latency** vs Hopper on gpt-oss 120B and Llama 3.3 70B. “Over a hundred PRs.”

## What InferenceMAX is

Automated, **daily** serving benches so software changes show up on a public board. Two models then:

- MoE: **gpt-oss 120B**
- Dense: **Llama 3.3 70B**

Three ISL/OSL regimes:

| ISL / OSL | Story they tell |
|---|---|
| 1K / 1K | chat, moderate |
| 1K / 8K | reasoning, long output |
| 8K / 1K | summarization, long input |

Single-point TPS lies: max-throughput configs are rarely min per-user latency. They map a **Pareto frontier** (responsiveness vs throughput).

Blackwell hardware they name: **192 GB HBM3e at 8 TB/s per B200**, NVLink **1.8 TB/s** per GPU, 5th-gen tensor cores + FP4.

Local figures (copyright remains with the original site; study copies):

![gpt oss 120b 1k 1k](../../../../assets/vllm/blog/performance/blackwell-inferencemax/01-gpt-oss-120b-1k-1k.png)

![llama 70b 1k 8k](../../../../assets/vllm/blog/performance/blackwell-inferencemax/02-llama-70b-1k-8k.png)

**Fig 1:** gpt-oss-120b 1k/1k Pareto, Blackwell vs Hopper — up to **4.3×** throughput at similar interactivity.  
**Fig 2:** Llama 3.3 70B 1k/8k — up to **3.7×**. Reproducible with SemiAnalysis configs.

## Optimizations they list

**Performance**

- **FlashInfer:** FP8 attention for GQA and MLA; fast FP8/FP4 GEMMs; MoE; fused ops. Example: AllReduce + RMSNorm + quant in one launch. Stack named: CUTLASS, CuTeDSL, cuBLAS, cuDNN, TRTLLM.
- **torch.compile fusions:** Attention + Output Quant; AllReduce + RMSNorm + Quant — generalize across architectures, no per-model hand fusion.
- **`--async-scheduling`:** overlap model execution with host setup so the GPU does not idle on sync.

**Usability**

- Auto-detect quantization and pick attention backend (Blackwell → FlashInfer / TRTLLM kernels, else FlashAttention) — no manual env soup.
- FlashInfer GEMM/MoE **autotune at startup** (tactic selection vs batch / ISL / OSL).
- [Quick-start recipes](https://github.com/vllm-project/recipes) for launch, accuracy, bench.

## Ongoing then

Speculative decoding + **Data+Expert Parallel (DEP)** at cluster scale for DeepSeek, Qwen, gpt-oss. NVIDIA `gpt-oss-120b-Eagle3-v2`: they anticipate ~**2–3×** throughput. DEP uses the 1800 GB/s NVLink path for higher concurrency than InferenceMAX showed.

## Acknowledgements

Same name block as the later gpt-oss Pareto post (Red Hat / NVIDIA / vLLM / Meta). SemiAnalysis: Kimbo Chen, Dylan Patel, and others.
