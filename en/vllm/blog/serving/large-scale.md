---
source: https://vllm.ai/blog/2025-12-17-large-scale-serving
lang: en
fetched: 2026-09-05
---

# vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP

Chinese: [zh/vllm/blog/serving/large-scale.md](../../../../zh/vllm/blog/serving/large-scale.md)  
Source: https://vllm.ai/blog/2025-12-17-large-scale-serving

2025-12-17. **vLLM Team.** Study extract. Later: [elastic-ep.md](elastic-ep.md), [mooncake.md](mooncake.md), [router.md](router.md).

## Introduction

In **v0.11.0**, the last V0 engine code was removed — complete migration to [V1](../architecture/v1-alpha.md). Community snapshot as of 2025-12-18: **1,969** contributors, **950+** commits in the past month.

Also named: inclusion in SemiAnalysis open-source [InferenceMax](https://inferencemax.semianalysis.com/); production trust at Meta, LinkedIn, Red Hat, Mistral, Hugging Face.

DeepSeek-style **disaggregated serving** + sparse MoE remains the SOTA shape they are chasing. Optimizations in the post:

- Async scheduling
- Dual-batch overlap
- Disaggregated serving
- CUDA graph mode `FULL_AND_PIECEWISE`
- DeepGEMM enabled by default
- DeepEP kernels
- Expert parallel load balancing
- SiLU kernel for DeepSeek-R1

Further reading they point at: llm-d [large scale serving](https://llm-d.ai/blog/llm-d-v0.3-expanded-hardware-faster-perf-and-igw-ga), PyTorch [disaggregated serving](https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/), NVIDIA Dynamo [distributed inference](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/#boosting_inference_performance_on_nvidia_gb200_nvl72_by_30x), Anyscale [wide-EP](https://www.anyscale.com/blog/ray-serve-llm-anyscale-apis-wide-ep-disaggregated-serving-vllm).

Local figures (copyright remains with the original site; study copies).

## Results

Community [Wide-EP benches](https://llm-d.ai/blog/llm-d-v0.3-expanded-hardware-faster-perf-and-igw-ga#wide-ep-performance) on a **Coreweave H200** cluster, InfiniBand **ConnectX-7**: sustained **~2.2k tokens/s per H200** in production-like multi-node deployments.

Earlier ~**1.5k tokens/s per GPU**. Gain mix they name: kernel work (silu-mul-quant fusion, Cutlass QKV, TP attention bug fixes) + **DBO for Decode**.

Operators: consolidate workloads, fewer replicas for a target QPS, lower token-per-dollar.

![prefill throughput](../../../../assets/vllm/blog/serving/large-scale/01-prefill_throughput.png)

**Caption.** Prefill Results.

![decode throughput](../../../../assets/vllm/blog/serving/large-scale/02-decode_throughput.png)

**Caption.** Decode Results.

## Key Components

### Wide-EP

Deploying DeepSeek-V3-family models at scale needs two facts:

- **Sparse expert activation:** DeepSeek-R1 fires **37B of 671B** per forward.
- **KV cache management:** pure TP is a poor fit for MLA — latent projections are duplicated across shards.

EP maximizes effective KV. Flag: `--enable-expert-parallel`. One set of experts shared across ranks; tokens route to the rank that owns the expert.

![wide ep](../../../../assets/vllm/blog/serving/large-scale/03-wide_ep.gif)

**Caption.** Wide-EP token routing.

Wide-EP = EP + **data parallelism**. DP backends: `mp` or `ray` (simpler inside a Ray cluster). Memory postcard: TP on DeepSeek-V3 leaves **~34GB free per H200**, but MLA still duplicates latent attention projections. In DP, attention layers are duplicated so latents stay independent and the effective batch can grow.

![kv cache](../../../../assets/vllm/blog/serving/large-scale/04-kv_cache.png)

Fatter EP → more sync. All-to-all: [DeepEP](https://github.com/deepseek-ai/DeepEP) high-throughput and low-latency kernels, Perplexity [MoE kernels](https://github.com/perplexityai/pplx-kernels), NCCL AllGather-ReduceScatter. Docs: vLLM MoE [kernel page](https://docs.vllm.ai/en/latest/design/moe_kernel_features/), [fused MoE modular all2all backends](https://docs.vllm.ai/en/latest/design/moe_kernel_features/#fused-moe-modular-all2all-backends).

![a2a backends](../../../../assets/vllm/blog/serving/large-scale/05-a2a_backends.png)

**Caption.** vLLM all-to-all backends.

Dense attention → DP Attention; sparse experts → EP; group size `DP × TP`. [elastic-ep.md](elastic-ep.md) later resizes that DP count.

### Dual-batch Overlap (DBO)

DeepSeek [microbatching](https://github.com/deepseek-ai/profile-data), exposed as `--enable-dbo`. Overlap compute and collective communication:

1. Collective `all_reduce` agrees microbatching is worth it. Threshold: `--dbo-decode-token-threshold`.
2. Main thread creates microbatch worker threads, which complete CUDA graph capture.
3. Modular MoE all-to-all kernel base class launches workers and **yields** while GPU work is in flight.

Without DBO, a DeepSeek Decode trace shows MoE Dispatch/Combine dominating despite small compute.

![dbo before](../../../../assets/vllm/blog/serving/large-scale/06-dbo_before.png)

**Caption.** Before DBO.

With DBO: the first microbatch worker finishes MoE dispatch and yields to the second; the second finishes its dispatch and yields back; the first finishes combine, then yields for the second’s combine. Higher GPU utilization when communication is fat — high EP degree. Elastic EP did **not** support DBO yet.

![dbo after](../../../../assets/vllm/blog/serving/large-scale/07-dbo_after.png)

**Caption.** After DBO.

### Expert Parallel Load Balancing (EPLB)

Train-time expert load is balanced; inference traffic is not ([NVIDIA MoE routing notes](https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/#experimental_results)). In Wide-EP, some ranks idle while others process large batches.

vLLM implements DeepSeek [EPLB](https://github.com/deepseek-ai/EPLB) hierarchical / global policies. Flag: `--enable-eplb`. Configurable window size, rebalance interval, redundant experts, logging.

![eplb](../../../../assets/vllm/blog/serving/large-scale/08-eplb.gif)

**Caption.** EPLB in action.

Each MoE forward records per-token load; a sliding window aggregates across EP ranks. At the rebalance interval: new logical→physical expert mapping and a **weight shuffle without restart**.

Elastic scale-up reshuffles **after** the topology switch; scale-down reshuffles **first**.

### Disaggregated Serving

DistServe (Hao AI Lab, 2024) [paper](https://hao-ai-lab.github.io/blogs/distserve-retro/). Especially useful for EP.

![disaggregated serving](../../../../assets/vllm/blog/serving/large-scale/09-disaggregated_serving.gif)

**Caption.** P/D disaggregation in action.

Experts live on many ranks, so a request starting on one rank may need an expert on any other. MoE layers must synchronize (unused ranks still **dummy-pass**) so combine collectives are ready. One compute-bound Prefill can stall the **whole EP group’s** forward — the case for splitting Prefill and Decode. DeepSeek deployments can also pin DeepEP to the high-throughput vs low-latency kernel per pool.

[router.md](router.md) sends the request to the right pool; this post is why MoE hurts if you do not split. Text P/D is not the encoder split in [epd.md](epd.md).

## Deployment Paths

### llm-d

Kubernetes-native distributed inference stack; well-lit paths to SOTA performance for key OSS models across accelerators and providers. Reproduce this post: [Wide-EP well-lit path](https://github.com/llm-d/llm-d/tree/main/guides/wide-ep-lws).

![llm d](../../../../assets/vllm/blog/serving/large-scale/10-llm-d.png)

### Dynamo

High-throughput, low-latency production LLMs. KV-aware routing, KV Block Manager (cache offload), Planner (dynamic load matching) for tighter SLAs while scaling across more GPUs. vLLM + wide-EP is first-class. Docs: [Dynamo](https://docs.nvidia.com/dynamo/latest/index.html). Recipe to replicate this post: [example](https://github.com/ai-dynamo/dynamo/pull/4463/files#diff-363ddf6952864a610a1047f6b99c52461d6de9a4e198f89eb49d34f009a4d22b).

![dynamo](../../../../assets/vllm/blog/serving/large-scale/11-dynamo.png)

### Ray Serve LLM

First-class patterns on Ray Serve: [Prefill/Decode disaggregation](https://docs.ray.io/en/latest/serve/llm/architecture/serving-patterns/prefill-decode.html), [data parallel attention](https://docs.ray.io/en/latest/serve/llm/architecture/serving-patterns/data-parallel.html), [prefix cache-affinity routing](https://docs.ray.io/en/latest/serve/llm/architecture/routing-policies.html). Modularity and ease of deploy on Ray clusters (including KubeRay on Kubernetes). Differentiator: Ray data processing and RL.

NIXL and LMCache connectors for KV transfer; independent autoscaling of each phase. A programmable layer for composing serving patterns. Elastic EP scale ops also depend on the Ray DP backend.

![ray serve llm](../../../../assets/vllm/blog/serving/large-scale/12-ray_serve_llm.png)

## Roadmap (then)

- Elastic expert parallelism
- Long context serving
- KV cache transfer via CPU
- Full determinism and batch invariance
- Large MoE optimizations (op fusion for DeepSeek-R1 and gpt-oss)
- Better FlashInfer integration (e.g. SwapAB)
- Independent TP sizes in disaggregated Prefill vs Decode
- GB200 optimizations for large-scale serving

Living page: [roadmap.vllm.ai](http://roadmap.vllm.ai). Elastic EP and remote KV became their own posts ([elastic-ep.md](elastic-ep.md), [mooncake.md](mooncake.md)). Sequence sharding for long context: [dcp.md](../performance/dcp.md).

## Summary

- vLLM fully on V1; DeepSeek-style MoE with wide-EP at **2.2k tok/s/H200**.
- Wide-EP maximizes KV efficiency for MLA; DBO and EPLB cut communication bottlenecks and load imbalance.
- Disaggregated Prefill/Decode further splits MoE Prefill vs Decode. Deployment corridors: llm-d, Dynamo, Ray Serve LLM.
