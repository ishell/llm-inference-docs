---
source: https://vllm.ai/blog/2025-12-17-large-scale-serving
lang: en
fetched: 2026-09-04
---

# vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP

Chinese: [zh/vllm/blog/serving/large-scale.md](../../../../zh/vllm/blog/serving/large-scale.md)

2025-12-17. **v0.11.0** deleted the last V0 engine code — full [V1](../architecture/v1-alpha.md) migration. Community snapshot then (as of 2025-12-18): **1,969** contributors, **950+** commits in a month. Also named: [SemiAnalysis InferenceMax](https://inferencemax.semianalysis.com/) inclusion, and production trust at Meta, LinkedIn, Red Hat, Mistral, Hugging Face.

DeepSeek-style **disaggregated serving** + sparse MoE is the SOTA shape they are chasing. Knob list in the post:

- Async scheduling
- Dual-batch overlap (DBO)
- Disaggregated Prefill/Decode
- CUDA graph mode `FULL_AND_PIECEWISE`
- DeepGEMM on by default
- DeepEP kernels
- Expert parallel load balancing (EPLB)
- SiLU kernel for DeepSeek-R1

Further reading they point at: llm-d large-scale serving, PyTorch disaggregated inference, NVIDIA Dynamo, Anyscale wide-EP.

Local figures (copyright remains with the original site; study copies):

![prefill throughput](../../../../assets/vllm/blog/serving/large-scale/01-prefill_throughput.png)

![decode throughput](../../../../assets/vllm/blog/serving/large-scale/02-decode_throughput.png)

![wide ep](../../../../assets/vllm/blog/serving/large-scale/03-wide_ep.gif)

![kv cache](../../../../assets/vllm/blog/serving/large-scale/04-kv_cache.png)

![a2a backends](../../../../assets/vllm/blog/serving/large-scale/05-a2a_backends.png)

![dbo before](../../../../assets/vllm/blog/serving/large-scale/06-dbo_before.png)

![dbo after](../../../../assets/vllm/blog/serving/large-scale/07-dbo_after.png)

![eplb](../../../../assets/vllm/blog/serving/large-scale/08-eplb.gif)

![disaggregated serving](../../../../assets/vllm/blog/serving/large-scale/09-disaggregated_serving.gif)

![llm d](../../../../assets/vllm/blog/serving/large-scale/10-llm-d.png)

![dynamo](../../../../assets/vllm/blog/serving/large-scale/11-dynamo.png)

![ray serve llm](../../../../assets/vllm/blog/serving/large-scale/12-ray_serve_llm.png)

## Results

Community benches on a **Coreweave H200** cluster, InfiniBand **ConnectX-7**: sustained **~2.2k tokens/s per H200** in production-like multi-node deployments (earlier ~**1.5k**). Gain mix they name: kernel work (silu-mul-quant fusion, Cutlass QKV, TP attention bug fixes) + **DBO for Decode**. Fewer replicas for a target QPS → lower token-per-dollar.

Prefill and Decode throughput plots are on the official page / local copies above.

## Wide-EP

DeepSeek-V3 family needs two facts:

- Sparse activation: DeepSeek-R1 fires **37B of 671B** per forward.
- KV: **pure TP is a bad fit for MLA** — latent projections are duplicated across shards.

`--enable-expert-parallel`: one set of experts shared across ranks; tokens route to the rank that owns the expert. **Wide-EP = EP + data parallelism** (`mp` or `ray` backends). Attention is duplicated per DP rank so latents stay independent and the effective batch can grow.

Memory postcard in the post: TP on DeepSeek-V3 leaves **~34GB free per H200**, but MLA still duplicates latents. DP Attention trades that duplication for KV room.

Fatter EP → more sync. All-to-all backends: [DeepEP](https://github.com/deepseek-ai/DeepEP) (high-throughput and low-latency variants), [Perplexity MoE kernels](https://github.com/perplexityai/pplx-kernels), NCCL AllGather-ReduceScatter. Docs: vLLM MoE kernel page. Dense attention → DP Attention; sparse experts → EP; group size `DP × TP`. [elastic-ep.md](elastic-ep.md) later resizes that DP count.

## Dual-batch overlap (DBO)

DeepSeek [microbatching](https://github.com/deepseek-ai/profile-data), exposed as `--enable-dbo`:

1. Collective `all_reduce` agrees that microbatching is worth it (`--dbo-decode-token-threshold`).
2. Main thread starts microbatch workers that CUDA-graph capture.
3. Modular MoE all-to-all base class launches workers and **yields** while GPU work is in flight.

Without DBO, traces show MoE dispatch/combine dominating despite small compute. With DBO, worker A finishes dispatch and yields to B, then combine overlaps the same way. Fatter EP → more useful. Elastic EP did **not** support DBO yet.

## EPLB

Train-time expert load is balanced; inference traffic is not ([NVIDIA MoE routing notes](https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/#experimental_results)). `--enable-eplb` implements DeepSeek [EPLB](https://github.com/deepseek-ai/EPLB) hierarchical / global policies: sliding window of per-token load across EP ranks; at the rebalance interval, a new logical→physical map and a **weight shuffle without restart**. Tune window, interval, redundant experts, logging. Elastic scale-up reshuffles **after** the topology switch; scale-down reshuffles **first**.

## Why MoE needs P/D disaggregation

DistServe (Hao AI Lab, 2024). Experts live on many ranks, so one compute-bound Prefill can stall the **whole EP group's combine** (unused ranks still dummy-step). Split Prefill/Decode and you can also pin DeepEP to the high-throughput vs low-latency kernel per pool. [router.md](router.md) sends the request to the right pool; this post is why MoE hurts if you do not split.

## Three deployment corridors

- **llm-d** — Kubernetes-native; [Wide-EP well-lit path](https://github.com/llm-d/llm-d/tree/main/guides/wide-ep-lws) to reproduce the numbers.
- **Dynamo** — KV-aware routing, KV Block Manager, Planner; vLLM + wide-EP is first-class. Recipe linked from the original.
- **Ray Serve LLM** — P/D, DP attention, prefix-affinity routing; NIXL / LMCache; independent autoscaling of phases; sits next to Ray data / RL. Elastic EP also needs the Ray DP backend.

## Roadmap (then)

Elastic EP, long context, CPU KV transfer, full determinism / batch invariance, larger MoE fusions (DeepSeek-R1, gpt-oss), FlashInfer SwapAB, **independent TP sizes** on Prefill vs Decode, GB200. Living page: [roadmap.vllm.ai](http://roadmap.vllm.ai/). Elastic EP and remote KV became their own posts ([elastic-ep](elastic-ep.md), [mooncake](mooncake.md)).
