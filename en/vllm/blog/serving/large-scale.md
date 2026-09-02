---
source: https://vllm.ai/blog/2025-12-17-large-scale-serving
lang: en
fetched: 2026-08-31
---

# Large Scale Serving: DeepSeek @ 2.2k tok/s/H200

2025-12-17. V0 gone in v0.11.0. Coreweave H200 + IB CX7, multi-node production shape: ~**2.2k tok/s per H200** (was ~1.5k). Community snapshot then: 1969 contributors, 950+ commits in a month. 

The recipe stack: async scheduling, **dual-batch overlap (DBO)**, P/D, CUDA graph `FULL_AND_PIECEWISE` (`-O2`), DeepGEMM default, DeepEP, **EPLB**, DeepSeek-R1 SiLU kernel. Each knob helps; together on MoE they are Wide-EP.

**Wide-EP.** DeepSeek-R1 activates 37B/671B. MLA hates naive TP (duplicated latents) — the opposite of the super-linear KV rooms in [distributed-inference.md](distributed-inference.md). `--enable-expert-parallel` + DP (`mp` / `ray`). Attention keeps its own latents per DP worker so the effective batch can grow. All-to-all: DeepEP (also Perplexity MoE, NCCL AllGather-ReduceScatter). Dense attention → DP Attention; sparse experts → EP; group size `DP × TP`. [elastic-ep.md](elastic-ep.md) later resizes that DP count.

**DBO.** `--enable-dbo`: `all_reduce` decides whether to split microbatches (`--dbo-decode-token-threshold`); two worker threads overlap MoE dispatch/combine. Fatter EP → more useful. Elastic EP did **not** support DBO yet.

**EPLB.** `--enable-eplb`: sliding-window expert load, new logical→physical map, hot-swap weights without restart (DeepSeek hierarchical / global). Elastic scale-up reshuffles after the topology switch; scale-down reshuffles **first**.

**P/D.** One fat prefill can stall the whole EP group’s combine. DistServe (2024). [router.md](router.md) sends the request to the right pool; this post is why MoE hurts if you do not split.

**Deploy corridors:** llm-d (K8s well-lit path to the numbers in the post); Dynamo (KV-aware routing, KV Block Manager, Planner); Ray Serve LLM (P/D, DP attention, prefix affinity, NIXL/LMCache) — Elastic EP also needs the Ray DP backend.

Roadmap then already named elastic EP, long context, CPU KV transfer, batch invariance, bigger MoE fusion, FlashInfer SwapAB, independent TP on P and D, GB200. Those two (elastic EP, remote KV) became their own posts.

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
