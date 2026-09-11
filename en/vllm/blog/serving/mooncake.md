---
source: https://vllm.ai/blog/2026-05-06-mooncake-store
lang: en
fetched: 2026-09-05
---

# Serving Agentic Workloads at Scale with vLLM × Mooncake

Chinese: [zh/vllm/blog/serving/mooncake.md](../../../../zh/vllm/blog/serving/mooncake.md)  
Source: https://vllm.ai/blog/2026-05-06-mooncake-store

2026-05-06. **Yifan Qiao, Trong Dao Le, Ao Shen, Zhewen Li, Bowen Wang.** [Mooncake](https://github.com/kvcache-ai/Mooncake) already moved KV for P/D via [`MooncakeConnector`](https://docs.vllm.ai/en/stable/features/mooncake_connector_usage/). This post adds a **cluster-wide Mooncake Store**. Implementation: [PR #40900](https://github.com/vllm-project/vllm/pull/40900); bench scripts in the [artifact tree](https://github.com/ivanium/vllm/tree/feat/mooncake-store-int/scripts/mooncake/artifacts). Numbers are a demo on Codex / SWE-bench Pro traces, not your SLA. KV-aware routing: [router.md](router.md). Local CPU offload: [kv-offload.md](kv-offload.md).

**TL;DR:** Agentic workloads generate massive shared prefixes that are often recomputed across turns. Integrating Mooncake’s distributed KV store into vLLM: **3.8×** throughput, **46×** lower TTFT, **8.6×** lower end-to-end latency on realistic traces, scaling nearly linearly to **60 GB200 GPUs**.

Local figures (copyright remains with the original site; study copies):

![hero vllm mooncake](../../../../assets/vllm/blog/serving/mooncake/01-hero_vllm_mooncake.svg)

## Agentic workloads are reshaping LLM serving

With agents such as Claude Code and OpenClaw, inference is shifting. Jensen’s GTC 2026 [keynote](https://www.nvidia.com/gtc/keynote/): LLMs moving beyond chatbots toward autonomous, long-running systems that plan, reason, and act.

The structure is long-horizon, multi-turn loops: a *reasoning* step (process context, intermediate thoughts) and an *action* step (tool calls, external outputs).

Traces from Codex and GPT-5.4 on SWE-bench Pro, open-sourced as [Inferact/codex_swebenchpro_traces](https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces).

![agentic trace](../../../../assets/vllm/blog/serving/mooncake/02-agentic_trace.svg)

**Figure 1.** Anatomy of an agentic trace from the Codex/SWE-bench Pro corpus. Each row is one LLM call; per-turn sizes use **medians** across **610** traces. The cached prefix (system prompt, skills/memory, prior turns) is reused; only the new tool output and the model’s decode are active each turn.

By turn 30, context grows to roughly **80K** tokens; the longest exceed **180K**. Each turn typically adds only a few hundred to a few thousand **new** tokens. Dataset-wide ISL:OSL ~**131:1**.

If those prefixes are cached, Prefill on the cached portion is essentially free. The per-turn bill is the delta.

Across 610 traces, median **33** turns:

- **94.2%** cache hit rate (if prefixes can be cached)
- **131:1** input-to-output
- ~**2,242** tokens of context growth per turn
- Median context **12K → 80K** per trace
- Inter-turn delay: median **5.2 s**, P99 **81.4 s**

Local KV offload to CPU DRAM or disk hits two walls:

- **Limited capacity and eviction.** A 100K-token context can occupy GBs (example: ~**3.8 GB** for Kimi-2.5 FP8 KV). Busy instances serving many long sessions saturate local capacity and evict.
- **Cross-instance misses.** To balance load, the router may send the next turn to a replica that never saw the prefix — full recompute.

**Takeaway:** an inference service can no longer be isolated vLLM replicas. Agentic workloads need a **distributed KV cache pool**: larger aggregate capacity and cross-instance hits.

## Distributed KV cache pool with Mooncake Store

Mooncake is an open-source library for KV transfer and distributed storage. vLLM already uses its transfer engine for P/D via `MooncakeConnector`. This step builds a distributed pool with Mooncake Store.

![overall design option C](../../../../assets/vllm/blog/serving/mooncake/03-overall_design_option_C.svg)

**Figure 2.** Overall design. Multiple vLLM instances embed Mooncake clients and share a cluster-wide Mooncake Store. The master manages KV-block metadata, service discovery, and client health; workers transfer KV blocks between GPU HBM and the distributed DRAM/SSD pool over RDMA.

Master: cluster-wide metadata (block hashes, sizes), client health, discovery, dead-node cleanup. Clients on GPU nodes manage local CPU/DRAM/SSD and speak **RDMA** to each other.

Plugs into the existing [`KVConnector`](https://github.com/vllm-project/vllm/blob/db9a84e0cd0e17ab693467ff4a71103abd4b77bf/vllm/distributed/kv_transfer/kv_connector/v1/base.py) (same door as P/D):

- **Scheduler:** hash prompt token blocks, query the master for matching KV blocks, use hits to schedule.
- **Worker:** embed a Mooncake client; background threads move data. GPU KV registered as RDMA buffers → **GPUDirect RDMA** (no SM copy kernel, no CPU staging).

## Design highlights

### SM-free and zero-copy KV transfer with GPUDirect RDMA

Conventional GPU→CPU: `cudaMemcpyAsync` (copy engines; weak on many small transfers) or a dedicated SM copy kernel (can fight attention for SMs). Third path: RDMA NIC + GPUDirect RDMA, HBM ↔ CPU memory, no staging buffer, no SMs. Works for many small KV blocks.

Mooncake Transfer Engine pools multiple RNICs and picks topology-aware paths so transfers aggregate bandwidth across NICs.

### Fully asynchronous transfer

RDMA ops are async; preparing descriptors and issuing reads/writes still burns CPU. Overhead grows with sequence length (more KV blocks). All RDMA runs on a dedicated background I/O thread so the main CPU path that launches GPU kernels is not blocked. From vLLM’s view, the transfer path is fully asynchronous.

### Enabling PD + distributed KV cache pool with MultiConnector

[`MultiConnector`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py) chains independent sub-connectors.

![animation](../../../../assets/vllm/blog/serving/mooncake/04-animation.gif)

**Figure 3.** P/D disaggregation combined with the distributed KV cache pool via `MultiConnector`.

**Prefill:** prepares KV for the P/D connector **and** stores into the pool. Hits: vLLM queries all connectors and can recover matching prefixes from the Mooncake Store connector.

**Decode:** writes into the pool become immediately visible to Prefill. Decode did **not** yet *read* from the pool: vLLM schedules each request onto both a Prefill and a Decode instance; Prefill loads prefix KV from the pool and forwards it over the P/D connector.

Multi-path load (Prefill instance **and** pool) was still in progress, later pointed at [DualPath](https://arxiv.org/abs/2602.21548)-like simultaneous loading.

## Performance

Implementation: [PR #40900](https://github.com/vllm-project/vllm/pull/40900). Artifact benches: [ivanium/vllm mooncake artifacts](https://github.com/ivanium/vllm/tree/feat/mooncake-store-int/scripts/mooncake/artifacts). Two results in the post.

Kimi-2.5 **NVFP4** on GB200, P/D: Prefill **TP4**, Decode **DP8 + EP** — their then-best latency–throughput tradeoff.

### Speeding up real agentic traces

Codex traces, **1P1D**, **12 GPUs**.

![pd compare mooncake vs nixl](../../../../assets/vllm/blog/serving/mooncake/05-pd_compare_mooncake_vs_nixl.png)

**Figure 4.** Mooncake Store vs baseline on realistic Codex traces (1P1D, 12 GB200). Throughput **3.8×**, P50 TTFT **46×** lower, E2E **8.6×** lower; hit rate **1.7% → 92.2%**.

Gains driven by cache hit rate: **1.7%** (system prompt only) → **92.2%** (nearly the whole prefix).

### Scaling out to multiple nodes

Synthetic dataset derived from Codex for controlled scaling.

Settings:

- **20K** common tokens (system instructions)
- **10K** first input
- **2,048** tokens per-turn input
- **900** output tokens
- **30** turns total
- Sessions scale with GPUs: 75 → 150 → 225 → 300 → 375
- Output/input kept ~**1.3%** to match Codex

![pd scaling](../../../../assets/vllm/blog/serving/mooncake/06-pd_scaling.png)

**Figure 5.** Scaling throughput with Mooncake Store from 12 to 60 GB200 GPUs under round-robin routing. Hit rate **>95%** at all scales; near-linear throughput.

**Round-robin** routing on purpose (forces cross-node fetches). Without a pool that pattern would miss constantly and collapse throughput. With Store: hit rate **>95%**, near-linear to **60 GPUs**.

## What's next? (then)

- **Distributed disk offloading.** Hierarchy beyond CPU DRAM to NVMe SSDs and distributed file systems.
- **KV cache offloading for hybrid models.** Mixed attention may need different caching strategies per layer.
- **Cache-aware routing.** Co-design router with the pool: send turns to instances that already hold the prefix; pool as fallback.
- **Further datapath optimization.** Multi-node NVLink in addition to RDMA; DualPath-like simultaneous KV loading from Prefill and Decode.

## Acknowledgements

Lineage: [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend). Chao Lei (Ant Group) initial implementation; Zijing Liu (Inferact) traces and analysis.

Also named: Jiahao Lu, Zuoyuan Zhang, Zihan Tang, Ke Yang (Approaching.AI); Pengbo Zhao, Fuqiao Duan, Tianyu Xu (Huawei); Tianchen Ding, Xuchun Shang, Xingrui Yi, Teng Ma (Alibaba Cloud Computing); Yunxiao Ning, Dejiang Zhu, Shoujian Zheng (Ant Group); Feng Ren (9#AISoft). Broader vLLM and Mooncake communities; Inferact collaboration.

Read after [router.md](router.md) and [large-scale.md](large-scale.md): the router decides *where* the next turn goes; the pool means *another instance does not have to reread*.
