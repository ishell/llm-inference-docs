---
source: https://vllm.ai/blog/2026-05-06-mooncake-store
lang: en
fetched: 2026-08-31
---

# Serving Agentic Workloads with vLLM × Mooncake

2026-05-06. https://vllm.ai/blog/2026-05-06-mooncake-store  
Study note, not a full reprint. Figures on the original page. Numbers are a demo on Codex / SWE-bench Pro traces.

## Why agents break isolated replicas

Agent loops (reason → tool → append → reason again) grow a huge **shared prefix**. Codex / GPT-5.4 traces on SWE-bench Pro (610 traces, median ~33 turns; dataset open-sourced with the post):

- By turn 30, median context ~**80K** tokens; longest >**180K**
- Each turn adds only hundreds to a few thousand new tokens
- **94.2%** would-be cache hit if prefixes are stored
- Average ISL:OSL ~**131:1**
- ~**2,242** tokens of context growth per turn
- Inter-turn delay: median **5.2 s**, P99 **81.4 s**

Cached prefix prefill is almost free. The per-turn bill is the delta.

Local CPU/disk KV offload hits two walls:

1. **Capacity / eviction.** A 100K-token context can be several GB (example: ~**3.8 GB** for Kimi-2.5 FP8 KV). Busy instances evict long prefixes.
2. **Cross-instance miss.** Load balancing may send the next turn to a replica that never saw the prefix.

Agent serving needs a **cluster-wide KV pool**.

## Mooncake Store in vLLM

Mooncake already powers P/D KV movement via `MooncakeConnector`. This post adds **Mooncake Store**: master for block hashes / discovery / health; clients on GPU nodes pooling DRAM/SSD; RDMA between clients.

Plugs into the existing **KVConnector**:

- Scheduler hashes prompt blocks, queries the master, uses hits to schedule.
- Each GPU worker embeds a client; background threads move data. KV in HBM is registered as RDMA buffers → **GPUDirect RDMA**, no SM copy kernels, no CPU staging.

Three design bets:

- **SM-free zero-copy** via NIC + GPUDirect; Transfer Engine pools RNICs and picks topology-aware paths.
- **Fully async I/O thread** so descriptor prep does not stall the CPU path that launches GPU kernels.
- **`MultiConnector`** stacks P/D + store. Prefill writes to both; hits can be recovered from the store. Decode writes become visible to prefill. Decode did **not** yet read from the pool: prefill loads prefix KV and forwards it over the P/D connector. Dual-path load (prefill instance + pool) was still in progress.

## Results (demo)

Kimi-2.5 NVFP4 on GB200, P/D: prefill **TP4**, decode **DP8+EP**.

**Codex traces, 1P1D, 12×GB200:** ~**3.8×** throughput, ~**46×** lower P50 TTFT, ~**8.6×** lower e2e. Hit rate **1.7% → 92.2%**.

**Scale 12→60 GB200** on a Codex-shaped synthetic mix, **round-robin** routing (forces cross-node fetches): hit rate **>95%**, near-linear throughput. Without the pool that routing pattern would miss constantly.

Roadmap in the post: disk/DFS offload, hybrid-attention cache policies, cache-aware routing, multi-path NVLink+RDMA. Implementation lineage: vLLM-Ascend.

Read after [router.md](router.md) and [large-scale.md](large-scale.md): the router decides *where* the next turn goes; the pool means *another instance does not have to reread*.
