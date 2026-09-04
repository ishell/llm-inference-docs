---
source: https://vllm.ai/blog/2026-05-18-pegaflow
lang: en
fetched: 2026-09-04
---

# vLLM x Novita AI: PegaFlow for Production-Grade External KV Cache

Chinese: [zh/vllm/blog/serving/pegaflow.md](../../../../zh/vllm/blog/serving/pegaflow.md)

2026-05-18. **Novita AI and the vLLM Team**. Repo: [novitalabs/pegaflow](https://github.com/novitalabs/pegaflow). Rust daemon + external KV connector; **no vLLM source edits**, **no long-lived fork**. Examples in the post use `vllm>=0.20.0`. Same door as [mooncake.md](mooncake.md) / [kv-offload.md](kv-offload.md): the pool outlives the engine. Study note; production-oriented evals on the page, not your SLA.

**TL;DR from the page:**

- **2.15×** faster vLLM startup when a **500 GiB** host KV pool was already owned by the external cache service.
- **56%** higher throughput for eight Qwen3-8B instances sharing one host cache instead of eight isolated caches.
- **72%** higher throughput for DeepSeek-V3.2 MLA with TP8 by storing logical KV **once** instead of once per TP rank.
- **194 GB/s** average remote-read throughput for large prefix pulls in an internal RDMA cluster with **8 × 400 Gbps** NICs per node.

Core claim: KV cache should be a **long-lived serving asset**, not temporary state tied to one inference process. Wired through existing `kv_transfer_config`.

Local figures (copyright remains with the original site; study copies):

![architecture](../../../../assets/vllm/blog/serving/pegaflow/01-architecture.png)

![startup time](../../../../assets/vllm/blog/serving/pegaflow/02-startup-time.svg)

![tail latency](../../../../assets/vllm/blog/serving/pegaflow/03-tail-latency.png)

![results overview](../../../../assets/vllm/blog/serving/pegaflow/04-results-overview.svg)

![rdma throughput](../../../../assets/vllm/blog/serving/pegaflow/05-rdma-throughput.svg)

![cache policy comparison](../../../../assets/vllm/blog/serving/pegaflow/06-cache-policy-comparison.png)

## Why KV cache needs a process boundary

KV is one of the most expensive runtime assets in production serving. It can occupy **hundreds of GiB per host**, takes time to allocate and warm, and often **outlives** the request mix that created it.

In-process, that asset dies with the engine: crashes, rolling upgrades, model switches. Restart → host KV pool gone. Fleet switches models → hundreds of GiB of pinned memory may need reallocating and warming before traffic.

PegaFlow moves the KV runtime into a **standalone daemon per machine**. The server owns: host KV pool, SSD cache, topology metadata, RDMA resources, indexing, background tasks. vLLM workers attach with **CUDA IPC** (data) and **gRPC** (local control).

**Figure 1.** PegaFlow beside vLLM. Local CUDA IPC + gRPC; PegaFlow manages pinned memory, SSD, RDMA, and optional cross-node indexing via the **MetaServer**.

Production requirement on the page: **one cache server, many engines and models** on the same host. Different models, TP layouts, and engine versions coexist under **namespace isolation**, sharing the same memory pool, SSD, and cross-node bandwidth.

Failure domains split: vLLM can crash, upgrade, or switch models while the cache stays up. Cache-layer issues need not take down the engine.

## Faster restarts with external cache ownership

Startup-path isolation: **8 × RTX 5090**, **Qwen3-8B TP8**, **dummy weights**, **eager** — strip weight-load and compile, measure a ~**500 GiB** host KV pool only.

- Embedded / in-process: vLLM **71.4 s** to ready.
- Pool pre-owned by PegaFlow: vLLM **33.2 s** after the server was ready → **2.15×** faster vLLM startup, from decoupling long-lived host allocation from the inference process.

**Figure 2.** Those two bars.

## Rust data path and tail-latency stability

The process move was first about lifecycle, sharing, and CPU isolation. Implementing it in **Rust** also bought latency stability.

The data plane avoids Python interpreter overhead, **GIL** contention, and stop-the-world GC. A production cache does more than move bytes: statistics, index uploads, prefetch, health checks, metrics, eviction, SSD management. Those tasks live in the same standalone Rust service and **do not share an interpreter** with vLLM.

**Figure 3.** Tail and average latency under baseline vs GIL-load. The **Rust Tokio** path is much less affected by background load than **Python uvloop** and **Python ZMQ** baselines.

## Pooling cache across instances and nodes

The same logical KV is often stored many times because process, model, or node boundaries hide caches from each other. Patterns named:

- **Multiple small-model instances on one host.** Eight Qwen3-8B on an 8-GPU host can store the **same system prompt eight times**.
- **Wide expert-parallel.** Multiple DP replicas on one machine keep separate prefix caches.
- **MLA + tensor parallelism.** DeepSeek-V3.2: logical latent KV could be stored once; in-process TP8 may store it **once per rank**.
- **Cross-node scheduling.** Hit on Node A, but A is overloaded so the request goes to Node B → prefix recomputed from scratch.

PegaFlow turns those fragments into a shared pool.

On one host, all local instances talk to the same PegaFlow server and share one CPU KV pool. Identical blocks can be stored **once physically** and reused by multiple engines (small-model multi-instance, WideEP DP replicas, TP workers).

Across hosts, a **PegaFlow MetaServer** keeps an **approximate global index**. Nodes fetch remote KV with **one-sided RDMA READ**; after connection setup the remote side spends **zero CPU**. A remote hit can be used much more like a local hit, skipping expensive Prefill recompute.

## Results

Fixed cache **budget**; only **visibility** of that cache changes.

### Single-node multi-instance sharing

Eight Qwen3-8B, one host, **500 GiB** budget.

| Setup | Cache layout | Throughput | Mean TTFT | Request hit rate |
| --- | --- | ---: | ---: | ---: |
| PegaFlow | 500 GiB shared pool | 11.97 req/s | 5.26 s | 52.35% |
| In-process | 8 × 62.5 GiB isolated pools | 7.68 req/s | 8.22 s | 11.77% |

Not more memory — the same 500 GiB, now one pool instead of eight islands. Throughput **+56%**, mean TTFT **−36%**, request hit rate **~4.4×**.

### MLA logical KV deduplication

DeepSeek-V3.2 MLA, TP8, **500 GiB** budget.

| Setup | Cache layout | Throughput | Mean TTFT | Request hit rate |
| --- | --- | ---: | ---: | ---: |
| PegaFlow | Logical KV stored once | 1.81 req/s | 35.66 s | 97.23% |
| In-process | KV stored per TP rank | 1.05 req/s | 60.88 s | 65.18% |

Not storing the same logical KV per rank expands **usable** capacity. Throughput **+72%**, mean TTFT **−41%**, hit rate near the practical upper bound for that trace.

**Figure 4.** The two fixed-budget local-sharing experiments. Effective capacity goes up because the same KV budget is visible across isolation boundaries.

### Cross-node RDMA sharing

Internal production cluster: **8 × 400 Gbps** RDMA NICs per node. Sample of thousands of recent online remote reads. Prefix pulls **≥ 1 GiB**:

- average effective throughput **194 GB/s**
- **P99 250 GB/s**
- peak **261.6 GB/s**

At that rate a **24 GiB** KV segment pulls from a remote node in roughly **100 ms** — replacing Prefill that would otherwise cost **seconds** of GPU time. Remote hits are not merely “better than a miss”; they can be fast enough to sit **on** the serving path.

**Figure 5.** Effective throughput for those large remote reads; 24 GiB ≈ 100 ms at the measured average.

## Three-level cache hierarchy

Pooling helps; host memory is still finite. Long reuse-distance prefixes get evicted; simple LRU is disrupted by **scan-like** traffic (many one-time blocks).

| Level | Medium | Access path | Typical role |
| --- | --- | --- | --- |
| L1 | Local pinned DRAM | Local memory | Fast local KV reuse |
| L2 | Remote DRAM | RDMA READ | Cross-node cache sharing |
| L3 | Local SSD | `io_uring` | Large-capacity spillover |

SSD cache is Rust on `io_uring`. Internal tests: one SSD ~**6.9 GB/s** peak read. Online steady-state kept around **6.5–6.6 GB/s** per disk — about **5%** peak bandwidth traded for stabler tail latency. **RAID0** across disks: throughput scales **approximately linearly**.

Scan-heavy workloads or smaller budgets can enable **TinyLFU** admission: admit a block only when it is likely to be reused, so one-time traffic does not flood the cache. **Disabled by default** — best admission policy depends on workload shape. On several internal traces it substantially beat LRU when the cache was small or scan pressure was high.

**Figure 6.** Policy comparison at small cache sizes. Scan-heavy traces make recency-only policies ineffective; admission-aware policies (TinyLFU) protect the cache from one-time blocks.

## Distance from the theoretical hit-rate ceiling

Online hit rate alone misleads. **3%** may be good if the workload has almost no reuse; **90%** may still be slack if the theoretical ceiling is much higher. The operator question is: **how close are we to the best hit rate this workload could reasonably achieve?**

PegaFlow estimates the ceiling online with **HyperLogLog**:

```
r* = (N − U) / N
```

`N` = block requests in the window; `U` = first-seen unique blocks. A **24-hour** window uses **< 1 MiB** with roughly **0.8%** error.

Rolling HLL windows, defaults: **15 minutes**, **1 hour**, **24 hours**. Put measured hit rate and `r*` on one dashboard:

- Close to the ceiling → adding capacity may not help much.
- Far below → room in capacity, admission, prefetch, or cross-node discovery.
- Ceiling itself low → the workload has little reuse; the bottleneck is not primarily the cache implementation.

## Integrating through the external connector

Many external KV systems want invasive edits to scheduler, block manager, or attention kernels. PegaFlow uses vLLM’s **external KV connector**. Configure `kv_transfer_config`; load the package dynamically with `kv_connector_module_path`. Runtime takeover of key KV operations; **no vLLM source change**, **no fork**.

From vLLM’s side, PegaFlow is not a replacement engine. It is an external cache backend on the KV transfer interface. vLLM still owns scheduling, execution, batching, and the OpenAI-compatible serving path. That boundary lets PegaFlow iterate the Rust data plane / SSD / RDMA / index / connector on its own clock.

## Quick start

```bash
uv pip install pegaflow-llm        # CUDA 12
uv pip install pegaflow-llm-cu13   # CUDA 13
```

Single-node server, pinned host memory + SSD:

```bash
pegaflow-server \
  --pool-size 30gb \
  --ssd-cache-path <ssd-cache-file-path> \
  --ssd-cache-capacity 512gb
```

Online deployments: add **`--use-hugepages`**. Huge pages should be **reserved in advance**. They speed CPU pinned-memory allocation and reduce RDMA **MTT** pressure by lowering address-translation overhead during registration and transfer.

Multi-node: start **MetaServer** first, then a PegaFlow server on each node with RDMA. When P2P is on, each server’s **`--addr` must be a routable IP**, not `0.0.0.0` or `127.0.0.1` — peers use it for the gRPC handshake and block queries.

```bash
pegaflow-metaserver --addr 0.0.0.0:50056
```

```bash
pegaflow-server \
  --addr this-node:50055 \
  --pool-size 30gb \
  --ssd-cache-path <ssd-cache-file-path> \
  --nics mlx5_0 mlx5_1 \
  --metaserver-addr http://metaserver-host:50056
```

Connect vLLM (`vllm>=0.20.0` in the post’s examples):

```bash
vllm serve <model> \
  --kv-transfer-config '{
    "kv_connector": "PegaKVConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "pegaflow.connector"
  }'
```

`PEGAFLOW_HOST` and `PEGAFLOW_PORT` point the connector at the service. Defaults: `http://127.0.0.1` and `50055`.

Repo also documents installation, server config, P2P RDMA, metrics, and connector examples.

## Public reference benchmark

In-repo KV cache bench: **H800**, **Llama-3.1-8B**, **8** prompts, **10K-token** Prefill, **1-token** Decode, **4.0 req/s**. Warm cache: mean TTFT **572.5 ms → 61.5 ms**; P99 TTFT **1113.7 ms → 77.0 ms**.

## Acknowledgements

Novita AI team for building and productionizing PegaFlow. vLLM maintainers and the broader community for discussions, reviews, and the connector infrastructure.
