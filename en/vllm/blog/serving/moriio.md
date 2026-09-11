---
source: https://vllm.ai/blog/2026-04-07-moriio-kv-connector
lang: en
fetched: 2026-09-05
---

# Next-Level Inference: Single-Node Prefill–Decode Disaggregation with MORI-IO

Chinese: [zh/vllm/blog/serving/moriio.md](../../../../zh/vllm/blog/serving/moriio.md)  
Source: https://vllm.ai/blog/2026-04-07-moriio-kv-connector

2026-04-07. AMD and Embedded LLM. Study extract, not an official reprint. Connector: `MoRIIOConnector` (MORI = Modular RDMA Interface). Landing [PR #29304](https://github.com/vllm-project/vllm/pull/29304). Library: [ROCm/mori](https://github.com/ROCm/mori). Workload: **Qwen3-235B-A22B-FP8**, **8 req/s**, input **2000** / output **1000**, one **8× MI300X** node. Headline: **~2.5×** goodput vs collocated serving on the same 8 GPUs. Original Figure 3 is an interactive Plotly slider; this note keeps Table 3 at the default SLO and skips the widget.

**TL;DR.** Prefill and decode fight over the same GPUs, so ITL spikes under load. Disaggregate them on a single 8-GPU MI300X node with AMD’s MORI-IO connector — **2.5×** higher goodput than standard collocated serving on the same 8 GPUs, with stable token generation. Full config: Table 3 and [Experimental Details](#experimental-details).

Local figures (copyright remains with the original site; study copies):

![read mode request flow diagram](../../../../assets/vllm/blog/serving/moriio/01-read-mode-request-flow-diagram.svg)

**Figure 1.** Read mode request flow. The proxy dispatches serially — step 3 (prefill response) must complete before step 4 (dispatch to decode).

![write mode request flow diagram](../../../../assets/vllm/blog/serving/moriio/02-write-mode-request-flow-diagram.svg)

**Figure 2.** Write mode request flow. The proxy fires both prefill and decode concurrently (step 2); prefill pushes KV layer-by-layer via RDMA WRITE (step 3) while decode waits.

![SLO attainment](../../../../assets/vllm/blog/serving/moriio/03-SLO-attainment.png)

**Figure 4.** SLO attainment (% of requests meeting both TTFT and ITL targets) across request rates. Both disaggregated modes beat all standard serving configs at every tested rate.

![read mode kv transfer sequence diagram](../../../../assets/vllm/blog/serving/moriio/04-read-mode-kv-transfer-sequence-diagram.svg)

**Figure 5.** Read mode timing. Overhead 1 (proxy serialization) and Overhead 2 (RDMA READ) add to TTFT.

![write mode kv transfer sequence diagram](../../../../assets/vllm/blog/serving/moriio/05-write-mode-kv-transfer-sequence-diagram.svg)

**Figure 6.** Write mode timing. RDMA WRITE overlaps prefill compute, so Overhead 2 does not add full wall-clock TTFT.

## Introduction

The previous MoE playbook [[1]](#ref-1) walked through spreading a large model across one 8-GPU AMD Instinct MI300X node with Tensor, Pipeline, Data, and Expert Parallelism. This post is the next bottleneck: Prefill–Decode interference when concurrency rises. AMD’s MORI-IO lets you disaggregate on **one box** — higher goodput, more predictable latency, no multi-node cluster required.

HBM is full, compute is balanced, vLLM looks healthy — until you raise concurrency. Then Inter-Token Latency (ITL) spikes. Prefill and decode are different jobs fighting over the same GPUs.

**Prefill is compute-bound:** the whole prompt in parallel, large GEMMs, cost scaling with input length.

**Decode is memory-bandwidth-bound:** one token at a time, repeatedly loading weights from HBM, low compute per byte.

Sharing an instance, they block each other. A fat prefill stutters dozens of decode streams; decode delays new prefills. Neither phase runs efficiently or predictably.

## Key Highlights

- **~2.5× higher goodput on the same hardware.** SLO-compliant throughput on one 8-GPU MI300X node by separating prefill and decode.
- **ITL spikes under load go away.** Dedicated decode GPUs keep token generation stable.
- **Single-node disaggregation — no cluster.** Prefill–Decode (PD) disaggregation entirely inside one node.
- **MORI-IO for fast KV transfer.** RDMA-based KV movement between phases.
- **Two modes, real trade-offs.** Write is fastest (lower TTFT); read is simpler to orchestrate. Both beat standard serving by a wide margin.

## The Misconception: "Disaggregation is Only for Datacenter Clusters"

“Prefill–Decode (PD) Disaggregation” usually conjures dedicated prefill nodes, dedicated decode nodes, and an RDMA fabric. The reflex: “I only have one 8-GPU node — this is not for me.”

That leaves single-node goodput on the table. PD can live entirely inside one 8-GPU box. If you care about strict latency SLOs, it is often the right move.

Split the phases: e.g. four GPUs for prefill, four for decode. Size, parallelize, and schedule each independently. Head-of-line blocking in a monolith goes away.

The hard part is the handoff. Prefill’s KV must reach decode — gigabytes. A slow transfer eats the gain.

AMD’s answer is **MORI-IO**, an RDMA KV connector in vLLM [[4]](#ref-4), on top of open-source MORI (Modular RDMA Interface) [[5]](#ref-5).

> **Scope:** single-node PD on one box with 8 GPUs, to raise goodput on hardware you already have.

## The Architecture: Serving with PD Disaggregation

Splitting the node is a three-piece microservice layout (Table 1):

| Component | Role |
|-----------|------|
| Prefill instance | Processes the prompt and produces KV (GPUs 0–3) |
| Decode instance | Autoregressive tokens from transferred KV (GPUs 4–7) |
| Proxy server | Client entry; routes to prefill first, then decode |

<p align="center"><em>Table 1. PD disaggregation deployment components.</em></p>

Both modes move KV from prefill to decode. They differ in **who initiates** and **when**:

- **Read mode:** Proxy waits for prefill, then forwards KV block locations to decode. Decode **pulls** via RDMA before generating.
- **Write mode:** Proxy dispatches prefill and decode together. As prefill finishes each layer, it **pushes** KV into decode’s pre-allocated memory — decode can start as soon as prefill finishes.

### Request Flow in Detail

The two MORI-IO transfer modes differ in **who starts the RDMA** and **how the proxy orchestrates**. Switch: `VLLM_MORIIO_CONNECTOR_READ_MODE`.

#### Read Mode — Decode Pulls KV Cache

Enable: `export VLLM_MORIIO_CONNECTOR_READ_MODE=1`

Proxy dispatches **serially**: wait for prefill, extract remote block IDs, forward them to decode. Decode RDMA-reads prefill’s memory. Figure 1.

Time-ordered sequence:

1. **Client → Proxy**
2. **Proxy → Prefill** (`max_tokens=1`)
3. **Prefill → Proxy:** `remote_block_ids` and `remote_engine_id`
4. **Proxy → Decode** with those IDs
5. **Decode pulls KV** (`WAITING_FOR_REMOTE_KVS`): RDMA read against prefill. Scheduler skips the request each step until the transfer completes.
6. **Decode → Prefill (cleanup):** Decode notifies prefill to free blocks.
7. **Decode → Proxy → Client:** tokens over SSE.

#### Write Mode — Prefill Pushes KV Cache (Default)

Enable: unset `VLLM_MORIIO_CONNECTOR_READ_MODE`, or `=0`.

Proxy fires prefill and decode **concurrently**. Prefill pushes KV layer-by-layer into decode’s pre-allocated memory. Figure 2.

Time-ordered sequence:

1. **Client → Proxy**
2. **Proxy → Prefill AND Proxy → Decode (concurrent):** each request carries the other’s connection details. Proxy does **not** await the prefill response.
3. **Prefill pushes KV:** `save_kv_layer` issues an RDMA write into decode’s blocks. For chunked prefill, blocks accumulate until the last chunk, then the write starts.
4. **Decode waits** (`WAITING_FOR_REMOTE_KVS`): scheduler polls `pop_finished_write_req_ids` until all blocks arrive.
5. **Decode generates:** request moves to the ready queue.
6. **Decode → Proxy → Client:** SSE.

The proxy’s key branch:

```python
# examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py

if TRANSFER_TYPE == "READ":
    # Serial: wait for prefill to finish, extract block IDs for decode to pull.
    prefill_response = await send_prefill_task
    req_data["kv_transfer_params"]["remote_engine_id"] = prefill_response[
        "kv_transfer_params"
    ]["remote_engine_id"]
    req_data["kv_transfer_params"]["remote_block_ids"] = prefill_response[
        "kv_transfer_params"
    ]["remote_block_ids"]

# In WRITE mode, execution falls through here immediately —
# no await on send_prefill_task. Both phases are already in flight.
decode_request_task = asyncio.create_task(
    start_decode_request(decode_instance_endpoint["request_address"], req_data, request_id)
)
```

Read **must** relay `remote_block_ids` through the proxy. Write does not: prefill owns the write and pushes to decode’s addresses.

### Read Mode vs. Write Mode: At a Glance

`MoRIIOConnector` owns the KV handoff. Before the first RDMA between a pair, MORI-IO does a one-time ZMQ metadata exchange: KV base addresses, block sizes, per-layer tensor strides. Handshake runs in a **background thread** so it does not block the engine loop; the RDMA session is cached for later requests.

Same handshake and RDMA transport. Differences are proxy dispatch and transfer direction (Table 2):

| Property | Read Mode | Write Mode |
|----------|-----------|------------|
| `VLLM_MORIIO_CONNECTOR_READ_MODE` | `=1` | Unset (or `=0`) |
| RDMA direction | Decode pulls from prefill | Prefill pushes to decode |
| Proxy dispatch | Serial (await prefill → dispatch decode) | Concurrent (both in parallel) |
| `remote_block_ids` via proxy | Required | Not required |
| KV cleanup signal | Decode notifies prefill after pull | Prefill tracks write completion per request |

<p align="center"><em>Table 2. Key differences of Read mode and Write mode.</em></p>

## Results: 2.5x Higher Goodput

### Why Goodput, Not Throughput

Raw throughput can hide SLO misses. Primary metric is **goodput**, DistServe-style [[3]](#ref-3):

**Goodput** = maximum request rate (req/s) such that requests satisfy both TTFT &lt; *T_ttft* and ITL &lt; *T_itl*.

Cost and quality in one number. Targets: **TTFT &lt; 1 second**, **ITL &lt; 50 ms per token**. Both must hold.

### Headline Result

**Figure 3** is goodput at request rate = 8 (original Plotly: one bar per request; gray bars miss at least one SLO; sliders change thresholds. Default TTFT &lt; 1 s, ITL &lt; 50 ms. Widget omitted here). Default-threshold counts are Table 3:

| Metric | Standard (1× TP8) | Standard (2× TP4) | MORI-IO Read (1P+1D) | MORI-IO Write (1P+1D) |
|--------|-------------------|---------------------|---------------------|----------------------|
| Requests meeting both SLOs | 26/100 | 30/100 | 70/100 | 73/100 |
| Primary failure mode | ITL spikes (P99 ITL ≫ 50 ms) | ITL spikes (bimodal: ~30 ms and ~150 ms) | TTFT exceeds 1 s for some | TTFT exceeds 1 s for some |
| Relative goodput | 0.9× | 1× | 2.4× | 2.5× |

<p align="center"><em>Table 3: SLO attainment at request rate = 8. Workload: Qwen3-235B-A22B-FP8, ISL=2000, OSL=1000, 8 req/s, 100 requests. See Experimental Details. Standard (2× TP4) is the baseline for relative goodput.</em></p>

Standard serving dies on ITL in two clusters — the high cluster at ~150 ms far exceeds 50 ms. Both disaggregated modes **eliminate ITL violations**; leftovers are TTFT as rate climbs. Write edges read (73 vs 70) because concurrent dispatch keeps more TTFT under 1 s.

### SLO Attainment Across Request Rates

**Figure 4**, rates 0.5–10:

- **Standard (1× TP8):** ITL violations from low rates, dominating the sweep. 26/100 at rate = 8.
- **Standard (2× TP4):** 100% at 0.5 → ~60% at 1 → ~25% by 2, then plateau. ITL violations saturate early.
- **MORI-IO Read (1P+1D):** 100% through ~rate 5, then down to ~44% at 10 as TTFT exceeds.
- **MORI-IO Write (1P+1D):** 100% through ~rate 5.5, then ~46% at 10 (TTFT).

## Understanding the Trade-offs

### Why ITL Improves

In a shared engine, prefill and decode compete inside each batch. One prefill forward is much longer than a decode step. Every decode in that batch waits → ITL inflates.

With disaggregation the decode engine runs **only** decode batches. No compute-heavy prefill interrupts the cadence. ITL is stable regardless of arriving traffic. Same ITL benefit in **both** modes.

### Why TTFT Gets Worse

Disaggregation adds work on the path to the first token. Standard:

```
TTFT = queue + prefill_forward_pass + sample_T1 + detokenize + SSE_encode + network
```

Read inserts two extra steps (Figure 5):

```
TTFT = queue(prefill) + prefill_forward_pass
     + [proxy serialization: await prefill, dispatch to decode]  <- Overhead 1
     + RDMA transfer (WAITING_FOR_REMOTE_KVS)                   <- Overhead 2
     + queue(decode) + sample_T1 + detokenize + SSE_encode + network
```

Write (Figure 6):

```
TTFT ≈ max(
           queue(prefill) + prefill_forward_pass + RDMA_write_time,
           queue(decode)
       ) + sample_T1 + detokenize + SSE_encode + network
```

Write drops Overhead 1. Concurrent dispatch overlaps decode queue wait with prefill compute. The remaining RDMA cost is structurally like the RDMA read in read mode.

#### Overhead 1: Proxy Serialization (Read Mode Only)

In read mode the proxy awaits the full prefill response before dispatching decode. Prefill compute plus a proxy round-trip land in client-visible TTFT. Write skips this — decode is already in flight.

```python
# examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py

if TRANSFER_TYPE == "READ":
    # In read mode, prefill and decode are executed serially.
    prefill_response = await send_prefill_task
    req_data["kv_transfer_params"]["remote_engine_id"] = prefill_response[
        "kv_transfer_params"
    ]["remote_engine_id"]
    req_data["kv_transfer_params"]["remote_block_ids"] = prefill_response[
        "kv_transfer_params"
    ]["remote_block_ids"]
```

#### Overhead 2: RDMA Transfer Wait

Decode enters `WAITING_FOR_REMOTE_KVS`. The scheduler skips it every step until RDMA completes, then moves it to the ready queue.

```python
# vllm/v1/request.py

WAITING_FOR_REMOTE_KVS = enum.auto()

# vllm/v1/core/sched/scheduler.py
# KVTransfer: skip request if still waiting for remote kvs.

if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
    is_ready = self._update_waiting_for_remote_kv(request)
    if is_ready:
        request.status = RequestStatus.WAITING
    else:
        logger.debug("%s is still in WAITING_FOR_REMOTE_KVS state.",
                     request.request_id)
        self.waiting.pop_request()
        skipped_waiting_requests.prepend_request(request)
        continue
```

In read mode this wait starts **after** prefill has finished. In write mode it starts as soon as the decode request arrives — overlapping ongoing prefill on the other instance.

**Bottom line:** stable ITL, longer wait for the first token. Read: TTFT grows by at least one full prefill forward (proxy serialization) plus RDMA. Write: no proxy serialization — TTFT grows only by RDMA, overlapped with prefill, smaller net penalty. ITL gains are identical.

### When Should You Use This?

Table 4:

| Your situation | Recommendation |
|----------------|----------------|
| ITL p99 exceeds SLO under production load | Disaggregate — primary use case |
| TTFT is the binding constraint (e.g. chatbot UX) | Standard serving may be preferable |
| High concurrency with long prompts | Disaggregate — worst prefill interference |
| Low request rates with short prompts | Standard serving is sufficient |

<p align="center"><em>Table 4: Deployment decision guide.</em></p>

## How to Set It Up

Three pieces: prefill instance, decode instance, proxy. Official disagg prefill docs: [[2]](#ref-2).

### Prefill Instance

KV producer (`kv_role: kv_producer`). Computes KV for the prompt; decode reads it via RDMA.

```bash
vllm serve <model> \
  ...
  --gpu_memory_utilization 0.9 \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "proxy_ip": "127.0.0.1",
      "proxy_ping_port": "36367",
      "http_port": "20005",
      "handshake_port": "6301",
      "notify_port": "6105"
    }
  }'
```

On startup the instance ZMQ-registers with the proxy (role, HTTP address, handshake/notify ports, parallelism) and keeps pinging so the proxy can detect unavailability.

### Decode Instance

KV consumer (`kv_role: kv_consumer`). Receives the request after prefill, then pulls KV via RDMA.

```bash
vllm serve <model> \
  ...
  --gpu_memory_utilization 0.9 \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
      "proxy_ip": "127.0.0.1",
      "proxy_ping_port": "36367",
      "http_port": "40005",
      "handshake_port": "7301",
      "notify_port": "7501"
    }
  }'
```

### Proxy Server

Lightweight HTTP orchestrator. Listens for ZMQ registrations on `proxy_ping_port`; round-robin routing.

```bash
python examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py
```

In READ mode the proxy waits for prefill, extracts `remote_block_ids`, and passes them to decode.

### Port Reference

Table 5. `MoRIIOConfig` (`moriio_common.py`) applies per-rank offsets:

| Port | Purpose |
|------|---------|
| `proxy_ping_port` | ZMQ registration with the proxy |
| `http_port` | vLLM HTTP; proxy forwards inference here |
| `handshake_port` | One-time metadata: consumer gets producer KV layout |
| `notify_port` | Per-request: prefill signals decode that blocks are ready |

<p align="center"><em>Table 5: MORI-IO port assignments.</em></p>

## Experimental Details

### Setup

Reproduce from Dockerfiles: `Dockerfile.rocm_base` (MORI commit `2d02c6a9` from [ROCm/mori](https://github.com/ROCm/mori)) and `Dockerfile.rocm` (vLLM main, [vllm-project/vllm](https://github.com/vllm-project/vllm)).

**Hardware:**

- GPU: 8× AMD Instinct MI300X (gfx942)
- CPU: 2× AMD EPYC 9654 96-Core Processor

**Software stack:**

- ROCm Driver: 6.10.5 (AMDGPU)
- Container: rocm/vllm-dev (ROCm 7.0.51831-a3e329ad8)
- vLLM: 0.16.0rc1.dev1+gc46b0cd0a (git sha: c46b0cd0a)
- PyTorch: 2.9.1+git8907517 (ROCm 7.0.51831-a3e329ad8)
- MORI library: commit [`c365eaed`](https://github.com/ROCm/mori/commit/c365eaed02b13e6b8f2e9c8215b21516d86856ce)

**Benchmark configuration:**

- Model: Qwen/Qwen3-235B-A22B-FP8
- Input sequence length: 2000 tokens
- Output sequence length: 1000 tokens
- Dataset: random
- Workload: 100 total requests
- Request rate: 0.5 to 10 (step 0.5)

### Baseline Configurations

Table 6:

| Configuration | Description |
|---------------|-------------|
| Standard (1× TP8) | One vLLM on all 8× MI300X (TP=8) with expert parallelism. Mixed prefill+decode on one engine. |
| Standard (2× TP4) | Two identical instances, 4× MI300X each (TP=4) + expert parallelism. Round-robin proxy. Both mixed. |
| MORI-IO Read (1P+1D) | Prefill GPUs 0–3, decode 4–7, each TP=4 + EP. `VLLM_MORIIO_CONNECTOR_READ_MODE=1` on both. Serial proxy; decode RDMA-pulls KV. Prefix caching off. |
| MORI-IO Write (1P+1D) | Same split. Write-mode KV. Stateful two-phase proxy. Prefix caching **off as required by MORI-IO**. |

<p align="center"><em>Table 6: Baseline configurations.</em></p>

> **Why this baseline?** Standard (2× TP4) and the disaggregated configs use the same 8× MI300X split into two 4-GPU groups. The only difference is mixed vs dedicated workloads. Standard (1× TP8) is an extra reference with all 8 GPUs in one engine.

**Generalizability:** Results use a MoE model (Qwen3-235B-A22B-FP8). Prefill/decode interference is fundamental to transformer inference and applies to dense models too. MoE tends to amplify it: expert routing adds per-step compute variance, so ITL jitter is worse.

## Conclusions and Way Forward

PD disaggregation is not datacenter-only — it measures on a single 8-GPU node. Dedicated GPUs plus MORI-IO RDMA KV transfer: **2.5×** goodput, ITL violations gone.

### What's Next

- **Multi-node:** Prefill and decode can span hosts. MORI-IO already uses RDMA over the fabric; same connector claimed across hosts with no code change.
- **Per-phase tuning:** Prefill can chase compute (larger token budgets, chunked prefill); decode can chase latency (smaller batches, stricter scheduling). Impossible when collocated.

## Appendix: Reproducible Configurations

Nightly images: [rocm/vllm-dev](https://hub.docker.com/r/rocm/vllm-dev). Or build from `Dockerfile.rocm_base` / `Dockerfile.rocm` (MORI [2d02c6a9](https://github.com/ROCm/mori/commit/2d02c6a9), vLLM [c46b0cd0a](https://github.com/vllm-project/vllm/commit/c46b0cd0a)).

Full CLI for Qwen3-235B-A22B-FP8 on MI300X:

### Standard Serving

```bash
# Instance 1 (GPU 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_ROCM_USE_AITER=1 vllm serve Qwen/Qwen3-235B-A22B-FP8 \
  -tp 4 \
  --enable-expert-parallel \
  --max-model-len 16384 \
  --max-num-batched-tokens 8192 \
  --distributed-executor-backend mp \
  --no-enable-prefix-caching \
  --port 8100

# Instance 2 (GPU 4-7)
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_ROCM_USE_AITER=1 vllm serve Qwen/Qwen3-235B-A22B-FP8 \
  -tp 4 \
  --enable-expert-parallel \
  --max-model-len 16384 \
  --max-num-batched-tokens 8192 \
  --distributed-executor-backend mp \
  --no-enable-prefix-caching \
  --port 8200

# Proxy
cd <path_to>/vllm
python benchmarks/disagg_benchmarks/round_robin_proxy.py
```

### Disaggregated Serving

```bash
# Prefill instance (GPU 0-3)
export VLLM_MORIIO_CONNECTOR_READ_MODE=1    # unset for write mode
export VLLM_ROCM_USE_AITER=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3
export MORI_DISABLE_AUTO_XGMI=1
export MORI_IO_ENABLE_NOTIFICATION=0

vllm serve Qwen/Qwen3-235B-A22B-FP8 \
  -tp 4 \
  --enable-expert-parallel \
  --port 20005 \
  --max-num-batched-tokens 4096 \
  --distributed-executor-backend mp \
  --gpu_memory_utilization 0.9 \
  --max-model-len 16384 \
  --max_num_seqs 64 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "proxy_ip": "127.0.0.1",
      "proxy_ping_port": "36367",
      "http_port": "20005",
      "handshake_port": "6301",
      "notify_port": "6105"
    }
  }'

# Decode instance (GPU 4-7)
export VLLM_MORIIO_CONNECTOR_READ_MODE=1    # unset for write mode
export VLLM_ROCM_USE_AITER=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HIP_VISIBLE_DEVICES=4,5,6,7
export MORI_DISABLE_AUTO_XGMI=1
export MORI_IO_ENABLE_NOTIFICATION=0

vllm serve Qwen/Qwen3-235B-A22B-FP8 \
  -tp 4 \
  --enable-expert-parallel \
  --port 40005 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 4096 \
  --distributed-executor-backend mp \
  --gpu_memory_utilization 0.9 \
  --max-model-len 16384 \
  --max_num_seqs 64 \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
      "proxy_ip": "127.0.0.1",
      "http_port": "40005",
      "proxy_ping_port": "36367",
      "handshake_port": "7301",
      "notify_port": "7501"
    }
  }'

# Proxy
cd <path_to>/vllm
python examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py
```

## Acknowledgements

**AMD:** Hongxia Yang, Gilbert Lei, Mingzhi Liu, Niko Ma, Tian Di, Randall Smith, Feiyue Zhai, Peng Sun, and the MORI team.

**Embedded LLM:** Pin Siang Tan, Jun Kang Chow, Ye Hur Cheong, Vensen Mu, Jeff Aw, Tun Jian Tan and the Embedded LLM team.

## References

1. <a id="ref-1"></a> AMD and Embedded LLM, "The vLLM MoE Playbook: A Practical Guide to TP, DP, PP and Expert Parallelism" <https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html>
2. <a id="ref-2"></a> vLLM Disaggregated Prefill Documentation <https://docs.vllm.ai/en/latest/features/disagg_prefill/>
3. <a id="ref-3"></a> DistServe: Maximizing Goodput in LLM Serving <https://haoailab.com/blogs/distserve/>
4. <a id="ref-4"></a> MORI-IO Connector PR #29304 <https://github.com/vllm-project/vllm/pull/29304>
5. <a id="ref-5"></a> MORI (Modular RDMA Interface) <https://github.com/ROCm/mori>

## Disclaimer

Testing at **Mar. 12, 2026**, measuring inference goodput on AMD Instinct MI300X.

**Hardware Configuration**

- MI300X: AMD EPYC 9654 96-Core Processor server with 8× AMD Instinct MI300X (192GB, 750W) GPUs, NPS1 (1 NUMA per socket), 2.2TiB (24 DIMMs, 4800 MT/s memory, 96 GiB/DIMM)

**Software Configuration**

Ubuntu 22.04 LTS with Linux kernel 5.15.0-153-generic, ROCm Driver 6.10.5 (AMDGPU), ROCm 7.0.51831-a3e329ad8, PyTorch 2.9.1+git8907517, vLLM 0.16.0rc1.dev1+gc46b0cd0a, MORI library commit c365eaed

Server manufacturers may vary configurations, yielding different results. Performance may vary based on configuration, software, vLLM version, and the use of the latest drivers and optimizations.

[router.md](router.md) is the cross-pod P/D gateway; this post is P/D **inside one box**. Same connector family as Mooncake / NIXL / CPU offload, different transport.
