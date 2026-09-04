---
source: https://vllm.ai/blog/2026-04-07-moriio-kv-connector
lang: en
fetched: 2026-09-04
---

# Next-Level Inference: Single-Node Prefill–Decode Disaggregation with MORI-IO

Chinese: [zh/vllm/blog/serving/moriio.md](../../../../zh/vllm/blog/serving/moriio.md)

2026-04-07. **One 8-GPU MI300X node.** Connector: `MoRIIOConnector` (MORI = Modular RDMA Interface). Demo: **Qwen3-235B-A22B-FP8**, **8 req/s**, input **2000** / output **1000**. Headline: **2.5×** goodput vs collocated serving on the **same 8 GPUs**. Study note. PR: [#29304](https://github.com/vllm-project/vllm/pull/29304). Library: [ROCm/mori](https://github.com/ROCm/mori). Prior AMD/Embedded LLM MoE playbook is the TP/DP/PP/EP sibling.

Local figures (copyright remains with the original site; study copies):

![read mode request flow diagram](../../../../assets/vllm/blog/serving/moriio/01-read-mode-request-flow-diagram.svg)

![write mode request flow diagram](../../../../assets/vllm/blog/serving/moriio/02-write-mode-request-flow-diagram.svg)

![SLO attainment](../../../../assets/vllm/blog/serving/moriio/03-SLO-attainment.png)

![read mode kv transfer sequence diagram](../../../../assets/vllm/blog/serving/moriio/04-read-mode-kv-transfer-sequence-diagram.svg)

![write mode kv transfer sequence diagram](../../../../assets/vllm/blog/serving/moriio/05-write-mode-kv-transfer-sequence-diagram.svg)

## The fight on one box

Prefill is **compute-bound** (large GEMMs over the whole prompt; cost scales with input length). Decode is **memory-bandwidth-bound** (one token at a time, repeatedly loading weights from HBM). Sharing an instance: a fat Prefill stutters dozens of Decode streams; Decode delays new Prefills. ITL spikes.

“P/D is only for multi-node fabrics” leaves single-node goodput on the table. Split e.g. **4+4** on the same box; hand KV over (can be gigabytes). MORI-IO is in-node **RDMA**. Scope of the post: **one box, 8 GPUs**.

## Architecture (Table 1)

| Component | Role |
| --- | --- |
| Prefill instance | Prompt → KV (example: GPUs 0–3) |
| Decode instance | Autoregressive tokens from transferred KV (GPUs 4–7) |
| Proxy | Client entry; orchestrates the two phases |

Mode is `VLLM_MORIIO_CONNECTOR_READ_MODE`. Before the first RDMA between a pair, ZMQ exchanges base addresses, block sizes, per-layer strides **in a background thread**; the RDMA session is cached.

### Read (`VLLM_MORIIO_CONNECTOR_READ_MODE=1`)

Proxy dispatches **serially**. Figure 1:

1. Client → proxy
2. Proxy → Prefill (`max_tokens=1`)
3. Prefill → proxy: `remote_block_ids`, `remote_engine_id`
4. Proxy → Decode with those IDs
5. Decode **pulls** KV (`WAITING_FOR_REMOTE_KVS`); scheduler skips until the RDMA read completes
6. Decode notifies Prefill to free blocks
7. Tokens stream SSE back

### Write (default: unset or `=0`)

Proxy fires **both** at once. Figure 2:

1. Client → proxy
2. Proxy → Prefill **and** Decode concurrently (each request carries the other’s connection details; proxy does **not** await Prefill)
3. Prefill **pushes** per layer (`save_kv_layer` RDMA write into Decode’s preallocated blocks). Chunked Prefill: accumulate until the last chunk, then write
4. Decode polls `pop_finished_write_req_ids` in `WAITING_FOR_REMOTE_KVS`
5. Ready queue → generation
6. SSE back

Toy proxy branch (from `examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py`): READ awaits Prefill and copies `remote_engine_id` / `remote_block_ids` into `kv_transfer_params`; WRITE falls through immediately and `asyncio.create_task`s Decode. Read **must** relay block IDs; write does not.

### Table 2

| Property | Read | Write |
| --- | --- | --- |
| `VLLM_MORIIO_CONNECTOR_READ_MODE` | `=1` | Unset / `=0` |
| RDMA | Decode pulls | Prefill pushes |
| Proxy | Serial | Concurrent |
| `remote_block_ids` via proxy | Required | Not required |
| KV cleanup | Decode notifies Prefill after pull | Prefill tracks write completion per request |

## Goodput, not raw throughput

DistServe: **goodput** = max request rate such that requests meet **TTFT &lt; T_ttft** and **ITL &lt; T_itl**. Here: TTFT **&lt; 1 s**, ITL **&lt; 50 ms**/token.

### Headline at 8 req/s / 100 requests (Table 3)

| Metric | Standard 1×TP8 | Standard 2×TP4 | MORI-IO Read 1P+1D | MORI-IO Write 1P+1D |
| --- | --- | --- | --- | --- |
| Both SLOs | 26/100 | 30/100 | 70/100 | **73/100** |
| Failure mode | ITL spikes (P99 ITL ≫ 50 ms) | ITL bimodal ~30 ms and ~150 ms | Some TTFT &gt; 1 s | Some TTFT &gt; 1 s |
| Relative | 0.9× | 1× | 2.4× | **2.5×** |

Collocated dies on ITL. Disagg **kills ITL violations**; leftovers are TTFT. Write beats Read by 3 requests because concurrent dispatch keeps more TTFT under 1 s.

### SLO vs rate 0.5–10 (Figure 4 / `03-SLO-attainment.png`)

- 1×TP8: ITL violations from low rates; **26/100** at rate 8
- 2×TP4: 100% at 0.5 → ~**60%** at 1 → ~**25%** by 2, then plateau
- Read: 100% to ~rate **5**, then to ~**44%** at 10 (TTFT)
- Write: 100% to ~rate **5.5**, then ~**46%** at 10

## Why ITL / TTFT move

Shared engine: one Prefill forward is much longer than a Decode step; every Decode in the batch waits → ITL inflates. Isolated Decode batches: ITL stable **in both modes**.

Standard TTFT:

```text
TTFT = queue + prefill_forward_pass + sample_T1 + detokenize + SSE_encode + network
```

Read adds two overheads (Figure 5 / sequence diagram):

```text
TTFT = queue(prefill) + prefill_forward_pass
     + [proxy serialize: await prefill, dispatch decode]   # Overhead 1
     + RDMA transfer (WAITING_FOR_REMOTE_KVS)              # Overhead 2
     + queue(decode) + sample_T1 + detokenize + SSE + net
```

Write (Figure 6):

```text
TTFT ≈ max(queue(prefill) + prefill_forward_pass + RDMA_write_time,
           queue(decode))
     + sample_T1 + detokenize + SSE + net
```

Write drops Overhead 1. RDMA write **overlaps** Prefill compute, so Overhead 2 does not add full wall-clock. Decode `RequestStatus.WAITING_FOR_REMOTE_KVS`: scheduler `_update_waiting_for_remote_kv`, else skip (`vllm/v1/request.py`, `vllm/v1/core/sched/scheduler.py`).

**Table 4 — when:**

| Situation | Recommendation |
| --- | --- |
| ITL p99 misses SLO under load | Disaggregate |
| TTFT is the binding UX constraint | Standard may win |
| High concurrency, long prompts | Disaggregate (worst Prefill interference) |
| Low rate, short prompts | Standard is enough |

## Setup sketch

Prefill is `kv_producer`; Decode `kv_consumer`. Instances ZMQ-register with the proxy (`proxy_ping_port`) and keep pinging. Proxy: `python examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py`. Round-robin. Docs: [disagg prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/).

Minimal extra config shape:

```bash
vllm serve <model> \
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

Decode: `kv_consumer`, `http_port` `40005`, `handshake_port` `7301`, `notify_port` `7501`.

**Table 5 — ports** (`MoRIIOConfig` / `moriio_common.py` applies per-rank offsets):

| Port | Purpose |
| --- | --- |
| `proxy_ping_port` | ZMQ registration |
| `http_port` | vLLM HTTP; proxy forwards here |
| `handshake_port` | One-time KV layout metadata |
| `notify_port` | Per-request: Prefill signals Decode that blocks are ready |

## Experimental details

Dockerfiles in vLLM: `Dockerfile.rocm_base` (MORI commit `2d02c6a9` in the build note; runtime library commit [`c365eaed`](https://github.com/ROCm/mori/commit/c365eaed02b13e6b8f2e9c8215b21516d86856ce)), `Dockerfile.rocm` (vLLM main). Hardware: **8× MI300X (gfx942)**; **2× EPYC 9654**. ROCm driver **6.10.5**; container `rocm/vllm-dev` (ROCm **7.0.51831-a3e329ad8**); vLLM **0.16.0rc1.dev1+gc46b0cd0a** (`c46b0cd0a`); PyTorch **2.9.1+git8907517**. Model as above; random dataset; **100** requests; rate **0.5–10** step **0.5**.

**Table 6:**

| Config | What |
| --- | --- |
| Standard 1×TP8 | One engine, all 8 GPUs, mixed Prefill+Decode, expert parallel |
| Standard 2×TP4 | Two mixed engines, RR proxy — fair GPU split vs 1P+1D |
| Read 1P+1D | GPU 0–3 Prefill / 4–7 Decode, TP=4+EP, `READ_MODE=1`, prefix cache **off** |
| Write 1P+1D | Same split, write mode; prefix cache **off as required by MORI-IO** |

MoE (this model) **amplifies** ITL jitter via expert routing; the Prefill/Decode fight also applies to dense models.

Nightly images: `rocm/vllm-dev`. Appendix commands (Qwen3-235B-A22B-FP8):

**Standard 2×TP4:**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_ROCM_USE_AITER=1 vllm serve Qwen/Qwen3-235B-A22B-FP8 \
  -tp 4 --enable-expert-parallel --max-model-len 16384 --max-num-batched-tokens 8192 \
  --distributed-executor-backend mp --no-enable-prefix-caching --port 8100
# second instance: CUDA_VISIBLE_DEVICES=4,5,6,7, --port 8200
# proxy: benchmarks/disagg_benchmarks/round_robin_proxy.py
```

**Disagg** (Read; unset `VLLM_MORIIO_CONNECTOR_READ_MODE` for Write). Also `HIP_VISIBLE_DEVICES`, `MORI_DISABLE_AUTO_XGMI=1`, `MORI_IO_ENABLE_NOTIFICATION=0`. Prefill `--port 20005 --max-num-batched-tokens 4096 --gpu_memory_utilization 0.9 --max_num_seqs 64` + producer kv-transfer-config above; Decode `--port 40005` + consumer config. Proxy: `moriio_toy_proxy_server.py`.

## Next (then)

Multi-node: same RDMA connector across hosts, no code change claimed. Per-phase knobs: Prefill can take larger token budgets / chunked Prefill; Decode smaller batches — impossible when collocated.

Disclaimer on the page: testing **2026-03-12**, MI300X, NPS1, 2.2 TiB DRAM, Ubuntu 22.04 / kernel 5.15.0-153-generic. Acknowledgements: AMD (Hongxia Yang, Gilbert Lei, Mingzhi Liu, Niko Ma, Tian Di, Randall Smith, Feiyue Zhai, Peng Sun, MORI team) and Embedded LLM (Pin Siang Tan, Jun Kang Chow, Ye Hur Cheong, Vensen Mu, Jeff Aw, Tun Jian Tan).

[router.md](router.md) is the cross-pod P/D gateway; this post is P/D **inside one box**. Same connector family as Mooncake / NIXL / CPU offload, different transport.
