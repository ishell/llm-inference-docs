---
source: https://vllm.ai/blog/2026-07-14-vllm-tilert-pd
lang: en
fetched: 2026-09-04
---

# vLLM x TileRT: Specialized Decode for Latency-Critical Serving

Chinese: [zh/vllm/blog/serving/tilert.md](../../../../zh/vllm/blog/serving/tilert.md)

2026-07-14. **TileRT team**. TileRT **0.1.5** on [PyPI](https://pypi.org/project/tilert/) (`pip install tilert`; Python 3.12, CUDA 13 wheels) and [tile-ai/TileRT](https://github.com/tile-ai/TileRT). Join is vLLM V1’s public connector — `KVConnectorBase_V1` under `MultiConnector`, loaded with `kv_connector_module_path`. **Zero changes to vLLM**: no fork, no patches, no wrapped internal workers. Demos then: **GLM-5 / 5.1** and **DeepSeek-V3.2**. Prefill needs MTP; a TileRT decode node has **one** in-flight request.

Same door as other connector P/D, not a new protocol: [Mooncake](mooncake.md) / NIXL move bytes; [Router](router.md) is the stock vLLM P/D gateway (this post’s ingress is TileRT’s own `pd_router`); cluster-scale disagg in [large-scale.md](large-scale.md); another connector-only split in [moriio.md](moriio.md). Study note; figure bars are not an SLA.

Disaggregated serving splits compute-bound Prefill from memory-bandwidth-bound Decode. vLLM already does that through a first-class connector. Once the phases are separate, **Decode becomes pluggable**. Prefill pool, scheduler, cache, serving API stay; the decode pool is a choice.

This post is that choice: **vLLM Prefill + TileRT Decode**, shipping with TileRT 0.1.5. For latency-critical traffic you get TileRT’s native per-user decode speed; everything else stays stock vLLM.

## Why a second decode option?

vLLM’s native Decode remains the right **default**: high-throughput batched serving across a huge range of models and hardware. A growing class of workloads — agentic loops, interactive coding assistants, real-time voice — cares less about aggregate throughput than how fast tokens reach **each** user. Those jobs are latency-bound. Native Decode and TileRT sit at different points on the same throughput–latency frontier, which is why they compose.

TileRT is an inference runtime whose single goal is per-user decode speed toward the hardware limit. Their longer argument: [speed as a scaling dimension](https://www.tilert.ai/blog/speed-as-the-next-scaling-law.html). This post is not the engine write-up. The practical question: can you adopt a specialized decode engine **without giving up** OpenAI-compatible APIs, scheduling, prefix caching, tool calling, and vLLM’s operational maturity?

The integration is meant to keep that tax small:

- **Prefill is vLLM.** Scheduling, chunked prefill, prefix caching — untouched.
- **The serving surface is vLLM.** Same APIs, same request format, same tooling.
- **Only Decode changes, and only for traffic you send there.** The TileRT-paired stack runs beside an existing vLLM deployment; each workload picks its endpoint.

## Architecture: coexistence by design

Core rule: **zero changes to vLLM**. No fork, no patches, no wrapped internal workers. The plugin is a `KVConnectorBase_V1` implementation, composed under `MultiConnector`, loaded through stock `kv_connector_module_path`. Adding a TileRT decode pool must not destabilize a vLLM you already run; upgrading vLLM must not mean re-porting a fork.

Local figures (copyright remains with the original site; study copies):

![pd arch](../../../../assets/vllm/blog/serving/tilert/01-pd_arch.png)

**Figure (architecture).** Latency-critical traffic is marked by the TileRT PD router and claimed by the TileRT connector. General traffic uses the native disaggregation path. Both share **one stock vLLM Prefill pool** composed under `MultiConnector`.

**Routing.** A lightweight router fronts the TileRT pool. For each request it sets `max_tokens=1` (vLLM does Prefill and emits the first token) and attaches the target decode node in the pass-through field: `kv_transfer_params = {"tilert_host": ..., "tilert_ctrl_port": ...}`. Native-pool traffic still goes through the usual disaggregation proxy, unmodified.

**Claim filtering.** The TileRT connector claims **only** requests that carry the mark; it is a strict no-op for everything else. The two decode pools can share one Prefill instance — even a **single forward batch**. Adopting TileRT for some traffic changes nothing for the rest.

**A pure producer.** The connector is `kv_producer` only. It never touches scheduling or sampling; it extracts and ships state after Prefill. In every other respect the Prefill instance is a stock vLLM server.

## How the handoff works

Cross-engine disaggregation is practical only if three things hold: the transfer is fast, Prefill is not slowed, and Decode resumes exactly where Prefill stopped.

**Data plane.** After Prefill, the request’s attention state — compressed KV, sparse-attention index caches, and a little metadata — moves to the decode node as **RDMA one-sided writes** into pre-registered GPU buffers. Transfer engine is **Mooncake or NIXL**. No intermediate serialization, no staging through host memory. The handoff **protocol** does not care which engine moves the bytes.

**Fully overlapped with Prefill.** State extraction sits inside the forward window: the request’s state is copied to a staging buffer **before** its cache blocks can be recycled; a background sender does the network transfer. A TileRT-bound request never blocks the next Prefill iteration, including native-pool requests sharing the same batch.

**Injection into a live engine.** On arrival, state is converted to TileRT’s native layout and injected into a **running** engine. Decoding starts immediately, with multi-token speculative decoding (MTP) active from the first step.

## Evaluation

![glm5 tilert mtp](../../../../assets/vllm/blog/serving/tilert/02-glm5_tilert_mtp.png)

**Figure (decode speed).** GLM-5.1-FP8 token generation on **8× NVIDIA B200**, TileRT **v0.1.5**. Output length **1K**, input length **1K–192K**. Three bars: TileRT **without MTP**; with MTP at average acceptance length **3.2**; peak under best-case MTP acceptance **4.0**.

The post does not print tok/s / TPS table cells — those heights live only in the figure.

## Choosing your decode pool

Route to **TileRT Decode** when per-user token speed is the binding constraint — interactive agents, real-time assistants, latency-SLO inference — **and** the model is one TileRT supports.

Stay on **native vLLM Decode** for maximum aggregate throughput, high-concurrency batching, and the long tail of models and features general-purpose Decode covers.

Both stacks expose the same OpenAI-compatible surface. Moving a workload is a **routing** change, not a client change.

**Current limitations (this release):** a TileRT decode node serves **one in-flight request at a time**; the router provides gated dispatch and back-pressure. Model coverage: **GLM-5/5.1** and **DeepSeek-V3.2**, with more to come.

## Getting started

Install TileRT on **both** Prefill and Decode nodes; Prefill needs the connector plugin.

```bash
# 0. One-time: convert the HF checkpoint to TileRT's weight format
python -m tilert.models.preprocess.weight_converter \
    --model_type glm-5 \
    --model_dir /path/to/GLM-5.1 \
    --save_dir /path/to/tilert-glm5.1-weights

# 1. TileRT decode node
python -m tilert.pd_vllm.decode_server \
    --engine tilert --model glm5 \
    --model-weights-dir /path/to/tilert-glm5.1-weights \
    --with-mtp --max-seq-len 202752 \
    --kv-cache-dtype fp8 \
    --ctrl-port 5556 --http-port 5557

# 2. vLLM prefill (stock vLLM; the connector loads as a plugin).
#    The MTP speculative config is required: prefill populates the
#    draft-layer KV that decode-side speculation resumes from.
vllm serve /path/to/GLM-5.1 \
    --served-model-name glm5.1 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --enforce-eager \
    --trust-remote-code \
    --return-tokens-as-token-ids \
    --gpu-memory-utilization 0.8 \
    --kv-cache-dtype fp8_ds_mla \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
    --kv-transfer-config '{
        "kv_connector": "TileRTConnector",
        "kv_connector_module_path": "tilert.pd_vllm.prefill_connector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config":{
            "tilert_host":"[TILERT_DECODE_SERVER_IP]",
            "tilert_ctrl_port":5556,
            "tilert_model":"glm5",
            "tilert_max_seq_len":202752
        }
    }'

# 3. Router: OpenAI-compatible ingress for the TileRT pool
python -m tilert.pd_vllm.pd_router \
    --vllm-url http://prefill-node:8000 \
    --decode decode-node:5556:5557 \
    --model-path /path/to/GLM-5.1 \
    --port 23333
```

Prefill-side `--speculative-config` with `"method": "mtp"` is **required**, not optional: Prefill fills draft-layer KV that Decode-side speculation resumes from. Decode CLI uses `--with-mtp`. Sequence cap in the example: **202752**. Decode KV dtype `fp8`; vLLM Prefill `--kv-cache-dtype fp8_ds_mla`. Router `--decode` is `host:ctrl-port:http-port`. Ingress port **23333**.

To run the TileRT pool **and** a native vLLM decode pool behind **one** shared Prefill instance, compose both connectors under `MultiConnector`. The configuration they validated runs **NIXL end to end** (vLLM’s standard `NixlConnector` for the native pool, the TileRT connector in NIXL mode for the TileRT pool), so shared Prefill uses a **single** transfer library. Only Prefill’s `--kv-transfer-config` changes.

## Looking ahead

Disaggregation is quietly changing what an inference stack is: less one engine, more a **composition of specialized engines** behind a shared serving layer. vLLM’s connector interface is what makes that composition possible today; this pairing is one example. It is also why an engine like TileRT can specialize this deeply: with the serving layer shared and the interfaces open, going deep on one dimension no longer means rebuilding everything else.

They asked for community feedback on the integration surface, the workloads where this helps, and which models to support next.

## Acknowledgements

The vLLM community for the V1 connector interface that made a **zero-modification** integration possible. The Mooncake and NIXL projects for the RDMA transfer engines. [Inferact Inc.](https://inferact.ai/) for collaboration on the vLLM–TileRT integration.
