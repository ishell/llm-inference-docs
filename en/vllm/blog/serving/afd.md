---
source: https://vllm.ai/blog/2026-07-23-vllm-afd-plugin
lang: en
fetched: 2026-09-05
---

# Announcing vLLM AFD Plugin: Disaggregating Attention and FFN for Flexible MoE Serving

Chinese: [zh/vllm/blog/serving/afd.md](../../../../zh/vllm/blog/serving/afd.md)  
Source: https://vllm.ai/blog/2026-07-23-vllm-afd-plugin

2026-07-23. **AFD Plugin Contributors**. Experimental external plugin: https://github.com/vllm-project/afd-plugin. Hooks `vllm.general_plugins` and `--additional-config`; **no vLLM source edits**. Then pinned **vLLM 0.19.1**, Python **3.10–3.13**, model runner **v1 only**. Full weights on **both** roles. Study rewrite; not an SLA. The page itself says it needs more large-scale testing across backends.

EPD splits the ViT; Router splits text Prefill/Decode; AFD splits Attention vs experts **inside the layer**. Plugin door: [plugin-system](../architecture/plugin-system.md). Hardware plugins: [hardware-plugin](../architecture/hardware-plugin.md). Runner they had not evaluated yet: [mrv2](../architecture/mrv2.md).

**TL;DR from the page:**

- Attention–FFN Disaggregation (AFD): Attention and FFN as independent services; request lifecycle and the OpenAI endpoint stay with vLLM.
- Backends: NVIDIA GPU and Ascend NPU. Connectors: `P2pNcclAFDConnector`, `CAMP2pAFDConnector`, `CAMAsyncAFDConnector`.
- Sync Decode uses `FULL_DECODE_ONLY` graphs; async Prefill had **no graph** yet.
- Wrappers: DeepSeek V2/V3-family (including V3.2), GLM MoE DSA. DBO is **exactly two** ubatches.
- DeepSeek-V3.2 W8A8 on 910C: 64A16F vs EP64 is **+11.3%** (16K) and **+9.0%** (32K) tokens/s/die. Async Prefill 10-layer experiment: median TTFT at 12 rps **15.1 s → 8.0 s**.

Original sections: Why Attention-FFN Disaggregation? → Inside the Architecture (Connector and backend support / Supported features) → A Performance Snapshot (Synchronous AFD Decode Throughput with `CAMP2pAFDConnector`: 16K / 32K; Asynchronous AFD Prefill Performance with `CAMAsyncAFDConnector`) → Getting Started (Install / Deployment Recipes) → Current Scope and Roadmap → Join the Community.

[vLLM AFD Plugin](https://github.com/vllm-project/afd-plugin) brings Attention-FFN Disaggregation to MoE models by separating Attention and FFN into independently deployed services. The request lifecycle and OpenAI-compatible serving interface stay; the two paths can scale independently.

Then-current support: NVIDIA GPUs and Ascend NPUs, sync and async connectors, DeepSeek V2/V3-family wrappers, and eager / graph / dual-batch paths inside validated limits.

> Still experimental; needs more large-scale testing across hardware backends.

## Why Attention-FFN Disaggregation?

MoE inference mixes two different kinds of work in every transformer layer. Attention is **stateful** and coupled to scheduling and the KV cache. The FFN / expert path is routed expert compute plus all-to-all. When both share one worker topology, the serving system must pick **one** scaling and execution choice for two very different jobs.

Making the split practical means answering several design problems:

1. **Different scaling.** Attention capacity follows request state, sequence length, and KV-cache pressure. Expert capacity follows token routing and expert load. Rank topologies should be allowed to differ instead of sharing one layout.
2. **Different runtime jobs.** Attention needs scheduling, KV coordination, and sampling. FFN only needs activations, routing metadata, and a way to return expert outputs. After the split, FFN can be a lightweight connector-driven **daemon**.
3. **Backend-specific communication.** CUDA and Ascend expose different collectives, graph runtimes, and optimized MoE ops. A common connector contract keeps the model-facing flow stable while each backend owns its data path.
4. **Overlap.** Asynchronous dispatch and MoE ubatching can overlap independent stages instead of serializing all expert work behind Attention.

Together: keep vLLM’s request-facing Attention path intact, and move FFN execution behind a narrow connector interface that can scale, communicate, and execute on its own.

## Inside the Architecture

![vllm afd plugin architecture](../../../../assets/vllm/blog/serving/afd/01-vllm-afd-plugin-architecture.svg)

**Figure.** vLLM AFD Plugin runtime (study copy; copyright remains with the original site).

The plugin integrates through `vllm.general_plugins` and `--additional-config`. It does **not** require edits to the vLLM source tree.

Three runtime parts:

- **Attention service.** The Attention worker keeps vLLM’s scheduler, KV cache, batching, model lifecycle, and sampling. A plugin-owned model runner installs AFD metadata in the forward context and publishes data-parallel, ubatch, layer, and graph state to FFN.
- **FFN service.** No request traffic, scheduler, or KV cache. A background loop receives metadata and activations, calls `compute_ffn_output()` on the plugin wrapper, and sends the result back. Requests always hit the Attention API server.
- **Connector layer.** At each split layer, the connector transfers Attention hidden states plus the execution metadata FFN needs, then returns FFN outputs. A backend-neutral interface defines the exchange; each backend implements its own communication and runtime opts.

The integration surface is intentionally small. vLLM keeps the serving control plane. The plugin owns AFD workers, model runners, connectors, metadata, split points, and a small set of version-scoped compatibility patches.

### Connector and backend support

| Connector | Backend | Execution | Recommended stage | Graph support |
| --- | --- | --- | --- | --- |
| `P2pNcclAFDConnector` | GPU | Synchronous P2P | Decode | `FULL_DECODE_ONLY` CUDA graph |
| `CAMP2pAFDConnector` | NPU | Synchronous CAMP2P/HCCL | Decode | `FULL_DECODE_ONLY` ACL graph |
| `CAMAsyncAFDConnector` | NPU | Asynchronous CAM | Prefill | **Not then supported** |

The same high-level exchange on every connector: Attention output to FFN, FFN output back. Backend packages stay separate so CUDA graph, ACL graph, NCCL, and Ascend custom ops do not leak into one another.

### Supported features

- **Native vLLM serving surface.** `vllm serve`, OpenAI-compatible endpoint, `--additional-config`.
- **GPU and NPU implementations.** GPU workers extend vLLM v1 classes; NPU workers extend **vLLM-Ascend** classes directly. Shared behavior lives in config, topology, metadata, and connector contracts — not cross-device inheritance.
- **Synchronous AFD for Decode throughput.** `P2pNcclAFDConnector` and `CAMP2pAFDConnector` synchronously exchange Attention activations and FFN outputs so the two roles can scale independently in throughput-oriented Decode. Graph paths then used `FULL_DECODE_ONLY` on CUDA and ACL.
- **Asynchronous AFD for Prefill.** `CAMAsyncAFDConnector` uses CAM async dispatch/combine to decouple Prefill Attention ranks from expert workers. With AFD-managed MoE ubatching, independent Attention and FFN stages overlap. This path then targeted **P/D-disaggregated Prefill** and did **not** support graph execution.
- **MoE model integration.** Wrappers for DeepSeek V2/V3-family (including DeepSeek V3.2) and GLM MoE DSA. Separate Attention vs FFN compute; layer implementations reuse upstream.
- **Graph and ubatching.** Sync GPU/NPU connectors support Decode-only graph capture. Dual Batch Overlap with **exactly two** ubatches; CAM async has its own AFD-managed MoE ubatching on Prefill.

## A Performance Snapshot

### Synchronous AFD Decode Throughput with `CAMP2pAFDConnector`

Sync Decode recipe: [vllm-project/afd-plugin#67](https://github.com/vllm-project/afd-plugin/pull/67). Conventional EP64 vs `CAMP2pAFDConnector` AFD. DeepSeek-V3.2 **W8A8** on Ascend **910C**. Saturated Decode throughput, not online-serving latency.

| Deployment | Physical topology | Total dies |
| --- | --- | ---: |
| EP64 | DP64, EP64, TP1 | 64 |
| 48A16F | 48 Attention ranks, 16 FFN ranks | 64 |
| 64A16F | 64 Attention ranks, 16 FFN ranks | 80 |

> Controlled performance, not accuracy or production serving. Limited machines: physical 48A16F / 64A16F **simulate** logical **192A64F / 256A64F**. Routed expert IDs replaced by a **deterministic forced-balancing cycle** — **outputs change**. `AFDDecodeBenchConnector` supplies decode-only KV; **DBO on** for AFD.

Normalize by total deployed dies:

```text
tokens/s/die = aggregate output token throughput / total deployed dies
```

Fixed-length inputs; outputs uniform **512–1,536** tokens.

#### 16K fixed input

![throughput dsv3 2 16k](../../../../assets/vllm/blog/serving/afd/02-throughput_dsv3-2_16k.png)

**Figure.** DeepSeek-V3.2 16K Decode throughput per die.

EP64 **232.6** tokens/s/die; 48A16F **220.3**; 64A16F **258.9**. vs EP64: 48A16F **−5.3%**, 64A16F **+11.3%**.

#### 32K fixed input

![throughput dsv3 2 32k](../../../../assets/vllm/blog/serving/afd/03-throughput_dsv3-2_32k.png)

**Figure.** DeepSeek-V3.2 32K Decode throughput per die.

EP64 **168.2** tokens/s/die; 48A16F **151.4**; 64A16F **183.3**. vs EP64: 48A16F **−10.0%**, 64A16F **+9.0%**.

Across both lengths, 48A16F sits below EP64; 64A16F has the highest normalized throughput (**+11.3%** at 16K, **+9.0%** at 32K). The Attention-to-FFN **ratio** is the sentence; disaggregation alone does not guarantee a gain.

They did not test higher Attention:FFN ratios. The trend at the ratios they ran: FFN ranks still had **compute headroom**. Raising the Attention share might still help.

### Asynchronous AFD Prefill Performance with `CAMAsyncAFDConnector`

Early CAM async experiment: **two** Ascend 910C nodes, DeepSeek V3.2 W8A8 **cut to 10 layers**, forced expert balancing. Baseline `DP4PCP8 TP1` vs Attention `DP3PCP8 TP1` + FFN `EP8`.

![text matched dp afd median ttft](../../../../assets/vllm/blog/serving/afd/04-text_matched_dp_afd_median_ttft.png)

**Figure.** Median TTFT for the CAM async experiment.

AFD lowers median/P50 TTFT across measured rates. At **12 rps**: **15.1 s → 8.0 s** (~**47%**). At **10 and 12 rps**, the gap is about **7.2 s**.

Path check of CAM async, **not** a full-model or every-topology claim. Gains vary by workload.

## Getting Started

Then: Python **3.10–3.13**, vLLM **`0.19.1`**.

### Install

Installation steps live in the plugin [README](https://github.com/vllm-project/afd-plugin#install). The blog does not duplicate them.

### Deployment Recipes

Launch commands depend on backend, connector, model, and rank topology. Use the in-tree [AFD Plugin recipes](https://github.com/vllm-project/afd-plugin/tree/main/recipe) instead of copying configs here:

- **GPU synchronous AFD:** [DeepSeek V2 Lite P2P NCCL recipes](https://github.com/vllm-project/afd-plugin/tree/main/recipe/gpu/p2p_nccl/deepseek_v2_lite) — Decode-oriented colocated and Prefill/Decode-disaggregated deployments, eager and CUDA graph, several DP/TP layouts.
- **NPU asynchronous Prefill AFD:** [DeepSeek V3.2 CAM async recipe](https://github.com/vllm-project/afd-plugin/blob/main/recipe/npu/cam_async/DeepSeek-V3.2.md) — environment, topology, AFD config, bench, then-current limits.

Latest connector matrix, config fields, and full launch commands: repository README and recipe directory.

## Current Scope and Roadmap

Boundaries they listed: exact vLLM pin, model runner v1 only, **full weights on both roles**, Decode-only graph modes, **exactly two** DBO ubatches, hardware-gated e2e tests.

Next phase as written:

- **Broader vLLM compatibility and upstream alignment:** newer vLLM, evaluate **model runner v2**, keep patches small, contribute abstractions upstream as they mature.
- **More flexible execution:** graph modes, ubatch counts, asynchronous stages, validated rank topologies.
- **Production-scale validation:** repeatable accuracy, latency, throughput, stability, and multi-node results on full models and realistic workloads.
- **Expanded model and connector coverage:** more MoE architectures and backend transports through the existing wrapper and connector interfaces, with recipes for each.
- **Multimodal and vLLM-Omni:** how AFD can meet [vLLM-Omni](https://github.com/vllm-project/vllm-omni) and heterogeneous multimodal pipelines — autoregressive (AR), Diffusion Transformer (**DiT**), and other stages that want independently scaled Attention vs FFN.
- **Heterogeneous hardware and low-latency serving:** Attention and FFN on different accelerator types and interconnects; connector, scheduling, placement, and compute–communication overlap for TTFT and ITL.

## Join the Community

Early-stage; feedback from model, serving, and hardware communities will shape it.

- **Code and documentation:** [github.com/vllm-project/afd-plugin](https://github.com/vllm-project/afd-plugin)
- **Runtime design docs:** [GPU Attention/FFN and Ascend Attention/FFN](https://github.com/vllm-project/afd-plugin/tree/main/docs)
- **Issues:** [GitHub Issues](https://github.com/vllm-project/afd-plugin/issues)
