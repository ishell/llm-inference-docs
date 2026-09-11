---
source: https://docs.vllm.ai/en/stable/usage/v1_guide/
lang: en
fetched: 2026-09-04
---

# vLLM V1 Guide

Chinese: [zh/vllm/features/v1-guide.md](../../../zh/vllm/features/v1-guide.md)  
Release story: [v1-alpha](../blog/architecture/v1-alpha.md). Anatomy: [anatomy](../blog/architecture/anatomy.md). Tuning order: [optimization.md](../optimization/optimization.md). RFC: [#18571](https://github.com/vllm-project/vllm/issues/18571) — V0 is **fully deprecated**. A V0-only use case that died on V1 belongs on GitHub or Slack.

V1 keeps V0’s proven pieces (models, GPU kernels, utilities) and re-architects scheduler, KV cache manager, worker, sampler, and API server. Aims: simple/modular/hackable; near-zero CPU overhead; optimizations in one architecture; **zero configs** by turning features on by default. Long-context numbers are promised on a “performance benchmark (To be added)” line. Blog: [vLLM V1: A Major Upgrade…](https://vllm.ai/blog/2025-01-27-v1-alpha-release) (2025-01-27). This page is a **living** list of behavior changes and support; it will keep moving.

Unified scheduler: `{request_id: num_tokens}` token budget per request. Chunked prefill, prefix cache, speculative decoding share that ledger — no strict Prefill/Decode side doors. Policies: FCFS or priority (`--scheduling-policy`; FCFS breaks ties).

## Differences from V0

- **Chunked prefill** on by default whenever possible (V0 was model-conditional). Decode first; leftover `max_num_batched_tokens` to prefill; overflow is chunked.
- **CUDA Graphs** capture uses **more memory** than V0.
- **Logprobs:** default is values **right after** the model’s raw output, **before** temperature / penalties. `--logprobs-mode`: `raw_logprobs` (default), `processed_logprobs`, `raw_logits`, `processed_logits`. Raw = before any logit processors (incl. bad words). Processed = after all processors including temperature and top_k/top_p.
- **Prompt logprobs + prefix caching:** supported, but logprobs are **not** cached. That request **ignores** the prefix cache and recomputes the full prompt prefill.

Default preemption is `RECOMPUTE`, not `SWAP`. Frequent preemption: more KV (`gpu_memory_utilization` / TP) — see optimization.

## Feature support legend

- 🟢 Functional — comparable to or better than V0
- 🟡 In Progress — planned; open PR/RFC
- 🔴 Removed — only comes back with strong demand

### Hardware

| Hardware | Status |
|---|---|
| NVIDIA | 🟢 |
| AMD | 🟢 |
| INTEL GPU | 🟢 |
| TPU | 🟢 |
| CPU | 🟢 |

More platforms via plugins: [vllm-ascend](https://github.com/vllm-project/vllm-ascend), [vllm-spyre](https://github.com/vllm-project/vllm-spyre), [vllm-gaudi](https://github.com/vllm-project/vllm-gaudi), [vllm-openvino](https://github.com/vllm-project/vllm-openvino).

### Models

| Model type | Status |
|---|---|
| Decoder-only | 🟢 |
| Encoder-Decoder | 🟢 Whisper; 🔴 others (natively) |
| Pooling | 🟢 |
| Mamba | 🟢 |
| Multimodal | 🟢 |

**Pooling:** fully supported; prefix caching and chunked prefill newly available for **last-pooling** models; more pooling categories still in progress.

**Mamba:** Mamba-2 / Mamba-1 (`Mamba2ForCausalLM`, `MambaForCausalLM`, `FalconMambaForCausalLM`) and hybrids (`Zamba2ForCausalLM`, `NemotronHForCausalLM`, `FalconH1ForCausalLM`, `GraniteMoeHybridForCausalLM`, `JambaForCausalLM`). Other hybrids too (`Lfm2ForCausalLM`). **Prefix caching is not yet supported** for any of the above.

**Encoder-decoder:** Whisper native. Others via plugin: **BART** / **Florence-2** through [bart-plugin](https://github.com/vllm-project/bart-plugin). For the rest (e.g. `MllamaForConditionalGeneration`), same pattern via the [plugin system](https://docs.vllm.ai/en/stable/design/plugin_system/). In-tree cousin: [hardware-plugin](../blog/architecture/hardware-plugin.md) / [plugin-system](../blog/architecture/plugin-system.md).

### Features

| Feature | Status |
|---|---|
| Prefix Caching | 🟢 |
| Chunked Prefill | 🟢 |
| LoRA | 🟢 |
| Logprobs Calculation | 🟢 |
| FP8 KV Cache | 🟢 |
| Spec Decode | 🟢 |
| Prompt Logprobs with Prefix Caching | 🟢 (recompute full prompt; see above) |
| Structured Output Alternative Backends | 🟢 |
| Concurrent Partial Prefills | 🟡 [#14003](https://github.com/vllm-project/vllm/issues/14003) |
| `best_of` | 🔴 [#13361](https://github.com/vllm-project/vllm/issues/13361) |
| Per-Request Logits Processors | 🔴 [#13360](https://github.com/vllm-project/vllm/pull/13360) |
| GPU <> CPU KV Cache Swapping | 🔴 |
| Request-level Structured Output Backend | 🔴 |

### Removed

- **Sampling:** `best_of` (limited usage). Per-request logits processors → **global** processors at startup ([RFC #17799](https://github.com/vllm-project/vllm/issues/17799)).
- **KV:** V1 does not need GPU↔CPU swap to handle preemption. Park KV on CPU later via [Offloading Connector](../blog/serving/kv-offload.md), not V0 sync swap.
- **Structured output:** request-level backend gone; alternative backends (outlines, guidance) with fallbacks are supported.

Multiprocess house: API server, engine core, one worker per GPU. Too few CPU cores and the GPU waits on the waiter — optimization puts this first.
