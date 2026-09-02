---
source: https://vllm.ai/blog/2025-01-27-v1-alpha-release
lang: en
fetched: 2026-08-31
---

# vLLM V1: A Major Upgrade (alpha announcement)

2025-01-27. Historical alpha post. Feature gaps listed here (no LoRA / spec decode / PP, …) are **not** today’s matrix. V1 later became default. Architecture map: `anatomy.md`.

Enable then: `export VLLM_USE_V1=1`. No API change.


Local figures (copyright remains with the original site; study copies):

![v1 server architecture](../../../../assets/vllm/blog/architecture/v1-alpha/01-v1_server_architecture.png)

![v1 scheduling](../../../../assets/vllm/blog/architecture/v1-alpha/02-v1_scheduling.png)

![v1 prefix caching](../../../../assets/vllm/blog/architecture/v1-alpha/03-v1_prefix_caching.png)

![v1 tp architecture](../../../../assets/vllm/blog/architecture/v1-alpha/04-v1_tp_architecture.png)

![persistent batch](../../../../assets/vllm/blog/architecture/v1-alpha/05-persistent_batch.png)

![torch compile cuda graph](../../../../assets/vllm/blog/architecture/v1-alpha/06-torch_compile_cuda_graph.png)

![v1 llama](../../../../assets/vllm/blog/architecture/v1-alpha/07-v1_llama.png)

![v1 qwen2vl](../../../../assets/vllm/blog/architecture/v1-alpha/08-v1_qwen2vl.png)

## Why

V0 scaled horizontally (models, features, hardware) but stacked poorly vertically. Debt in the foundation. V1 goals: modular code, near-zero CPU overhead, one architecture that *combines* optimizations, features on by default.

Rebuild: scheduler, KV manager, worker, sampler, API server. Still shares V0 model impls, kernels, distributed control plane. Influences named: LightLLM, LMDeploy, SGLang, TGI, TRT-LLM.

## What was new

1. **Execution loop.** GPU steps ~5 ms on Llama-8B/H100 → CPU (API, schedule, detok, stream) shows. v0.6 split API server via ZMQ. V1 isolates `EngineCore` (scheduler + executor); tokenize / MM preprocess / detok / stream overlap the core loop.
2. **Scheduler.** No prefill/decode caste. Step = `{request_id: num_tokens}`. Chunked prefill, prefix cache, spec decode all fit. Fixed token budget, dynamic split.
3. **Prefix cache ~zero overhead.** Hash + LRU. V0 could *lose* throughput at 0% hit rate → default off. V1: <1% hit at 0% hits; large wins when hot → **default on**.
4. **TP.** V0 colocated scheduler + worker 0 (asymmetric). V1 caches request state on workers, sends diffs; scheduler and worker 0 in separate processes.
5. **Persistent batch.** Patch cached tensors; Numpy over Python.
6. **torch.compile + piecewise CUDA graphs.**
7. **MLLMs first-class.** Non-blocking preprocess + cache; image hashes in prefix cache; **encoder cache** so text prefill can chunk without recomputing vision embeddings.
8. **FlashAttention 3** for mixed prefill+decode batches.

## Then-current numbers

Up to **~1.7×** throughput vs V0 without multi-step scheduling. Same kernels, less CPU. Larger jump on Qwen2-VL / VisionArena.

## Alpha limits (Jan 2025)

Decoder-only Transformers, Mixtral-style MoE, some VLMs; quant OK. No encoder-decoder, Mamba/Jamba, embeddings. Missing logprobs, PP, structured/spec decode, Prometheus, LoRA. NVIDIA Ampere+ only.
