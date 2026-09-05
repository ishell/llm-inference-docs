---
source: https://vllm.ai/blog/2025-01-27-v1-alpha-release
lang: en
fetched: 2026-09-05
---

# vLLM V1: A Major Upgrade to vLLM's Core Architecture

Chinese: [zh/vllm/blog/architecture/v1-alpha.md](../../../../zh/vllm/blog/architecture/v1-alpha.md)  
Source: https://vllm.ai/blog/2025-01-27-v1-alpha-release

2025-01-27. **vLLM Team.** Study extract, not an official reprint. **Alpha announcement.** Feature gaps listed here (no LoRA / spec decode / PP / Prometheus / logprobs / structured decoding; encoder-decoder; Mamba/Jamba; embeddings; non-Ampere NVIDIA) are **then-current**, not today’s matrix. V1 later became the default engine. Architecture map: [anatomy.md](anatomy.md). CPU-off-the-GPU-path work that led here: [v0.6-throughput.md](../../performance/v0.6-throughput.md).

Alpha of **vLLM V1**: after ~1.5 years of lessons, they revisited key design decisions, consolidated features, and simplified the codebase for flexibility and scale. They already claimed **SOTA**, with more optimizations to come. Enable: `VLLM_USE_V1=1`, **no API change**. After testing and feedback they planned to make V1 the default.

Do not treat the marketing logo as a figure. Local figures (copyright remains with the original site; study copies):

![v1 server architecture](../../../../assets/vllm/blog/architecture/v1-alpha/01-v1_server_architecture.png)
![v1 scheduling](../../../../assets/vllm/blog/architecture/v1-alpha/02-v1_scheduling.png)
![v1 prefix caching](../../../../assets/vllm/blog/architecture/v1-alpha/03-v1_prefix_caching.png)
![v1 tp architecture](../../../../assets/vllm/blog/architecture/v1-alpha/04-v1_tp_architecture.png)
![persistent batch](../../../../assets/vllm/blog/architecture/v1-alpha/05-persistent_batch.png)
![torch compile cuda graph](../../../../assets/vllm/blog/architecture/v1-alpha/06-torch_compile_cuda_graph.png)
![v1 llama](../../../../assets/vllm/blog/architecture/v1-alpha/07-v1_llama.png)
![v1 qwen2vl](../../../../assets/vllm/blog/architecture/v1-alpha/08-v1_qwen2vl.png)

## Why vLLM V1?

### Learning from vLLM V0

Over ~1.5 years V0 scaled **horizontally**: many models, features, hardware backends. Vertical integration lagged. Features grew independently, so combining them cleanly was hard. Technical debt piled up in the foundation, which is why they revisited the core.

### Goals of V1

- Simple, modular, easy-to-hack codebase.
- High performance with **near-zero CPU overhead**.
- **Combine** key optimizations in one architecture (not mutually exclusive plugins).
- **Zero configs**: turn the good defaults on.

### Scope of V1

Re-architected: scheduler, KV cache manager, worker, sampler, API server.

Still shared with V0: model implementations, GPU kernels, distributed control plane, utility functions. The bet: keep V0’s coverage and stability, replace the loop that was burning CPU.

## What’s New in vLLM V1?

### 1. Optimized Execution Loop & API Server

vLLM is a continuous-batching engine plus an OpenAI-compatible API server. Between GPU forwards, the CPU owns request state: run the API server, schedule, prepare inputs, detokenize, stream. GPUs got faster; model execution shrank; that CPU work became louder. On **Llama-8B / NVIDIA H100**, a GPU step can be ~**5 ms**. At that timescale, tokenization, scheduling, detokenization, and streaming are the bottleneck.

[v0.6.0](https://vllm.ai/blog/2024-09-05-perf-update) already split the API server into another process over **ZeroMQ**, overlapping the HTTP path with AsyncLLM. V1 pushes multiprocessing **inside** AsyncLLM: an isolated `EngineCore` loop that only runs the **scheduler + model executor**. Tokenization, multimodal preprocess, detokenization, and streaming overlap that core loop so throughput can rise.

### 2. Simple & Flexible Scheduler

No caste system between Prefill and Decode. User prompt tokens and model-generated tokens are the same currency. A step is a dictionary `{request_id: num_tokens}` — how many tokens to process for each request. That shape is general enough for chunked prefills, prefix caching, and speculative decoding. Chunked prefill is just: fixed token budget, dynamically split across requests (figure above).

### 3. Zero-Overhead Prefix Caching

Still **hash-based** prefix cache + **LRU** eviction, same idea as V0.

V0 problem: enabling prefix cache could **hurt** throughput when the hit rate was low, because of CPU overhead. So it was **default off**. V1: constant-time eviction data structure, fewer Python objects. Near-zero degradation even at 0% hit rate.

In their experiments: at **0% hit rate**, throughput drop is **<1%**. At high hit rate, several-fold wins. Because of that near-zero tax, prefix caching is **default on** in V1.

### 4. Clean Architecture for Tensor-Parallel Inference

V0 colocated the scheduler and **Worker 0** in one process to avoid broadcasting inputs. That saved IPC, but made the architecture **asymmetric** and more complex. V1 caches request state on the workers and sends only **diffs** each step. IPC shrinks enough that the scheduler and Worker 0 can live in **separate processes**. Workers then look the same for 1-GPU and multi-GPU. Most distributed logic is abstracted away from the worker.

### 5. Efficient Input Preparation

V0 rebuilt input tensors and metadata every step — another CPU bill. V1 uses **Persistent Batch** (technique they attribute to [LMDeploy](https://github.com/InternLM/lmdeploy)): cache the tensors, patch diffs each step. Updates go through **Numpy**, not Python-native loops.

### 6. torch.compile and Piecewise CUDA Graphs

V1 leans on vLLM’s `torch.compile` integration so many models get compiled without a custom kernel per architecture. **Piecewise CUDA graphs** cover the cases a single whole-graph capture cannot. They promised dedicated follow-up posts ([torch-compile.md](torch-compile.md) is the later write-up).

### 7. Enhanced Support for Multimodal LLMs

Multimodal LLMs as first-class. Three changes:

1. **Non-blocking preprocess + cache.** JPG/PNG → pixel tensors, crop, transform: that work can idle the GPU if it sits on the worker. V1 moves it to a separate process and caches processed inputs so the same image is not decoded twice.
2. **Prefix cache for images.** Token-ID hashes plus **image hashes** identify KV for image inputs. Multi-turn chats that keep sending the same picture reuse KV.
3. **Encoder cache** for chunked Prefill. In V0, image and text had to share a step: the decoder’s `<img>` token depends on vision embeddings, and those embeddings were discarded after the step. V1 stores vision embeddings in an encoder cache, so text Prefill can be chunked across steps without recomputing vision every step.

### 8. FlashAttention 3

Last piece: a flexible attention kernel for batches that **mix Prefill and Decode**. [FlashAttention 3](https://arxiv.org/abs/2407.08608) was the kernel they named for that mixed-batch shape: wide feature coverage, strong performance across use cases.

## Performance

Up to **~1.7×** throughput vs V0 **without multi-step scheduling**. Kernels for V0 and V1 were almost the same; the gap is CPU overhead across the stack. Gains are larger on VLMs (Qwen2-VL) because V1 treats vision as first-class.

### Text Models: Llama 3.1 8B & Llama 3.3 70B

ShareGPT. V1 lower latency than V0, especially at high QPS, from higher throughput. Kernels almost identical, so the gap is architecture (less CPU tax).

### Vision-language Models: Qwen2-VL

[VisionArena](https://arxiv.org/abs/2412.08687). Larger speedups than the text pair: preprocess off the worker + more flexible multimodal scheduling. Prefix caching for multimodal is native in V1; they **skipped** those benchmark plots here.

### Looking Forward

They framed the numbers as a starting point: the new architecture was supposed to make later features cheaper to land. More enhancements promised in the following weeks.

## Limitations & Future Work

Labeled **then-current (alpha, January 2025)**. Do not read as a 2026 support matrix.

### Model Support

Decoder-only Transformers (Llama-class), Mixtral-style MoE, several VLMs (Qwen2-VL). All quantization methods supported. **Not** then: encoder-decoder (example: multimodal Llama 3.2), Mamba-based (Jamba), embedding models. Pointer then: docs [supported models](https://docs.vllm.ai/en/latest/models/supported_models.html).

### Feature Limitations

Missing: logprobs, prompt logprobs sampling parameters, pipeline parallelism, structured decoding, speculative decoding, Prometheus metrics, LoRA. They listed people already implementing several of these (below), plus new optimizations for the V1 engine.

### Hardware Support

Ampere or later **NVIDIA** only. Other backends such as TPU were in progress.

Leave `VLLM_USE_V1` unset to stay on V0 (backward compatible).

## How to Get Started

1. `pip install vllm --upgrade`
2. `export VLLM_USE_V1=1`
3. [Python API](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic.py) (offline `basic.py`) or OpenAI-compatible server: `vllm serve <model-name>`. No API change.

They asked people to try it and send feedback.

## Acknowledgment

Design builds on [LightLLM](https://github.com/ModelTC/lightllm), [LMDeploy](https://github.com/InternLM/lmdeploy), [SGLang](https://github.com/sgl-project/sglang), [TGI](https://github.com/huggingface/text-generation-inference), and [TRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).

Incomplete contributor list from the post (roles as of the alpha):

- Effort driven mainly by UC Berkeley, Neural Magic (now Red Hat), Anyscale, and Roblox.
- [Woosuk Kwon](https://github.com/WoosukKwon) — initiated; scheduler and model runner.
- [Robert Shaw](https://github.com/robertgshaw2-redhat) — optimized execution loop and API server.
- [Cody Yu](https://github.com/comaniac) — prefix caching for text and image.
- [Roger Wang](https://github.com/ywang96) — overall MLLM support in V1.
- [Kaichao You](https://github.com/youkaichao) — `torch.compile` integration and piecewise CUDA graphs.
- [Tyler Michael Smith](https://github.com/tlrmchlsmth) — tensor parallelism via Python multiprocessing.
- [Rui Qiao](https://github.com/ruisearch42) — tensor parallelism via Ray; pipeline parallelism (in progress then).
- [Lucas Wilkinson](https://github.com/LucasWilkinson) — FlashAttention 3.
- [Alexander Matveev](https://github.com/alexm-redhat) — multimodal preprocessor; TPU (in progress then).
- [Sourashis Roy](https://github.com/sroy745) — logit penalties in the sampler.
- [Cyrus Leung](https://github.com/DarkLight1337) — MLLM input-processing refactor and V1 integration.
- [Russell Bryant](https://github.com/russellb) — multiprocess issues.
- [Nick Hill](https://github.com/njhill) — engine loop and API server.
- [Ricky Xu](https://github.com/rickyyx) and [Chen Zhang](https://github.com/heheda12345) — KV cache manager refactor.
- [Jie Li](https://github.com/jeejeelee) and [Michael Goin](https://github.com/mgoin) — MLLM support and optimization.
- [Aaron Pham](https://github.com/aarnphm) — structured decoding (in progress then).
- [Varun Sundar Rabindranath](https://github.com/varun-sundar-rabindranath) — multi-LoRA (in progress then).
- [Andrew Feldman](https://github.com/afeldman-nm) — logprobs / prompt logprobs (in progress then).
- [Lily Liu](https://github.com/LiuXiaoxuanPKU) — speculative decoding (in progress then).
- [Kuntai Du](https://github.com/KuntaiDu) — Prefill disaggregation and KV cache transfer (in progress then).
- [Simon Mo](https://github.com/simon-mo) and [Zhuohan Li](https://github.com/zhuohan123) — V1 system design.

Read this post for the disease V1 was written to cure: CPU stealing GPU time, features that would not compose, prefix cache too expensive to default on. Anatomy is the later map of the city.
