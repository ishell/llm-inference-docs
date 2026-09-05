---
source: https://vllm.ai/blog/2025-10-27-semantic-router-modular
lang: en
fetched: 2026-09-04
---

# Modular LoRA: stop running a full model per classifier

Chinese: [zh/vllm/blog/serving/semantic-router-modular.md](../../../../zh/vllm/blog/serving/semantic-router-modular.md)

2025-10-27. **Ivar Flakstad (Hugging Face), OneZero-Y, Huamin Chen (Red Hat), Xunzhuo Liu (Tencent)**. Repo: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router). Launch: [semantic-router.md](semantic-router.md). Shared-base LoRA lands in [Iris](semantic-router-iris.md). Signal spine: [semantic-router-signal.md](semantic-router-signal.md). Later: [athena](semantic-router-athena.md), [amd](semantic-router-amd.md), [mom-amd](semantic-router-mom-amd.md), [vision](semantic-router-vision.md), [themis](semantic-router-themis.md). Do not confuse with the in-engine [router.md](router.md). Flash Attention 2 speedups (ModernBERT ~**3×**, Qwen3 ~**4×**, 14B 70–110 vs 30–35 tok/s) are **citations**, not a vLLM-SR cluster run. LoRA “**<1%** params” and the 10-thread / 30-classification test are theirs.

Siblings: [session](semantic-router-session.md), [fusion](semantic-router-fusion.md), [micro-agent](semantic-router-micro-agent.md), [mom](semantic-router-mom.md).

Semantic routing hits a scaling wall when each classification request runs several fine-tuned models independently: cost grows **linearly** with the number of models. This post is the Rust classification-layer refactor: architectural modularity, LoRA, concurrency.

Local figures (copyright remains with the original site; study copies):

![modular](../../../../assets/vllm/blog/serving/semantic-router-modular/01-modular.png)

**Figure 1.** Layered candle-binding: core independent of any one architecture.

## Background: from BERT to a modular system

Previous implementation leaned on BERT and ModernBERT for intent and jailbreak. ModernBERT is strong for English classification. Limits they name:

- **Language coverage**: original ModernBERT’s multilingual support is thinner than models trained on more diverse data. Note on the page: [mmBERT](https://huggingface.co/blog/mmbert) (1800+ languages) shipped **after** this refactoring started — an alternative multilingual path, not what this patch trains.
- **Context length**: ModernBERT to **8,192** tokens with RoPE ([Transformers docs](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/modernbert)). Qwen3-Embedding they cite at **32,768**.
- **Model coupling**: classification logic tied to specific architectures; hard to add new models.

The modular architecture is how newer models (mmBERT, Qwen3-Embedding, EmbeddingGemma) can sit side by side; the router picks per task.

## Architectural restructuring

Layered architecture in the **candle-binding** crate. Core stays independent of a specific model; new architectures plug in without rewriting existing code. `DualPathUnifiedClassifier` chooses between traditional fine-tuned models and LoRA-adapted models from the task.

## Long-context embedding models

### Qwen3-Embedding

Context up to **32,768** tokens ([Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)). RoPE for longer-distance frequency resolution. Trained on text from **100+** languages (same model card) — the multilingual gap ModernBERT-only routing hit.

### EmbeddingGemma-300M

Google’s smaller embedding model. Context **2,048** tokens. **Matryoshka** representation: embeddings truncatable to **768 / 512 / 256 / 128** dimensions without retraining ([google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)).

**MQA**: 3 query heads, 1 key-value head — less memory bandwidth. Distinctive dense bottleneck after the transformer blocks: **768 → 3072 → 768**, tied to the Matryoshka training story.

## LoRA for multi-task classification

Naive path: intent + PII + jailbreak = **three full BERT forwards**.

![full params](../../../../assets/vllm/blog/serving/semantic-router-modular/02-full-params.png)

**Figure 2.** Three independent fine-tunes: O(n) full forwards for n tasks.

Each model pays the expensive base transformer. Complexity **O(n)** in the number of classification tasks.

LoRA shares the base pass:

![lora](../../../../assets/vllm/blog/serving/semantic-router-modular/03-lora.png)

**Figure 3.** One base forward, then cheap adapters. LoRA typically **<1%** of parameters.

Base model once → intermediate representations. Each LoRA adapter applies task-specific low-rank updates. Adapters typically modify **<1%** of parameters; that last step is much cheaper than a full model.

`parallel_engine.rs` uses [Rayon](https://github.com/rayon-rs/rayon) for data parallelism across adapters. Three classifications: one full pass + three lightweight adapter applications, not three full forwards.

**LoRA wins on multi-task, not single-task.** Single-task: no base sharing; a traditional fine-tune may be faster. Speedup depends on base-compute vs adapter-compute ratio.

## Concurrency through `OnceLock`

Previous global classifier state: `lazy_static` — lock contention under concurrent load. Refactor: [`OnceLock`](https://doc.rust-lang.org/std/sync/struct.OnceLock.html) from std.

After first init, reads are lock-free pointer reads. Test file they name: `oncelock_concurrent_test.rs` — **10** concurrent threads, **30** total classifications; they report throughput scales linearly with thread count. With `lazy_static`, concurrent requests queued on a mutex. With `OnceLock`, they run without that contention.

### Optional Flash Attention 2

Optional Cargo feature for CUDA builds. Requires **Ampere+** (compute capability **≥ 8.0**). Blocked attention in on-chip SRAM; fewer DRAM round-trips.

Cited (not a vLLM-SR cluster measurement):

- **ModernBERT**: up to **3×** faster self-attention, less memory ([source they link](https://medium.com/@alpernebikanli/some-berts-and-modernbert-39b261b1ce83)). Alternating attention: global every third layer, local sliding-window otherwise ([Answer.AI](https://www.answer.ai/posts/2024-12-19-modernbert.html)).
- **Qwen3**: FlashAttention-2 up to **4×** on attention. 14B variant: **70–110** tok/s vs **30–35** without it, more pronounced at long context ([source they link](https://qwen3lm.com/qwen3-flashattention2-inference-guide/)).

Rust keeps Flash Attention optional so hosts without compatible GPUs still run; gains only when the hardware supports it.

## Cross-language integration

Rust classification engine + **Go FFI**. Cloud-native deployments are Go-shaped; the hot path is not.

### Why Rust for ML inference

- Near-C performance, zero-cost abstractions, low-latency
- Memory safety at compile time
- Ownership system + Rayon: data-race-free parallelism
- No GC pauses

Candle sits on those Rust properties with an ML-shaped API.

### Why Go FFI

Go owns the cloud-native control plane. FFI is the bridge:

- **Envoy**: semantic router as an [Envoy `ext_proc` filter](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter) in Go; FFI lets the filter call Rust classification without rewriting the Envoy layer
- **Kubernetes operators**: typically Go / controller-runtime; embed classification instead of another network hop
- **Service meshes**: Istio, Linkerd, Consul — Go; ML classification without breaking mesh control planes
- **API gateways**: Kong, Tyk and other Go components; semantic routing at the gateway without a new microservice

### Deployment flexibility

- **Embedded**: Go links the Rust library via CGO — lower latency, simpler deploy
- **Process isolation**: classification as a separate process (gRPC or Unix sockets)
- **Mixed**: Go for networking/orchestration, Rust for ML inference

Main routing logic, config, cache: **Go**. Compute-intensive classification: **Rust**. Clean FFI boundary.

## Performance characteristics (their qualitative table)

- **Single vs multi-task**: LoRA little help on single-task. Clear win when several classifications share one input.
- **Long-context**: Qwen3-Embedding routes on documents up to **32K** without truncation (beyond ModernBERT **8K**). Flash Attention 2 on compatible GPUs: advantage grows with context.
- **Multilingual**: routing where ModernBERT training data was thin.
- **High concurrency**: `OnceLock` removes lock contention; classification throughput can scale with CPU cores (their claim from the test above).
- **GPU**: Flash Attention 2 **3–4×** on attention is the **citation band**, more pronounced at long sequences.

## Future directions (named, not shipped here)

- More embedding models via a `CoreModel` trait
- Flash Attention 3 when Candle has it
- Quantization (4-bit, 8-bit)
- Custom LoRA adapters for domain-specific routing
- FFI for Python, Java, C++

Foundation: new research without an architecture rewrite. FFI stays a stable interface so Rust can evolve under existing Go deployments.

## Resources

- [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
- [Candle](https://github.com/huggingface/candle)
- [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m)
