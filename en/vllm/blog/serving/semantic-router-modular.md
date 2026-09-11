---
source: https://vllm.ai/blog/2025-10-27-semantic-router-modular
lang: en
fetched: 2026-09-04
---

# From Monolithic to Modular: Scaling Semantic Routing with Extensible LoRA

Chinese: [zh/vllm/blog/serving/semantic-router-modular.md](../../../../zh/vllm/blog/serving/semantic-router-modular.md)

2025-10-27. **Ivar Flakstad (Hugging Face), OneZero-Y, Huamin Chen (Red Hat), Xunzhuo Liu (Tencent)**. Repo: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router). Launch: [semantic-router.md](semantic-router.md). Shared-base LoRA lands in [Iris](semantic-router-iris.md). Spine: [signal-decision](semantic-router-signal.md). HaluGate uses the same Candle door: [halugate](halugate.md). Later mmBERT refresh: [athena](semantic-router-athena.md). Do not confuse with the in-engine [router.md](router.md). Flash Attention 2× / tok/s citations are **literature**, not a vLLM-SR cluster run.

Siblings: [amd](semantic-router-amd.md), [mom-amd](semantic-router-mom-amd.md), [vision](semantic-router-vision.md), [themis](semantic-router-themis.md), [session](semantic-router-session.md), [fusion](semantic-router-fusion.md), [micro-agent](semantic-router-micro-agent.md), [mom](semantic-router-mom.md).

When each classification request runs several fine-tuned models independently, cost grows **linearly** with the number of models. This post is the Rust classification-layer refactor: architectural modularity, LoRA, concurrency.

Local figures (copyright remains with the original site; study copies):

![modular](../../../../assets/vllm/blog/serving/semantic-router-modular/01-modular.png)

**Figure 1.** Layered candle-binding: core independent of a specific architecture; `DualPathUnifiedClassifier` picks fine-tuned vs LoRA.

## Background: BERT toward a modular system

Previously: BERT / ModernBERT for intent and jailbreak. ModernBERT is strong on English classification, with limits they name:

- **Language coverage:** original ModernBERT multilingual support is thinner than models trained on more diverse data. Note on the page: [mmBERT](https://huggingface.co/blog/mmbert) (1800+ languages) shipped **after** this refactoring began — an alternative to the multilingual problem, later centered in [Athena](semantic-router-athena.md).
- **Context length:** ModernBERT to **8,192** tokens via RoPE; Qwen3-Embedding cited at **32,768**.
- **Model coupling:** classification logic tied to specific architectures; hard to add new models.

Modular architecture: newer models (mmBERT included) can sit beside Qwen3-Embedding and EmbeddingGemma; the router picks per task.

## Architectural restructuring

Layered architecture in the **candle-binding** crate. Core stays independent of a given model; new architectures without rewriting existing code. `DualPathUnifiedClassifier` selects traditional fine-tuned vs LoRA-adapted based on the task.

## Long-context embedding models

### Qwen3-Embedding

Up to **32,768** tokens ([Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)). RoPE for extended context. Trained on text from **100+** languages (model card citation) — multilingual routing where ModernBERT-only struggled.

### EmbeddingGemma-300M

Smaller, quality-focused. Context **2,048**. Matryoshka: truncate embeddings to **768 / 512 / 256 / 128** without retraining ([embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)). Multi-Query Attention: **3** query heads, **1** KV head. Dense bottleneck after transformer blocks: **768 → 3072 → 768**.

## LoRA for multi-task classification

Naive: intent + PII + jailbreak = three full fine-tuned forwards.

![full params](../../../../assets/vllm/blog/serving/semantic-router-modular/02-full-params.png)

**Figure 2.** Each task pays a full base transformer. O(n) in the number of tasks.

LoRA shares the base pass:

![lora](../../../../assets/vllm/blog/serving/semantic-router-modular/03-lora.png)

**Figure 3.** One base forward; adapters typically **<1%** of parameters.

`parallel_engine.rs` uses [Rayon](https://github.com/rayon-rs/rayon) so adapters run concurrently. Three classifications: one full pass + three light adapters, not three full models. LoRA wins on **multi-task**, not single-task (fine-tuned may be faster when there is nothing to share).

## Concurrency through `OnceLock`

`lazy_static` for global classifier state → lock contention under concurrent load. Replaced with [`OnceLock`](https://doc.rust-lang.org/std/sync/struct.OnceLock.html): lock-free reads after init (pointer reads, no sync). Tests in `oncelock_concurrent_test.rs`: **10** threads, **30** classifications; throughput claimed to scale linearly with thread count. Concurrent requests no longer queue behind a mutex.

### Flash Attention for GPU acceleration

Flash Attention 2 optional for CUDA builds; Ampere+ (compute capability ≥ 8.0). Blocked attention in on-chip SRAM vs repeated DRAM reads.

Citations on the page (not a vLLM-SR cluster measurement):

- ModernBERT: up to **~3×** faster self-attention, less memory; alternating global (every third layer) vs local sliding-window
- Qwen3: FlashAttention-2 up to **~4×** on attention; 14B variant **70–110** vs **30–35** tok/s without it, more pronounced at long context

Cargo feature: deploy without compatible GPUs; turn it on when hardware supports it.

## Cross-language integration

Rust classification engine + **Go FFI**.

**Why Rust:** near-C performance; memory safety; ownership vs data races with Rayon; no GC pauses. Candle on that stack.

**Why Go FFI:** Envoy `ext_proc` filter is Go — FFI lets the filter call Rust classification without rewriting the Envoy layer. Kubernetes operators (controller-runtime) can embed classification instead of another network hop. Service meshes (Istio, Linkerd, Consul) and API gateways with Go components can keep ML classification without extra microservices.

**Deployment flexibility:**

- **Embedded:** Go links the Rust lib via CGO
- **Process isolation:** separate process, gRPC or Unix sockets
- **Mixed:** Go networking / orchestration + Rust inference

Main routing, config, cache in Go; compute-intensive classification in Rust.

## Performance characteristics (as they list)

- **Single vs multi-task:** LoRA little help if there is no sharing; multi-task on the same input is where adapters pay off. Speedup = base compute vs adapter compute.
- **Long context:** Qwen3-Embedding to 32K without truncation vs ModernBERT 8K. FA2 on compatible GPUs helps more as length grows.
- **Multilingual:** routing for languages where ModernBERT training was thin.
- **High concurrency:** `OnceLock` removes lock contention; classification throughput with CPU cores.
- **GPU:** FA2 **3–4×** on attention (citation), more at long sequences.

## Future directions

Add embeddings by implementing the `CoreModel` trait; Flash Attention 3 when Candle has it; 4-bit / 8-bit quantization; custom LoRA for domain routing; FFI for Python / Java / C++. Modular foundation so research can land without an architectural rewrite; FFI stays stable while Rust evolves under Go deployments.

## Resources

- [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
- [Candle](https://github.com/huggingface/candle)
- [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m)
