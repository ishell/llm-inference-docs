---
source: https://vllm.ai/blog/2025-10-16-vllm-tpu
lang: en
fetched: 2026-09-04
---

# vLLM TPU: A Unified Backend for PyTorch and JAX

Chinese: [zh/vllm/blog/architecture/vllm-tpu.md](../../../../zh/vllm/blog/architecture/vllm-tpu.md)

2025-10-16. **Google Team**. Engine: [tpu-inference](http://tpu.vllm.ai). Hardware door: [hardware-plugin.md](hardware-plugin.md); broader plugin story: [plugin-system.md](plugin-system.md). TPU is a plugin, not a fork. Throughput on the page is a then-current demo, not your chip’s SLA.

`pip install vllm-tpu`. Gen-1 (PyTorch/XLA + MPMD) reached ~**3.6×** Llama 3.1-8B on v6e-1 and ~**2.1×** 70B on v6e-8 by Cloud Next. This generation lowers Torchax and JAX through one JAX→XLA path. Same `llama.py`, about **+20%** throughput. Default: TPU-optimized code in tpu-inference if present, else upstream PyTorch via Torchax. **RPA v3**: arbitrary head dim / quant / TP; KV scatter fused into attention; three compiled subkernels; ~**+10%** vs v2 on Trillium (v6e). Default **SPMD**.

Local figures (copyright remains with the original site; study copies):

![vllm tpu](../../../../assets/vllm/blog/architecture/vllm-tpu/01-vllm-tpu.png)

vLLM TPU is now powered by tpu-inference: JAX and PyTorch under one lowering path. Faster than the previous generation, with broader model coverage and features. The page’s three jobs for developers:

1. Push TPU **performance** in the open.
2. **Flexibility**: run PyTorch model definitions on TPU without extra code changes, and treat JAX as a first-class path.
3. Keep vLLM **standardization**: same UX, telemetry, and interface.

![whats new](../../../../assets/vllm/blog/architecture/vllm-tpu/02-whats-new.png)

## Gen-1: racing Cloud Next

In February 2025, [vLLM V1](https://docs.vllm.ai/en/latest/usage/v1_guide.html) was just taking shape. A small team of Googlers and core contributors aimed to ship a performant TPU backend on a handful of models in time for [Cloud Next 2025](https://cloud.withgoogle.com/next/25). Two months, three hard problems:

- **V1 integration.** The new V1 path needed a new ragged paged attention kernel ([RPA v2](https://github.com/pytorch/xla/blob/master/torch_xla/experimental/pallas_kernels/ragged_paged_attention_v2.py)), mainly for chunked prefill and prefix caching. Those KV techniques were familiar on TPU; making them TPU-friendly *with* vLLM’s paged attention was not.
- **MPMD.** vLLM then coordinated processes with [MPMD](https://en.wikipedia.org/wiki/Flynn%27s_taxonomy#Multiple_programs,_multiple_data_streams_\(MPMD\)). TPU’s compiler-centric model leans on [SPMD](https://en.wikipedia.org/wiki/Single_program,_multiple_data) to overlap multi-device and multi-host communication.
- **PyTorch/XLA ([PTXLA](https://github.com/pytorch/xla)).** PTXLA made the vLLM integration easier (PyTorch runs natively on TPU). Optimizing lower in the stack was another story.

They still improved throughput **3.6×** for Llama 3.1-8B on v6e-1 and **2.1×** for Llama 3.1-70B on v6e-8. vLLM TPU also made [the Cloud Next stage](https://www.youtube.com/live/Md4Fs-Zc3tg?si=t3V52Kac5Y5VTNN0&t=1137). The two progress charts below close that loop.

## This generation: tpu-inference

PTXLA was a real accomplishment. They still wanted more open-source TPU performance, and both PyTorch and JAX models on TPU along the most performant path they could keep.

### One backend: everything JAX→XLA

The redesign with [tpu-inference](http://tpu.vllm.ai) runs PyTorch (via [Torchax](https://google.github.io/torchax/)) and [JAX](https://docs.jax.dev/en/latest/index.html) through a single JAX→XLA lowering path.

Versus PyTorch/XLA, the page calls JAX the more mature stack: better coverage and performance for its [primitives](https://docs.jax.dev/en/latest/jax-primitives.html), especially for complex parallelism. So **every** vLLM model now lowers with JAX — even when the definition is PyTorch. Higher-level frameworks recede; kernel and compiler work get the time. To XLA, Torchax and JAX use the same high-performance primitives ahead of compilation. Dev notes: [torchax_model_development.md](https://github.com/vllm-project/tpu-inference/blob/main/docs/developer_guides/torchax_model_development.md).

That is the design *then*. They also say they will evaluate a **native PyTorch port** on TPU later, if it wins.

> **Takeaway 1:** vLLM TPU now lowers all models with JAX. With no model-code changes (e.g. [llama.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py)), throughput is ~**20%** higher — JAX primitives building the HLO that XLA compiles.

### Install, serve, two registries

One install path. Torchax and JAX are JAX underneath, so PyTorch-written and JAX-written models share dependencies:

```bash
pip install vllm-tpu
```

Serving:

```bash
MODEL_ID="google/gemma3-27b-it" # model registered in tpu-inference or vllm
vllm serve $MODEL_ID
```

Two model registries:

1. **tpu-inference** (default, [JAX model list](https://github.com/vllm-project/tpu-inference/tree/main/tpu_inference/models/jax))
2. **vLLM upstream** ([registry.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py))

![vllm serve model](../../../../assets/vllm/blog/architecture/vllm-tpu/03-vllm-serve-model.png)

Unification means less duplication of community model work, more time for TPU kernels and XLA. PyTorch (via Torchax) and JAX share kernels and compilers.

> **Takeaway 2:** Default is TPU-optimized code in tpu-inference if it exists; otherwise fall back to upstream PyTorch, lowered with JAX via [Torchax](https://google.github.io/torchax/user_guide/how-it-works/). For most users this is an implementation detail.

If Torchax already runs PyTorch on TPU and still compiles with JAX JIT, why rewrite some models in tpu-inference? Not to duplicate for its own sake.

They ship a few reference models so JAX users have a shallower ramp ([tpu_inference/models/jax](https://github.com/vllm-project/tpu-inference/tree/main/tpu_inference/models/jax)). Observation: torchax-lowered and naively reimplemented JAX models had roughly the same performance — Torchax is already efficient at converting high-level models.

The real gain, and the reason to keep reimplementation: optimize the JAX for TPU and use the architecture directly. Logical choices a vLLM developer makes when writing a model do not always favor TPU. The difference is not JAX vs Torchax; GPUs and TPUs want different strategies.

> **Takeaway 3:** For any model, it is *all* JAX under the hood. Unless logical differences in the implementation hurt TPU performance, a native JAX rewrite is unlikely to help. Keep the rewrite door if it is how you get the best out of TPUs.

### RPA v3

RPA v2 already lifted throughput. Broader OOTB models and use cases needed more flexibility. Four points on the page:

1. **More models.** v2 only supported head dim **128**. v3 takes arbitrary model specs, quantization dtypes, and arbitrary TP.
2. **Better performance.** v2 updated KV cache and ran attention sequentially. v3 fuses KV scatter into the RPA kernel and *completely* hides scatter latency during execution.
3. **Deployment flexibility.** v2 could waste work on decode-heavy or varied-length prefill. v3 compiles **three** subkernels: prefill-only, decode-only, mixed batch. Pairing the right subkernel at runtime saves DMA and compute. It also unlocks patterns like disaggregated serving.
4. **No flexibility tax.** v3 is ~**10%** higher throughput than v2 on Trillium (v6e). Models can also run on **v5p** (more tuning needed).

A technical deep dive on RPA v3 was promised in the docs.

> **Takeaway 4:** RPA v3 is flexible *and* performant — a reference for production-grade Pallas kernels in OSS. TPU-friendly MoE and MLA kernels are meant to land the same way.

### SPMD by default

This release makes [SPMD](https://en.wikipedia.org/wiki/Single_program,_multiple_data) the default programming model for vLLM TPU. Not the previous multi-worker model adapted from GPU. Developers write for one massive device; XLA partitions models and tensors and inserts communication.

> **Takeaway 5:** SPMD enables overlapping communication with computation. It is a shift toward native, compiler-first TPU integration.

### Bringing it together

![llama3 8b throughput progress](../../../../assets/vllm/blog/architecture/vllm-tpu/04-llama3-8b-throughput-progress.png)

![llama3 70b throughput progress](../../../../assets/vllm/blog/architecture/vllm-tpu/05-llama3-70b-throughput-progress.png)

From the February 2025 prototype to this post, those same workloads are nearly **2×–5×**, with better coverage and usability.

> **Takeaway 6:** Versus the first TPU prototype in Feb 2025, vLLM TPU was then nearly **5×**. With that foundation, the next increment of TPU inference performance can be pushed in the open.

## Models, features, and what’s next

Foundational release: vLLM TPU will cut regular OSS releases. Each release, CI/CD publishes tables of vetted vLLM-native models. They also keep a list of stress-tested tpu-inference models, mainly as a JAX reference. Features go through testing before a release.

**Supported model families (then)**

- Dense
- Multimodal (**tpu-inference models only**)

> **Note on model support:** Until more capabilities land, start from the stress-tested list: [model_support_matrix.csv](https://github.com/vllm-project/tpu-inference/blob/main/support_matrices/model_support_matrix.csv). Components for larger, more complex models (XL MoE, vision encoders, MLA, …) were still landing in tpu-inference. To prioritize something: [feature request](https://github.com/vllm-project/tpu-inference/issues/new/choose).

**Supported / verified TPU generations**

- Trillium (v6e), v5e

**Features**

- Prefix caching
- Chunked Prefill
- Multimodal Inputs
- SPMD
- Structured Decoding
- Speculative decoding: Ngram
- Out-of-tree model support
- Optimized Runtime Sampling (top k, top p, temperature, logit output)
- Quantization (weights, activations, and KV cache)

**TPU-friendly kernels**

- Ragged Paged Attention V3
- Collective Communication Matmul
- Quantized Matmul, Attention and KV Cache

**Experimental**

- v5p
- Multimodal (through Torchax)
- Multi-lora
- Speculative decoding: tree-based Eagle 3
- Single-host P/D disaggregated serving

**What’s next (as listed)**

- Sparsecore offloading
- Speculative decoding: Eagle 3, MTP
- TPU-friendly kernels: XL MoE, MLA
- RL integrations: single-host and multi-host; colocated and disaggregated; single-controller via Pathways; multi-sampling via prefix caching; weight sync and resharding; throughput-optimized rollout via Data Parallelism; LoRA; tool calls and multi-turn rollout. Partner projects: [Tunix](https://github.com/google/tunix), [MaxText](https://github.com/AI-Hypercomputer/maxtext), [SkyRL](https://github.com/NovaSky-AI/SkyRL)
- Distributed: multi-host dynamic P/D; prefix-cache offload to CPU and remote stores; optimized Data Parallel Attention load balancing. Partner project: [llm-d](https://github.com/llm-d/llm-d)
- [Contributions welcome](https://github.com/vllm-project/tpu-inference/blob/main/CONTRIBUTING.md)

## Try it out

Google Cloud: [GKE](https://cloud.google.com/tpu?hl=en#cloud-tpu-in-gke), [Compute Engine](https://cloud.google.com/tpu?hl=en), [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/open-models/vllm/use-vllm-tpu). Install and developer guides:

- [Contribution Guide](https://github.com/vllm-project/tpu-inference/blob/main/CONTRIBUTING.md)
- [Quick Start](https://github.com/vllm-project/tpu-inference/blob/main/docs/getting_started/quickstart.md)
- [Trillium (v6e) Recipes](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM)
- [Developer Guide: JAX](https://github.com/vllm-project/tpu-inference/blob/main/docs/developer_guides/jax_model_development.md)
- [Developer Guide: Torchax](https://github.com/vllm-project/tpu-inference/blob/main/docs/developer_guides/torchax_model_development.md)

Tutorials: GKE [here](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-vllm-tpu), Vertex AI [here](https://cloud.google.com/vertex-ai/generative-ai/docs/open-models/vllm/use-vllm-tpu).

## Acknowledgment

Thanks to the vLLM community. Special thanks to [Woosuk Kwon](https://github.com/WoosukKwon) for spearheading TPU’s V0 and continuing to support the growing team. Guidance throughout: [Simon Mo](https://github.com/simon-mo), [Robert Shaw](https://github.com/robertgshaw2-redhat), [Michael Goin](https://github.com/mgoin), [Yanping Huang](https://github.com/bignamehyp). Integral to V1 integration and the Cloud Next push: [Nicolo Lucchesi](https://github.com/NickLucche), [Alexander Matveev](https://github.com/alexm-redhat), [Akshat Tripathi](https://github.com/Akshat-Tripathi), [Saheli Bhattacharjee](https://github.com/sahelib25).
