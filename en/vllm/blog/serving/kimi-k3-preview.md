---
source: https://vllm.ai/blog/2026-07-22-kimi-k3-preview
lang: en
fetched: 2026-09-04
---

# A Preview of Production-Scale Kimi K3 Support on vLLM

Chinese: [zh/vllm/blog/serving/kimi-k3-preview.md](../../../../zh/vllm/blog/serving/kimi-k3-preview.md)

2026-07-22. **vLLM Team**. Weights planned **2026-07-27**. Launch numbers: [kimi-k3.md](kimi-k3.md). Tool-calling handshake cousin: [kimi-k2-accuracy.md](kimi-k2-accuracy.md). CUDA debug cousin named in the collaboration list: [cuda-debugging-source.md](../architecture/cuda-debugging-source.md). DCP PR mentioned there: [dcp.md](../performance/dcp.md). Cache / P/D: [mooncake.md](mooncake.md). Study note; optimization still in flight. **Not a new engine** — hybrid cache, kernels, recipes. **Not the launch guide** — that is [kimi-k3.md](kimi-k3.md).

Last week Moonshot AI [introduced Kimi K3](https://www.kimi.com/blog/kimi-k3): 2.8T, native vision, 1M context, Kimi Delta Attention (KDA), Attention Residuals (AttnRes), highly sparse MoE. Open weights by 2026-07-27. vLLM, Moonshot, NVIDIA, AMD, and the community are closing integration so the open-source community can serve from day 0.

This post is a **preview**. Core model path, KDA-aware prefix caching, multimodal, tool-calling parsers, and hardware-specific work are taking shape. Selected trusted partners (dual approval: Moonshot + vLLM/Inferact) have started deployment validation on the same code being prepared for open source.

As the announcement blog said, KDA poses new challenges for conventional prefix caching. Moonshot contributed a corresponding implementation to vLLM, to ship with the weights. Design write-up later — now [kimi-k3.md](kimi-k3.md) plus this preview.

**Figure (social preview, not stored locally):** original `/assets/figures/2026-07-22-kimi-k3-preview/social-preview.png`.

Local figures (copyright remains with the original site; study copies):

![kda prefix state](../../../../assets/vllm/blog/serving/kimi-k3-preview/01-kda-prefix-state.png)

![fine grained prefix cache](../../../../assets/vllm/blog/serving/kimi-k3-preview/02-fine-grained-prefix-cache.png)

## TL;DR

- **Day-0 open-source serving:** model implementation, Docker images, deployment recipes, production validation for the weight release.
- **A new hybrid architecture:** KDA-dominant linear attention with periodic full-attention layers, AttnRes across depth, Stable LatentMoE, native vision.
- **Prefix caching required core changes:** physical KDA state-block size split from prefix-match granularity — useful partial prefix hits without storing recurrent state at every small attention block.
- **Kernel work:** FlashKDA, fused KDA Decode, fused KDA projections and convolution, fused AttnRes, reimplemented MLA, SiTU-enabled MXFP4 MoE, optimized expert routing.
- **NVIDIA and AMD:** NVIDIA kernels under final tuning; initial AMD path with FlyDSL MoE already in place, broader validation ongoing.

## Kimi K3 at a Glance

Not a larger Kimi K2. The serving problem changes in several dimensions at once.

| Property | Kimi K3 configuration | Serving implication |
| --- | --- | --- |
| **Model scale** | **2.8T parameters** | Large-scale expert parallelism, high-bandwidth accelerator domains |
| **Context length** | **1M tokens** | Cache capacity, prefix reuse, chunked Prefill, Prefill/Decode disaggregation become first-order |
| **Attention** | **Hybrid KDA and full attention** | Recurrent-state caches and paged KV must advance on the **same** logical prefix |
| **Depth** | **Attention Residual** | Cross-layer representation reads/writes need dedicated kernels |
| **MoE** | **896 routed experts, 16 active per token, plus shared experts** | Routing, dispatch, load balance, MoE kernels sit on e2e |
| **Quantization** | **MXFP4 weights in the provided release configuration** | Efficient FP4 MoE path with Kimi K3’s **SiTU** activation |
| **Multimodality** | **Native vision with a vision tower** | Multimodal preprocessing (image-only) and a robust vision parallelism strategy |

Each choice moves cost somewhere new. KDA avoids a conventional KV pair per past token, but introduces a large recurrent state. AttnRes loosens a single residual stream, but adds cross-layer memory traffic. Extreme MoE sparsity avoids activating all 2.8T per token, but raises the stakes for routing and communication. vLLM’s job is to make the pieces work behind one familiar serving API.

## A Collaboration Built Over Multiple Kimi Generations

- [GOSIM 2024](https://china2024.gosim.org/schedules/vllm-in-moonshot.html): Moonshot engineers on vLLM at scale inside Moonshot, and vLLM + Mooncake Prefill/Decode-disaggregated architecture.
- Later, Kimi K2 training and inference at the [vLLM Beijing Meetup](https://pytorch.org/blog/vllm-beijing-meetup-advancing-large-scale-llm-deployment/) — strict SLOs on online traffic and RL workloads.
- Day-0 launch partner for Kimi K2, Kimi K2-Thinking, Kimi K2.5, Kimi Linear, and so on.
- Deep technical collaboration: [Kimi K2 tool-calling accuracy](kimi-k2-accuracy.md), [improved CUDA debugging](../architecture/cuda-debugging-source.md), [decode context parallelism](https://github.com/vllm-project/vllm/pull/23734) ([dcp.md](../performance/dcp.md)), Mooncake-based P/D, large-scale performance validation. Kimi K2.5 also in public [InferenceX serving results](https://inferencex.semianalysis.com/inference?g_rundate=2026-04-07&g_model=Kimi-K2.5&g_runid=24100518225&i_gpus=gb200_dynamo-vllm&i_dstart=2026-04-07&i_dend=2026-04-07).

Day-0 is rarely one PR after an announcement. Architecture details early, real checkpoints under realistic parallelism, gaps in the engine, upstream work that remains useful after one launch.

## The Hardest Part: Prefix Caching for KDA

Full attention and KDA remember a prefix in very different ways.

Full attention: a prefix is per-token key and value vectors. vLLM stores those in paged blocks, hashes complete token blocks, reuses a matching sequence of blocks.

KDA is recurrent. Instead of a conventional KV pair per token, each KDA layer advances a matrix-like recurrent state plus a short convolution state. To resume from a cached prefix, the engine needs the KDA state **at the exact prefix boundary**. Replaying an earlier state to reach that boundary would erase much of the benefit of prefix caching.

**Figure.** How conventional attention and KDA represent cached prefixes.

The straightforward solution — store KDA state at every small attention-cache boundary — is too expensive. A KDA state is much larger than one ordinary token’s KV, so implementations use a relatively large physical state block to amortize storage. Before this work, that physical block size also constrained where a prefix-cache hit could land. With a multi-thousand-token state block, two requests sharing almost the entire prompt could still miss, because their common boundary did not fill the same physical block.

The new design separates three concepts that used to move together:

- **Physical block size:** how KDA state and full-attention KV are allocated on the GPU.
- **Scheduler alignment:** where execution must stop so all cache groups remain consistent.
- **Prefix-match unit:** the finer token interval at which a shared prefix is hashed and may be matched.

**Figure.** Fine-grained prefix matching inside a larger physical KDA state block.

vLLM can register a valid KDA state at a fine-grained boundary **inside** a larger physical state block. Later hit on that partial block: copy into a private destination before the request extends it. Copy-on-write keeps the shared cached prefix; the new request can continue safely.

Easy-to-miss details:

- Scheduler stops at the right block and hash boundaries so the recurrent state being registered really corresponds to the advertised token prefix.
- Full-attention and KDA cache groups agree on one `num_computed_tokens`, even though physical block sizes differ.
- Partial cache entries use chained, fine-grained hashes so a boundary identifies the **entire** prefix, not only the tail tokens.
- Same-step reuse is deferred until the state copy is safe — no races between cache registration and extension.
- Cache transfer and disaggregated Prefill/Decode can carry the same logical prefix across workers.

Motivated by Kimi K3 and other hybrid attention models, but **core vLLM infrastructure**, not a model-specific shortcut. Design, invariants, and benches in more detail: now [kimi-k3.md](kimi-k3.md).

## Performance Work: Removing the New Bottlenecks

| Area | Current status |
| --- | --- |
| **Model and configuration** | Language and vision model definitions integrated; separate **NVIDIA** and **AMD** implementations where hardware paths differ |
| **Optimized MLA for native P/D** | Manual kernel fusion, separate Prefill/Decode paths. Gate projection parallel with attention; multi-stream in Decode; fused epilogue in Prefill — aimed at P/D |
| **Serving semantics** | Chat rendering, tokenizer, streaming parsing, tool calls, reasoning output, structured-output — **final e2e validation** |
| **KDA Prefill** | FlashKDA and Triton integrated; final backend selection and numerical validation **in progress** |
| **KDA Decode** | Fused **NVIDIA** Decode kernel: convolution, recurrent KDA update, gating, normalization; portable fallbacks retained |
| **Prefix caching** | Fine-grained partial prefix hits for hybrid full-attention + recurrent-state caches integrated; disaggregated and offload **being validated** |
| **Attention Residuals** | Triton and **NVIDIA** kernels; fusion of residual addition and output RMSNorm on supported shapes |
| **MoE** | **SiTU** wired into **MXFP4 TRTLLM-Gen** and **DeepGEMM**; optimized grouped top-k routing. **AMD** FlyDSL **MLIR** stack: hardware-tuned **A16W4/A8W4** fused ops + **SiTU** |
| **Production stack** | Non-disaggregated serving working; Dynamo + vLLM + Mooncake disaggregated serving, expert parallelism, vendor verification in the **final validation loop** |

Kimi K3 changes the hot path, so the team optimized more than the attention kernel.

### KDA Prefill and Decode

Prefill integrates FlashKDA and Flash Linear Attention (FLA). Around the core recurrence: fuse input projections and causal convolution; gather initial recurrent states in one operation.

Decode: fused NVIDIA kernel on supported architectures and shapes. Convolution, KDA state update, output gate, and normalization together instead of a launch per piece every generated token. Kimi K3 has many KDA layers; a small per-layer launch or memory penalty becomes a large **TPOT** penalty.

### Attention Residuals

AttnRes retrieves from representations written by earlier layer blocks, not only one uniformly accumulated residual stream. Naive: extra reads, writes, reductions, and normalization launches through the **93-layer** network.

Release branch: Triton plus an NVIDIA kernel that fuse residual update, AttnRes mixing, and output RMSNorm where supported. Sequence-parallel work shards attention-residual traffic across ranks. Early kernel-level results encouraging; e2e still being measured across Prefill lengths and parallel configs.

### Optimized MLA module for native P/D disaggregation

Kimi K3 still uses MLA every four layers. Previously, vLLM leaned on a `torch.compile` custom-fusion path: slow startup, many kernels still unfused. This release: a new MLA module with **manual** fusion. Prefill and Decode need different kernel launch orders, so two code paths with different fusion patterns, specialized for P/D. Kimi K3 also adds a gate projection that can run in parallel with main attention: optional multi-stream for the gate in Decode; in Prefill — where multi-stream overlap is not optimal — fuse elementwise multiply and sigmoid into the gate-projection epilogue.

### MXFP4 MoE

Release configuration: MXFP4 weights and SiTU activation. Before this work, the MXFP4 TRTLLM-Gen path did not support SiTU and fell back slower. vLLM now maps Kimi K3’s SiTU parameters into the optimized FP4 expert path and chunks large token-by-top-k launch grids safely.

Validated on **16-GPU DP16+EP16**: all ranks selected the optimized MXFP4 backend and passed correctness.

AMD: Kimi K3 MoE on FlyDSL’s MLIR Python kernel stack — hardware-tuned A16W4/A8W4 quantized fused operators and SiTU, on FlyDSL’s modular abstractions.

## What to Expect on Open-Source Day

Planned day-0 package:

- vLLM model, parser, cache, and kernel integration
- initial open-source Docker images
- validated launch recipes for NVIDIA configurations
- an initial AMD path with FlyDSL MoE kernel; more ROCm tuning to follow
- multimodal, tool-use, reasoning, and structured-output examples
- initial performance results

Trusted deployment partners already exercising the release candidate under dual approval from Moonshot and vLLM/Inferact. Production feedback without distributing prerelease artifacts broadly. Tests the complete serving system — frontend semantics, batching, cache transfer, expert parallelism, observability, failure handling — not only isolated kernels.

## Acknowledgements

Joint effort: model vendor, inference engine, hardware communities.

**Moonshot AI** — Kimi K3, architecture ahead of the weight release, initial model integration and KDA prefix-caching, correctness and production validation.

**Inferact** — integrating into vLLM, extending the core cache manager for partial hybrid prefix hits, serving semantics and multimodal, deployment recipes, e2e performance.

**NVIDIA** — KDA Decode and Attention Residual kernels, MXFP4 MoE, performance across the board.

**AMD** — initial day-0 ROCm; continuing to expand Kimi K3 across AMD GPUs.

The broader open-source community for anticipation, testing, and feedback.

## One More Thing: Why the Announcement and Open-Source Release Are Separated

Announce the model first; release weights and inference-engine support later.

vLLM proposed the split; Moonshot agreed. Practical: a frontier-model announcement has last-mile uncertainty. The model team is simultaneously stabilizing products, APIs, evaluations, safety, documentation, commercial launch. If open-source weights and open-source support must land at the **same** moment, a community project like vLLM suffers from the moving deadline.

Separating timelines:

1. The vendor can freeze the final checkpoint, configuration, tokenizer, and serving semantics.
2. The open-source inference team gets a stable window for correctness, performance, Docker, recipes.
3. The community gets a public, bounded expectation instead of an ambiguous “coming soon.”

Not a retreat from day-0. A more sustainable way to deliver day-0 against the artifact users will actually download.
