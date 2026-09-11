---
source: https://vllm.ai/blog/2026-07-22-kimi-k3-preview
lang: en
fetched: 2026-09-05
---

# A Preview of Production-Scale Kimi K3 Support on vLLM

Chinese: [zh/vllm/blog/serving/kimi-k3-preview.md](../../../../zh/vllm/blog/serving/kimi-k3-preview.md)

2026-07-22. vLLM Team. Weights were planned for **2026-07-27**; the launch note is [kimi-k3.md](kimi-k3.md). Announcement: [Moonshot](https://www.kimi.com/blog/kimi-k3); [tweet](https://x.com/Kimi_Moonshot/status/2077830229968683203). Selected trusted partners need dual approval from Moonshot **and** vLLM/Inferact, on the same code being prepared for open source. KDA prefix cache: Moonshot contributed the implementation, to ship with the weights; design in a later post. Quote from the page: **vLLM is proud to be a long-term partner of Moonshot AI and a popular inference engine for Kimi-series models.** Social preview image skipped. Local figures remain copyright of the original site.

Last week Moonshot introduced Kimi K3: 2.8T, native vision, 1M context, Kimi Delta Attention (KDA), Attention Residuals (AttnRes), highly sparse MoE. The open-source community is excited that open-weight models are catching proprietary ones. Weights were then scheduled for 2026-07-27. Meanwhile vLLM, Moonshot, NVIDIA, AMD, and the broader community were finishing integration and validation so the community could serve from **day 0**.

This post is a preview; performance work was still ongoing. The core model path, KDA-aware prefix caching, multimodal integration, tool-calling parsers, and hardware-specific optimizations were already taking shape.

## TL;DR

- **Day-0 open-source serving:** model implementation, Docker images, deployment recipes, and production validation aimed at the weight release.
- **A new hybrid architecture:** KDA-dominant linear attention plus periodic full attention, AttnRes across depth, Stable LatentMoE, native vision.
- **Prefix caching required core changes:** physical KDA state-block size is separated from prefix-match granularity, so useful partial prefix-cache hits are possible without storing recurrent state at every small attention block.
- **Kernel work across the stack:** FlashKDA, fused KDA decode, fused KDA projections and convolution, fused AttnRes, reimplemented MLA, SiTU-enabled MXFP4 MoE, optimized expert routing.
- **NVIDIA and AMD:** NVIDIA kernels under final tuning; an initial AMD path with a FlyDSL MoE kernel already in place and moving through broader validation.

## Kimi K3 at a Glance

K3 is not a larger K2. The serving problem changes in several dimensions at once.

| Property | Kimi K3 | Serving implication |
| --- | --- | --- |
| **Model scale** | **2.8T** | Large-scale expert parallelism and high-bandwidth accelerator domains |
| **Context** | **1M tokens** | Cache capacity, prefix reuse, chunked prefill, and P/D disaggregation become first-order |
| **Attention** | **Hybrid KDA + full attention** | Recurrent state caches and paged KV must advance on the same logical prefix |
| **Depth** | **AttnRes** | Cross-layer reads and writes need dedicated kernels |
| **MoE** | **896 routed, 16 active per token, plus shared** | Routing, dispatch, load balance, and MoE kernels dominate end-to-end |
| **Quantization** | **MXFP4** in the provided release | An efficient FP4 MoE path that also takes K3’s **SiTU** |
| **Multimodality** | Native vision + vision tower | Multimodal preprocessing (image-only then) and a robust vision parallelism strategy |

For inference systems, each choice moves cost somewhere new. KDA avoids retaining a conventional KV pair for every past token, but introduces a large recurrent state. AttnRes reduces the limits of a single residual stream, but creates extra cross-layer memory traffic. Extreme sparsity avoids activating all 2.8T per token, but raises the stakes for routing and communication. vLLM’s job is to make the pieces work together behind one familiar serving API.

## A Collaboration Built Over Multiple Kimi Generations

K3 continues a long Moonshot–vLLM collaboration.

- [GOSIM 2024](https://china2024.gosim.org/schedules/vllm-in-moonshot.html): Moonshot on vLLM at scale inside Moonshot, and vLLM + Mooncake P/D disaggregation.
- [vLLM Beijing Meetup](https://pytorch.org/blog/vllm-beijing-meetup-advancing-large-scale-llm-deployment/): K2 training and inference, online traffic under strict SLOs, RL workloads.
- vLLM was a day-0 partner for K2, K2-Thinking, K2.5, Kimi Linear, and so on.
- Deep collaboration: [Kimi K2 tool-calling](https://vllm.ai/blog/Kimi-K2-Accuracy) for correctness, [CUDA debugging](https://vllm.ai/blog/improved-cuda-debugging), [decode context parallelism](https://github.com/vllm-project/vllm/pull/23734), Mooncake P/D, large-scale performance validation. K2.5 also appeared in public [InferenceX](https://inferencex.semianalysis.com/inference?g_rundate=2026-04-07&g_model=Kimi-K2.5&g_runid=24100518225&i_gpus=gb200_dynamo-vllm&i_dstart=2026-04-07&i_dend=2026-04-07). Study notes: [kimi-k2-accuracy.md](kimi-k2-accuracy.md), [cuda-debugging-source.md](../dev/cuda-debugging-source.md), [dcp.md](../features/dcp.md), [mooncake.md](../features/mooncake.md).

That history matters. Day-0 is rarely one PR written after the announcement. It comes from sharing architecture early, testing real checkpoints under realistic parallelism, finding gaps in the serving engine, and upstreaming improvements that remain useful after one launch. Quote from the page: **vLLM is proud to be a long-term partner of Moonshot AI and a popular inference engine for Kimi-series models.**

Now the hardest technical challenge they ran into.

## The Hardest Part: Prefix Caching for KDA

Conventional full attention and KDA remember a prefix in very different ways.

Full attention: a prefix is per-token K/V. vLLM stores those in paged blocks, hashes complete token blocks, and can reuse a matching sequence of blocks for another request.

KDA is recurrent. Each layer advances a matrix-like recurrent state plus a short convolution state. To resume from a cached prefix, the engine needs the KDA state **at the exact prefix boundary**. Replaying an earlier state to that boundary would erase much of the benefit of prefix caching.

![conventional attention vs KDA cached prefixes](../../../../assets/vllm/blog/serving/kimi-k3-preview/01-kda-prefix-state.png)

The straightforward solution — store KDA state at every small attention-cache boundary — is too expensive. A KDA state is much larger than one ordinary token’s KV, so implementations use a relatively large physical state block to amortize storage. Before this work, that physical block size also constrained where a prefix-cache hit could land. With a multi-thousand-token state block, two requests sharing almost the entire prompt could still miss because their common boundary did not fill the same physical block.

The new design separates three concepts that used to move together:

- **Physical block size:** how KDA state and full-attention KV are allocated on the GPU.
- **Scheduler alignment:** where execution must stop so all cache groups remain consistent.
- **Prefix-match unit:** the finer token interval at which a shared prefix is hashed and may be matched.

![fine-grained prefix matching inside a larger physical KDA state block](../../../../assets/vllm/blog/serving/kimi-k3-preview/02-fine-grained-prefix-cache.png)

This lets vLLM register a valid KDA state at a fine-grained boundary **inside** a larger physical state block. When a later request hits that partial block, the cached state is **copied** into a private destination before the request extends it. Copy-on-write preserves the shared prefix while the new request continues generation safely.

The implementation also handles details that are easy to miss:

- The scheduler stops at the right block and hash boundaries so the recurrent state being registered really corresponds to the advertised token prefix.
- Full-attention and KDA cache groups agree on one `num_computed_tokens`, even though their physical block sizes differ.
- Partial cache entries use chained, fine-grained hashes so a boundary identifies the **entire** prefix, not only the tail tokens.
- Same-step reuse is deferred until the state copy is safe, avoiding races between registration and extension.
- Cache transfer and disaggregated P/D paths can carry the same logical prefix across workers.

This work was motivated by K3 and many other hybrid attention models, but it is **core vLLM infrastructure**, not a model-specific shortcut. The vLLM and Moonshot teams collaborated deeply on the design. They will publish a separate post with the design, invariants, and benchmarks. Launch numbers: [kimi-k3.md](kimi-k3.md).

## Performance Work: Removing the New Bottlenecks

Progress at preview time:

| Area | Then-status |
| --- | --- |
| **Model and configuration** | Language and vision definitions integrated; separate **NVIDIA / AMD** implementations where hardware paths differ |
| **Optimized MLA for native PD** | Manual kernel fusion, separate prefill/decode paths. Gate projection runs in parallel with attention; multi-stream in decode, fused epilogue in prefill — aimed at PD disaggregation |
| **Serving semantics** | Chat rendering, tokenizer, streaming parse, tool calls, reasoning, structured-output implemented and under **final end-to-end validation** |
| **KDA prefill** | FlashKDA and Triton integrated; final backend selection and numerical validation **in progress** |
| **KDA decode** | Fused **NVIDIA** decode kernel covering convolution, recurrent update, gating, and normalization integrated; portable fallbacks retained |
| **Prefix caching** | Fine-grained partial hits for hybrid full-attention + recurrent-state caches integrated; disaggregated and offload **being validated** |
| **AttnRes** | Triton and **NVIDIA** kernels integrated, including fusion of residual add and output RMSNorm on supported shapes |
| **MoE** | **SiTU** wired into **MXFP4 TRTLLM-Gen** and **DeepGEMM**; optimized grouped top-k routing. **AMD** implements FlyDSL **MLIR** with hardware-tuned **A16W4/A8W4** fused ops and **SiTU** |
| **Production stack** | Non-disaggregated serving working; Dynamo + vLLM + Mooncake disaggregation, EP, and vendor verification in the **final validation loop** |

K3 changes the hot path, so the team optimized more than the attention kernel.

### KDA prefill and decode

Prefill integrates FlashKDA and Flash Linear Attention (FLA). Around the core recurrence, vLLM fuses input projections and causal convolution, and gathers initial recurrent states in one operation.

Decode uses a fused NVIDIA kernel on supported architectures and shapes. Short convolution, KDA state update, output gate, and normalization are not launched separately for every token. K3 has many KDA layers; a small per-layer launch or memory penalty quickly becomes a large TPOT penalty.

### Attention Residuals

AttnRes retrieves from representations written by earlier layer blocks rather than relying on one uniformly accumulated residual stream. A naive implementation creates extra reads, writes, reductions, and normalization launches throughout the **93-layer** network.

The release branch includes a Triton implementation and an NVIDIA kernel that fuse residual update, AttnRes mixing, and output RMSNorm for supported cases. Sequence-parallel work also shards attention-residual traffic across ranks. Early kernel-level results were encouraging; end-to-end gains were still being measured across prefill lengths and parallel configurations.

### Optimized MLA module for native PD disaggregation

K3 still uses MLA every four layers. In the previous model, vLLM relied heavily on a `torch.compile` custom-fusion path to map small kernels into fused kernels, which slowed startup and still left many kernels unfused. This release implements a new MLA module that fuses those kernels manually. MLA also needs different launch orders for prefill and decode, so there are two code paths with different fusion patterns, specialized for PD-disaggregated deployment. K3 also introduces a gate projection that can run in parallel with the main attention path. Decode optionally adds multi-stream for the gate projection; in prefill, where multi-stream overlap is not optimal, elementwise multiply and sigmoid are fused into the gate-projection epilogue.

### MXFP4 MoE

The release configuration uses MXFP4 weights and SiTU. Before this work, the MXFP4 TRTLLM-Gen path did not support SiTU and would fall back to a slower implementation. vLLM now maps K3’s SiTU parameters into the optimized FP4 expert path and chunks large token-by-top-k launch grids safely.

This was already validated on a **16-GPU DP16+EP16** configuration: all ranks selected the optimized MXFP4 backend and passed correctness checks.

On AMD, K3 MoE is supported on FlyDSL’s MLIR Python kernel stack, including hardware-tuned A16W4/A8W4 quantized fused operators and SiTU, built on FlyDSL’s modular abstractions.

## What to Expect on Open-Source Day

The planned day-0 package:

- vLLM model, parser, cache, and kernel integration;
- initial open-source Docker images;
- validated launch recipes for NVIDIA configurations;
- an initial AMD path with FlyDSL MoE, more ROCm tuning to follow;
- multimodal, tool-use, reasoning, and structured-output examples;
- initial performance results.

Trusted deployment partners were already exercising the release candidate under dual approval from Moonshot and vLLM/Inferact. Real production feedback without broadly distributing prerelease weights. It also tests the complete serving system — frontend semantics, batching, cache transfer, EP, observability, failure handling — not only isolated kernels. Launch numbers: [kimi-k3.md](kimi-k3.md).

## Acknowledgements

K3 day-0 is a joint effort across the model vendor, inference engine, and hardware communities.

- **Moonshot:** created K3; shared architecture ahead of the weight release; contributed initial model integration and KDA prefix-caching; collaborated on correctness and production validation.
- **Inferact:** integrated the model into vLLM; extended the core cache manager for partial hybrid prefix hits; serving semantics and multimodal; deployment recipes; end-to-end performance.
- **NVIDIA:** KDA decode and AttnRes kernels, MXFP4 MoE, performance across the board.
- **AMD:** initial day-0 ROCm, and continuing to expand K3 across AMD GPUs.
- The broader open-source community: anticipation, testing, feedback. The page looks forward to putting weights and engine support in your hands.

## One More Thing: Why the Announcement and Open-Source Release Are Separated

K3 also features a release process the page hopes more model vendors will consider: **announce the model first, then release the weights and inference-engine support later.**

The vLLM team proposed the separation; Moonshot agreed and executed. The reason is practical. A frontier-model announcement has unavoidable last-mile uncertainty. The model team is simultaneously stabilizing products, APIs, evaluations, safety, documentation, and the commercial launch. If open-source weights and open-source support must land at the **exact same moment**, a community project such as vLLM suffers from the moving deadline.

Separating the timelines gives both sides a better contract:

1. The model vendor can concentrate on its product launch and freeze the final checkpoint, configuration, tokenizer, and serving semantics.
2. The open-source inference-engine team gets a stable integration window for correctness tests, performance tuning, Docker builds, and recipe validation.
3. The community gets a public, bounded expectation instead of an ambiguous “coming soon.”

The separation is not a retreat from day-0 support. It is a more sustainable way to deliver day-0 against the artifact users will actually download. The page encourages more model vendors to follow.
