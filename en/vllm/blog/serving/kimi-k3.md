---
source: https://vllm.ai/blog/2026-07-27-k3
lang: en
fetched: 2026-09-05
---

# Kimi K3 Is Here: Efficient Day-0 Support on vLLM

Chinese: [zh/vllm/blog/serving/kimi-k3.md](../../../../zh/vllm/blog/serving/kimi-k3.md)

2026-07-27. vLLM Team and Inferact. Weights: [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3). DSpark: [`Inferact/Kimi-K3-DSpark`](https://huggingface.co/Inferact/Kimi-K3-DSpark). Recipes: [recipes.vllm.ai/moonshotai/Kimi-K3](https://recipes.vllm.ai/moonshotai/Kimi-K3). Preview: [kimi-k3-preview.md](kimi-k3-preview.md). Model post: [kimi.com/blog/kimi-k3](https://www.kimi.com/blog/kimi-k3). FlashKDA, Flash-Flash-KDA. Because of complicated dependencies, **only Docker images were usable at launch**; images depend on several pre-release dependencies, including [FlashInfer](https://github.com/flashinfer-ai/flashinfer). Social preview and page GIF/MP4 skipped. Local figures remain copyright of the original site.

Last week the [preview](https://vllm.ai/blog/2026-07-22-kimi-k3-preview) covered production-scale integration; today the weights are public and support is live. The most exciting challenge: making KDA, MXFP4 MoE, KV cache, P/D disaggregation, speculative decoding, and long-context recipes work together in a runnable serving engine. The preview explained kernels and cache, especially prefix caching on recurrent state. This post is the practical guide: how vLLM adapts to the architecture, the kernel work behind the numbers, and what is ready on day 0.

## Quick start

```bash
# See the linked recipes for the exact Docker command.
vllm serve moonshotai/Kimi-K3 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --load-format fastsafetensors \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3
```

The easiest path: 8 NVIDIA B300 or 8 AMD MI355X with the command above.

Inferact also trained and open-sourced a [DSpark speculator](https://huggingface.co/Inferact/Kimi-K3-DSpark). Add to the serve command:

```bash
--speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
```

Docker images for various platforms and deployment strategies: [recipes](https://recipes.vllm.ai/moonshotai/Kimi-K3). Complicated dependencies; **only Docker was usable then**.

## TL;DR

- **A 2.8T multimodal MoE:** 16 of 896 experts active per token; context up to 1M; KDA, AttnRes, LatentMoE, native MXFP4.
- **Up to 370 tok/s per user:** 118 tok/s without speculative decoding; 370 tok/s with DSpark (**3.14×**) on 16 NVIDIA GB300 NVL72 GPUs, from extensive architecture-specific optimizations.
- **Broad production features:** speculative decoding, P/D disaggregation, Mooncake agentic KV, tool calling, reasoning, structured output. Launch covers NVIDIA Hopper/Blackwell and AMD MI355X.
- **Open-source DSpark:** block-diffusion speculative decoding, trained with vLLM and TorchSpec, open-sourced by Inferact.
- **Hybrid prefix caching:** serving recurrent + full-attention required a redesign of hybrid prefix caching over recurrent KDA state. The change now benefits every similar hybrid linear model.

## Kimi K3's architecture, and how vLLM serves it

![Kimi K3 architecture](../../../../assets/vllm/blog/serving/kimi-k3/01-architecture.png)

_Architecture innovations from the [original release blog](https://www.kimi.com/blog/kimi-k3)._

K3 departs from a standard Transformer in a few ways; each one changes what a serving engine has to do. The preview covers internals; here the recap is what is new and how vLLM adapts.

### Kimi Delta Attention: a hybrid recurrent + full-attention stack

**What's new:** Most layers are KDA, a linear-attention mechanism that keeps a fixed-size recurrent state instead of a growing KV cache, interleaved with periodic full-attention layers that preserve exact global recall. That is what makes a 1M-token context affordable.

**How vLLM serves it:** One hybrid KV-cache manager holds two kinds of memory side by side under one scheduler: paged KV for full-attention layers, compact recurrent-state blocks for KDA. A dedicated KDA attention backend runs FlashKDA for prefill and a fused CUDA kernel (or the Flash-Linear-Attention/Triton path when running speculative decoding) for decode.

The hardest part is prefix caching across the hybrid cache: full-attention stores per-token KV; KDA updates recurrent and convolution state at every token but cannot afford a snapshot at every possible prefix boundary. vLLM decouples large physical KDA state blocks from fine-grained prefix matching, registering snapshots within those blocks and copying them before extension so long shared prompts can reuse both KDA state and paged KV. This hybrid-cache machinery is [new in vLLM core](https://vllm.ai/blog/2026-07-22-kimi-k3-preview) and now benefits every hybrid model similar to K3. Note: [kimi-k3-preview.md](kimi-k3-preview.md).

![hybrid KDA and full-attention cache](../../../../assets/vllm/blog/serving/kimi-k3/02-hybrid-cache.png)

_K3 interleaves KDA with periodic full-attention; vLLM’s hybrid cache manages recurrent state and paged KV together._

### Attention Residuals: learned mixing of residual contributions across depth

**What's new:** For each token, Block AttnRes replaces ordinary residual accumulation with depth-wise attention: every Transformer sublayer uses a learned pseudo-query to weight RMS-normalized residual states from preceding layer blocks, then receives the corresponding weighted combination as its input.

**How vLLM serves it:** Optimized Triton and CUDA kernels compute the depth-wise attention logits, softmax, and hidden-state aggregation in a single fused operation. Residual updates and output RMSNorm are folded into the same kernel where supported, reducing intermediate traffic and launch overhead in both prefill and decode.

### Stable LatentMoE: quantile-balanced latent-space experts at 16-of-896 sparsity

**What's new.** NVIDIA’s [LatentMoE](https://research.nvidia.com/labs/nemotron/LatentMoE/) projects dispatched token activations into a narrower latent dimension for routed-expert computation, then projects the combined expert output back to the model width — reducing expert-weight bandwidth and all-to-all so more experts can be used at similar inference cost. K3’s [Stable LatentMoE](https://www.kimi.com/blog/kimi-k3) scales this to 896 experts with 16 active per token and uses [Quantile Balancing](https://kexue.fm/archives/11619) to derive expert allocation from router-score quantiles instead of heuristic balancing updates.

**How vLLM serves it:** Experts are sharded with expert parallelism. Two MoE backends: TRT-LLM-Gen for TP > 1; MegaMoE for disaggregated/DEP. Optional Expert-Parallel Load Balancing (EPLB) so each rank has a similar amount of compute. Weights execute natively in MXFP4 on the MoE path.

### Chat template: a render program, not a Jinja template

**What's new:** K3’s chat template must encode system, user, and assistant messages, multimodal content, tool definitions, and tool results using exact control tokens. Instead of a [Jinja chat template](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/main/chat_template.jinja) that renders the request as text before tokenization, K3 uses a **Python program to build the prompt token sequence directly**. Its output likewise contains distinct regions for reasoning, answer text, and tool calls that must be parsed into an API response.

**How vLLM serves it:** Both the Python and Rust frontends implement the input renderer and streaming output parser, preserving control-token boundaries while treating user-supplied and tool-supplied text as ordinary content. For tool calls and structured outputs, vLLM integrates K3’s format with [XGrammar](https://xgrammar.mlc.ai/) so structured regions are constrained during decoding and returned as separate reasoning, content, and tool-call fields. K2’s Jinja handshake pitfalls: [kimi-k2-accuracy.md](kimi-k2-accuracy.md).

## Built for production

Serving a 2.8T hybrid MoE well means being fast for each user, efficient for many concurrent sessions, and scalable for agents. vLLM aims at all three on day 0.

### Ultra-low latency: speculative decoding with DSpark

Ultra-low latency on a 2.8T model without accuracy loss makes speculative decoding the natural choice. That is why vLLM supports DSpark from day 0 — and why Inferact trained and released a [DSpark speculator](https://huggingface.co/Inferact/Kimi-K3-DSpark). The draft is trained with vLLM using [TorchSpec](https://github.com/lightseekorg/TorchSpec) for numerical parity between speculator inference and training. Note: [dspark-adaptive.md](../features/dspark-adaptive.md).

DSpark uses a block-diffusion backbone to generate multiple speculative tokens in one parallel pass from K3’s rich intermediate states, so drafting cost stays flat as the block deepens. A low-rank Markov head supplies intra-block dependency; a confidence head predicts the likelihood each draft is accepted. The draft is MLA-native, mirroring K3’s own attention, so draft and target share a similar KV layout and stay compatible with advanced KV management and P/D-disaggregated setups.

![DSpark positional acceptance](../../../../assets/vllm/blog/serving/kimi-k3/03-dspark-acceptance-rates.png)

_Positional acceptance rates across datasets._

With DSpark: **3.14×** on a single-user request, 118 → 370 tok/s, measured with SPEED Bench. Coding and other low-entropy tasks: about **4.73** accepted tokens per step. High-entropy tasks such as creative writing: about **2.61**.

Confidence-based scheduling with DSpark was an ongoing effort. Once enabled, it uses the confidence head in the DSpark model to predict how likely each drafted token is to be accepted, prioritizing strong proposals and pruning weak ones so verification is not spent on tokens that will not survive.

Both the draft model and the inference support are open source as of this release. See the deployment guide below.

![DSpark draft-and-verify](../../../../assets/vllm/blog/serving/kimi-k3/04-dspark-schematic.png)

_A lightweight DSpark draft proposes candidates that K3 verifies in one parallel pass, accelerating single-stream decode._

### Sequence parallelism for TEP prefill

![sequence parallelism](../../../../assets/vllm/blog/serving/kimi-k3/05-sequence-parallelism.jpg)

_Sequence parallelism shards token ownership across ranks; the attention residual is applied per shard; one all-gather rebuilds the full batch before the next layer’s QKV projection._

For prefill, attention tensor parallelism is combined with MoE expert parallelism (TEP). Versus pure TP, TEP reduces communication and keeps whole experts on each rank, yielding more efficient expert GEMM shapes.

A naive TEP implementation performs two all-reduces per layer — one after the attention output projection and one after the MoE — so every rank materializes the full batch and redundantly applies the attention residual to all of it. [Sequence parallelism](https://arxiv.org/abs/2205.05198) replaces the all-reduce after `o_proj` with a reduce-scatter so each rank owns a shard of the tokens; the attention residual is applied per shard; the MoE’s all-to-all performs dispatch and combine; a single all-gather restores the full batch before the next layer’s QKV projection.

Two key advantages:

- **Reduced communication:** Reduce-scatter + all-to-all dispatch + combine + all-gather are theoretically cheaper than two all-reduces. In practice NCCL’s reduce-scatter and all-gather are not optimized for prefill message sizes and yield **no speedup**. Custom reduce-scatter and all-gather kernels are **1.7×–4.5×** faster than NCCL, especially at small-to-medium message sizes.
- **Sharded attention residual:** The residual stays sharded across ranks throughout the layer, so each rank computes and maintains only its shard. This matters especially for K3, where AttnRes turns the residual stream into persistent cross-layer state with its own compute and memory footprint.

Enabled by default when appropriate: TP with MegaMoE, or TP + DP + EP. **No extra flags.**

### Large-scale serving: prefill/decode disaggregation

For high throughput, vLLM serves K3 with expert and data parallelism across nodes and with P/D disaggregation — prefill-heavy and decode-heavy work on separate replicas, each sized for its own bottleneck. One validated topology: TEP8 prefill → DEP16 decode, NIXL as the KV transfer engine. Note: [mooncake.md](../features/mooncake.md).

PD disaggregation is unforgiving for a hybrid model: recurrent KDA state, full-attention paged KV, and block tables all have to arrive correctly. The NIXL connector treats the shared KV-cache page as two logical views: token-level MLA cache and request-level KDA state, including convolution and recurrent state. During the handshake it exchanges MLA/KDA metadata, then builds separate transfer descriptors for each transfer.

Under heterogeneous TP, the hybrid allocator uses different block sizes for prefill and decode. The NIXL connector tracks the logical-to-physical block mapping and **zeroes** any untransferred tail regions, preventing stale data from previous requests from leaking through padding or layout gaps.

Page P/D GIF not copied.

### Reconciling partial block cache hits and KV cache offloading

As in the preview, fine-grained prefix hits may end **inside** a physical cache block. That is subtle for KV offloading: vLLM may first find a local GPU hit with a partial tail, then discover a longer prefix in an external store such as Mooncake. With full-block hits, remote reuse extends cleanly beyond the local prefix. A partial tail, however, can **overlap** the remote result.

The scheduler therefore compares the **exact** reusable token lengths from both tiers and selects the longer prefix. If the remote hit wins, it releases the block reserved for the shorter local tail and reconciles all cache groups to the new prefix length.

This mechanism was built entirely through the existing KV Connector APIs, which already provide the required semantics. `MooncakeStoreConnector`, `SimpleCPUOffloadConnector`, and other connectors can support multi-tier partial-prefix reuse without model-specific paths. Note: [kv-offload.md](../features/kv-offload.md).

Design: [RFC #45702](https://github.com/vllm-project/vllm/issues/45702); implementation [PR #45939](https://github.com/vllm-project/vllm/pull/45939), [#46384](https://github.com/vllm-project/vllm/pull/46384), [#49502](https://github.com/vllm-project/vllm/pull/49502).

### Agentic serving: smarter cache retention policies

K3’s linear-attention layers need only a constant-size KDA state, so they are memory-efficient at long context. One layer’s KDA state is roughly equivalent to the MLA cache for a few thousand tokens. Although large, this state does not grow with sequence length, unlike a conventional KV cache. That distinction becomes significant for agentic workloads spanning hundreds of thousands to one million tokens.

The same design complicates prefix caching. KDA state is updated in place during decoding, so vLLM must **copy** the state at a selected prefix boundary before the next forward pass overwrites it. Caching at every token position would be prohibitively expensive: each KDA checkpoint is much larger than one token’s MLA cache and would quickly exhaust even a distributed cache pool.

To improve cache-space efficiency while preserving useful prefixes, vLLM supports two complementary retention policies.

#### Interval-based retention

Caching every KDA state is wasteful; caching too sparsely forces the next request to recompute a large suffix. Interval-based retention treats selected positions as checkpoints — for example, one every 32K tokens.

Prompt boundaries are even better checkpoints. In agentic workloads the next turn usually begins by replaying the previous turn’s prompt, so the state at the end of that prompt is especially likely to be reused. vLLM detects and retains these boundaries automatically.

Periodic checkpointing: `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`. Setting it to `0` disables periodic checkpoints and retains only prompt-end states — a good fit when multi-turn conversations dominate. Larger intervals trade some recomputation for lower cache usage.

Interval-based retention was introduced for DeepSeek V4 and hybrid sliding-window models in [PR #43447](https://github.com/vllm-project/vllm/pull/43447); day-0 K3 and hybrid linear-attention support in [PR #45845](https://github.com/vllm-project/vllm/pull/45845).

![interval-based KDA retention](../../../../assets/vllm/blog/serving/kimi-k3/07-interval-cache-retention.png)

_MLA caches KV for every block; a KDA state is kept only at checkpoints: prompt ends (green) always retained, fixed-interval checkpoints (orange) configurable._

#### Marconi-style selective retention

Prompt-end retention works well for conversational state, but valuable shared prefixes can appear elsewhere. A system prompt, repository snapshot, or tool specification may be reused across many requests without aligning with a prompt boundary.

[Marconi-style retention (MLSys '25)](https://mlsys.org/virtual/2025/poster/3260) uses a simple rule: **cache on the second hit**. The first observation shows the prefix exists; the second shows it is actually shared. Only then does vLLM spend cache capacity on its KDA state.

Retention becomes demand-driven. One-off prefixes do not crowd the cache; recurring prefixes are promoted automatically — users need not predict which parts of the workload will become hot.

Selective retention: [PR #37898](https://github.com/vllm-project/vllm/pull/37898); day-0 K3 in [PR #47782](https://github.com/vllm-project/vllm/pull/47782). Page selective GIF not copied: request 1 keeps a KDA state only at its own prompt end, past the shared prefix, so request 2 gets a KV hit but a KDA miss; that second sighting caches a state at the prefix boundary, and request 3 reuses it.

Together the policies cover predictable and emergent reuse: interval retention checkpoints structurally important boundaries; Marconi-style retention learns which other prefixes are worth keeping.

## Performance optimizations

Serving a model this size has its own challenges. The entire model can barely fit in a single NVIDIA DGX B300 and needs a minimum of **16 NVIDIA B200/GB200** GPUs on that generation. Serving must trade interactivity against total system throughput: tensor parallelism is good for interactivity but offers low overall throughput because effective KV cache size is limited; large-scale expert parallelism can limit per-user output-token speed because of network-bandwidth bottlenecks. The optimizations below improve both sides so users can choose the recipe that fits. Many are already in the [preview](https://vllm.ai/blog/2026-07-22-kimi-k3-preview).

### Attention Residuals

Block AttnRes attends over up to **eight** cached block representations plus the current within-block residual. For each token, vLLM computes logits from RMS-normalized sources, applies softmax across these depth-wise candidates, and aggregates their representations. The implementation resembles FlashAttention’s online-softmax but operates across **model depth** rather than sequence positions and has at most nine sources. Mixing happens in a single fused kernel, incorporating the residual update at the input and optionally applying RMSNorm to the output. A portable Triton path covers the general case; a specialized CUDA kernel accelerates supported Blackwell configurations.

### KDA decode

![fused KDA decode](../../../../assets/vllm/blog/serving/kimi-k3/09-kda-decode.png)

_The fused KDA decode kernel folds causal convolution, recurrent update, and RMSNorm into one launch._

A KDA layer involves many operations: input projections, causal 1D convolutions, QK norm, gate computation, KDA recurrent update, and output gated RMSNorm. On supported configurations, vLLM fuses the post-projection decode path — from the causal convolutions through gated RMSNorm — into a single specialized CUDA kernel. The kernel updates convolution and recurrent states in place and writes the normalized output directly, avoiding intermediate tensors, repeated state traffic, and per-operation launch overhead across K3’s many KDA layers. Portable Triton fallbacks cover unsupported configurations.

### KDA prefill

KDA prefill became a favorite example of open-source development in practice. Moonshot first released [FlashKDA](https://github.com/MoonshotAI/FlashKDA), a high-performance CUTLASS implementation. It was quickly integrated into vLLM, then the less glamorous production details: broader GPU coverage, metadata dtypes, tensor layouts, reliable vendoring. [Shikhar Mishra](https://github.com/Itssshikhar) then optimized the kernels for H100 and published [Flash-Flash-KDA](https://github.com/Itssshikhar/Flash-Flash-KDA), improving data movement while preserving numerical correctness. Within a day the improvements were validated on GB300 NVL72, the recurrence pipeline and synchronization refined, and the work folded into the FlashKDA integration. Not a one-way handoff: an open kernel extended by the serving community, improved by an independent contributor, and quickly carried into production.

### KDA metadata builder

![KDA metadata before/after](../../../../assets/vllm/blog/serving/kimi-k3/10-kda-metadata-builder.png)

During DSpark bring-up, KDA metadata preparation emerged as significant overhead. K3 initially reused the generic GDN metadata builder, which prepared FLA metadata K3 does not consume and used sequences of small eager PyTorch operations to assemble and stage GPU metadata. A dedicated Kimi K3 KDA metadata builder prunes unused paths and replaces those sequences with fused Triton kernels, reducing each sequence to a single launch. At batch size 1: metadata-preparation latency **96%**, **870 µs → 34 µs**; end-to-end DSpark latency **−6%**.

### Low-latency BF16 GEMM

In low-batch-size, latency-sensitive settings, generic BF16 GEMM in several linear projection layers is replaced with `skinnyGEMM`. Generic cuBLAS kernels are optimized for more general shapes and are not best here. The kernel bypasses shared-memory staging, loads activations and weights directly into registers, and uses CUDA Core FMA for the math. That avoids the heavy TMA and Tensor Core setup used for maximum throughput. Microbenchmarks: kernel-level **8%–100%**; about **10%** end-to-end latency reduction in small-batch settings.

### Low-latency MoE tail fusion

![LatentMoE tail fusion](../../../../assets/vllm/blog/serving/kimi-k3/11-latent-moe-tail-fusion.png)

_The LatentMoE tail optimization replaces two all-reduces, RMSNorm, latent up-projection, and an elementwise add with three kernels to reduce compute and better overlap communication and computation._

vLLM uses a novel strategy to reduce latent-MoE tail latency in ultra-low-latency serving. At the end of LatentMoE, the reduced activation from routed experts must be normalized with RMSNorm and up-projected before it is added to the shared-expert output. In the normal TP case this requires two all-reduces on the routed and shared experts — or one all-reduce with concatenation — and replicates the up-projection.

To avoid redundant compute in the replicated linear projection, vLLM performs reduce-scatter on the shared experts and keeps all-reduce on the routed experts because their activations need to be normalized. The replicated routed-expert activation then performs matrix multiplication with the up-projection in a column-parallel fashion and is added elementwise to the already-sharded shared-expert output. Finally the results are all-gathered onto each rank using broadcast. About **20%** latency reduction in this step and about **7%–8%** end-to-end.

## Quality and Performance Benchmarks

### Accuracy and correctness evaluation

vLLM takes accuracy as seriously as speed. K3 was validated end to end through a served OpenAI-compatible endpoint, with exact configurations in the recipes, and it passes the accuracy evaluations cleanly. At maximum reasoning-effort: GSM8K **0.976**, GPQA-Diamond **0.939**, OCRBench **0.889**, MMMU Pro Vision **0.818**.

One evaluation caveat: K3 thinks a lot before it answers. A low score is more often a truncated answer than a wrong one, so increase reasoning effort, set `max_tokens` generously, and check for cut-off generations before debugging anything else.

### Serving performance

![single-user decode](../../../../assets/vllm/blog/serving/kimi-k3/12-serving-performance.png)

_Decode throughput at batch size 1 on GB300 NVL72, TP8 and TP16._

At launch: 111 tok/s per user on TP8 and 118 on TP16 at batch size 1. DSpark boosts interactivity by roughly **3×**: 331 tok/s per user on TP8 and 370 on TP16.

![GB300 NVL72 Pareto](../../../../assets/vllm/blog/serving/kimi-k3/13-pareto-gb300.png)

Initial Pareto-frontier results on GB300 NVL72, from high-throughput serving at 2K+ TPGS to low-latency serving at 100+ TPS/user.

### Reproduce our benchmark

Full recipes for the decode throughput numbers above, TP8 with DSpark:

```bash
export NCCL_DMABUF_ENABLE=0
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_USE_RUST_FRONTEND=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HEAD_ADDR=127.0.0.1  # Change if vllm-bench runs on another host.

vllm serve moonshotai/Kimi-K3 \
  --enable-prefix-caching \
  --tensor-parallel-size 8 \
  --nnodes 2 \
  --node-rank 0 \
  --moe-backend auto \
  --trust-remote-code \
  --load-format fastsafetensors \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.9 \
  --max-model-len auto \
  --max-cudagraph-capture-size 256 \
  --kv-cache-dtype fp8 \
  --attention-config '{"mla_prefill_backend":"FLASHINFER","use_prefill_query_quantization":true}' \
  --speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'

# Batch size = 1, 8K/1K random (no speculative decoding)
vllm-bench \
  --backend openai \
  --base-url "http://${HEAD_ADDR}:8000" \
  --model moonshotai/Kimi-K3 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --random-range-ratio 0.8 \
  --prompt-token-ids \
  --ignore-eos \
  --sweep-max-concurrency 1 \
  --sweep-num-prompts-factor 10 \
  --seed 42 \
  --percentile-metrics "ttft,tpot,itl,e2el" \
  --metric-percentiles "50,90,99" \
  --save-result

# Batch size = 1, SPEED Bench (speculative decoding)
vllm-bench \
  --backend openai \
  --base-url "http://${HEAD_ADDR}:8000" \
  --model moonshotai/Kimi-K3 \
  --dataset-name speed-bench \
  --speed-bench-config throughput_16k \
  --speed-bench-max-input-len 10240 \
  --speed-bench-category low_entropy \
  --output-len 1536 \
  --num-prompts 10 \
  --no-oversample \
  --max-concurrency 1 \
  --temperature 1.0 \
  --top-p 0.95 \
  --save-result \
  --save-detailed
```

Full recipes, including multi-node, expert-parallel, and vision: [Kimi K3 recipes](https://recipes.vllm.ai/moonshotai/Kimi-K3).

## Important Deployment Tips

1. **Prefix caching:** `--enable-prefix-caching` turns it on. Prefix caching is typically on by default in vLLM, but it is **currently disabled by default for Kimi K3** while the hybrid-cache design continues to evolve. Pass the flag explicitly.
2. **Tool calling:** Validate on your own traffic before depending on it. K3 has occasionally emitted a tool-call format its own parser does not expect, yielding an empty `tool_calls` result, while clean probes on the same setup parse perfectly. Prompt- and run-dependent, not a blanket failure. Production agents should validate against your schema, retry or fall back when `tool_calls` comes back empty, and consider strict or structured tool calling, which constrains the output grammar during generation.
3. **All-to-all backend:** `--all2all-backend` determines how the MoE backend communicates during expert parallelism. Use `flashinfer_nvlink_one_sided` for NVIDIA NVLink and `deepep_v2` for RDMA.
4. **MoE backend:** Several backends for different scenarios. Recommend `deep_gemm_mega_moe` for any DEP environment.
5. **Rust frontend:** `VLLM_USE_RUST_FRONTEND=1`. Fully supports this model.
6. **ViT parallelism:** `--mm-encoder-tp-mode=data` is enabled by default. K3’s vision encoder has `head_size=12`, which cannot be sharded evenly under TP=8. The vision encoder has fewer than 1B parameters while the backbone has about 2T, so ViT DP is on by default to avoid all-reduce overhead from the encoder.

## Kimi K3 vLLM FAQ

### How many GPUs do I need to serve Kimi K3?

At least one 8× B300 (or GB300 NVL72) node; 16× B200 is also supported. Most production deployments run multi-node with expert and data parallelism, over RDMA or NVLink.

### How do I enable DSpark speculative decoding?

Add:

```bash
--speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
```

It roughly triples single-stream decode on reasoning and coding workloads.

### Which MoE and all-to-all backend should I use?

Use `deep_gemm_mega_moe` for disaggregated or DEP deployments and `flashinfer_trtllm` for TP > 1. Match all-to-all to the interconnect: `flashinfer_nvlink_one_sided` for NVLink, `deepep_v2` for RDMA.

### Does Kimi K3 support prefix caching, and is it on by default?

It supports prefix caching over both full-attention KV and recurrent KDA state, but it is **not enabled by default**, so pass `--enable-prefix-caching`.

### Does vLLM support Kimi K3 on AMD GPUs?

Yes. ROCm support ships at launch, with broader tuning on the roadmap.

### How is this different from the Kimi K3 preview post?

The [preview](https://vllm.ai/blog/2026-07-22-kimi-k3-preview) is the architecture and kernel deep dive, including how KDA prefix caching and the kernels are built. This post is the practical launch guide and the artifacts: how vLLM adapts, recipes, flags, performance, and what K3 is ready for in production. Note: [kimi-k3-preview.md](kimi-k3-preview.md).

## Roadmap and Future Work

- **RL support for Kimi K3:** vLLM rollout support has already been added. Work with RL ecosystem projects for end-to-end RL training.
- **Continuous performance improvement** after day 0.
- **Decode Context Parallelism (DCP):** prototype shows good speedup; upstream soon. Early experiments: **40%** higher throughput than TP8 under selected workloads. Note: [dcp.md](../features/dcp.md).
- **Expert-Parallel Load Balancing (EPLB):** improve EPLB performance.
- **Confidence-based scheduling:** use the DSpark confidence head to prune draft tokens to verify.
- **Broader AMD ROCm tuning.**

## Quick links

- **Model:** [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
- **DSpark draft:** [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark)
- **Recipes and Docker:** [recipes.vllm.ai/moonshotai/Kimi-K3](https://recipes.vllm.ai/moonshotai/Kimi-K3)
- **Kimi K3 technical blog:** [kimi.com/blog/kimi-k3](https://www.kimi.com/blog/kimi-k3)
- **vLLM design for K3:** [the preview post](https://vllm.ai/blog/2026-07-22-kimi-k3-preview)

## Acknowledgements

Thanks to Moonshot for creating K3, sharing the architecture ahead of release, and co-designing KDA-aware caching; to Inferact for end-to-end vLLM integration and deployment validation; to NVIDIA for fused KDA decode, KDA prefill, and Attention Residual kernels and the MXFP4 MoE collaboration; to AMD for ROCm bring-up; to inference partners including Alibaba Cloud, Baseten, DigitalOcean, and Modal; to Shikhar for Flash-Flash-KDA; and to the vLLM community. The cache infrastructure built for Kimi K3 now belongs to every hybrid model with a similar architecture. The page cannot wait to see what you serve.
