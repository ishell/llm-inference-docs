---
source: https://vllm.ai/blog/2026-07-27-k3
lang: en
fetched: 2026-09-04
---

# Kimi K3 Is Here: Efficient Day-0 Support on vLLM

Chinese: [zh/vllm/blog/serving/kimi-k3.md](../../../../zh/vllm/blog/serving/kimi-k3.md)

2026-07-27. **vLLM Team and Inferact**. Demo numbers on GB300 NVL72. Architecture / KDA prefix-cache design: [kimi-k3-preview.md](kimi-k3-preview.md). Tool-calling handshake cousin: [kimi-k2-accuracy.md](kimi-k2-accuracy.md). Speculative-decoding relatives: [../performance/dspark-adaptive.md](../performance/dspark-adaptive.md), [../performance/spec-decode.md](../performance/spec-decode.md). Cache pool cousins: [mooncake.md](mooncake.md), [kv-offload.md](kv-offload.md). Study note; Docker-only at launch (prerelease FlashInfer). **Not a new engine** — hybrid cache, kernels, recipes.

`moonshotai/Kimi-K3`: 2.8T MoE, 16 of 896 experts per token, 1M context, native vision, MXFP4 weights. Hybrid Kimi Delta Attention (KDA) + periodic full attention, Attention Residuals (AttnRes), Stable LatentMoE. Chat template is a Python renderer, not Jinja.

**Figure (social preview, not stored locally):** “Kimi K3 day-0 support on vLLM.”

Local figures (copyright remains with the original site; study copies):

![architecture](../../../../assets/vllm/blog/serving/kimi-k3/01-architecture.png)

![hybrid cache](../../../../assets/vllm/blog/serving/kimi-k3/02-hybrid-cache.png)

![dspark acceptance rates](../../../../assets/vllm/blog/serving/kimi-k3/03-dspark-acceptance-rates.png)

![dspark schematic](../../../../assets/vllm/blog/serving/kimi-k3/04-dspark-schematic.png)

![sequence parallelism](../../../../assets/vllm/blog/serving/kimi-k3/05-sequence-parallelism.jpg)

![pd disaggregation animation](../../../../assets/vllm/blog/serving/kimi-k3/06-pd-disaggregation-animation.gif)

![interval cache retention](../../../../assets/vllm/blog/serving/kimi-k3/07-interval-cache-retention.png)

![selective cache retention](../../../../assets/vllm/blog/serving/kimi-k3/08-selective-cache-retention.gif)

![kda decode](../../../../assets/vllm/blog/serving/kimi-k3/09-kda-decode.png)

![kda metadata builder](../../../../assets/vllm/blog/serving/kimi-k3/10-kda-metadata-builder.png)

![latent moe tail fusion](../../../../assets/vllm/blog/serving/kimi-k3/11-latent-moe-tail-fusion.png)

![serving performance](../../../../assets/vllm/blog/serving/kimi-k3/12-serving-performance.png)

![pareto gb300](../../../../assets/vllm/blog/serving/kimi-k3/13-pareto-gb300.png)

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

Easiest launch: **8 NVIDIA B300** or **8 AMD MI355X**. Inferact open-sourced a [DSpark speculator](https://huggingface.co/Inferact/Kimi-K3-DSpark):

```bash
--speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
```

Recipes and Docker: [recipes.vllm.ai/moonshotai/Kimi-K3](https://recipes.vllm.ai/moonshotai/Kimi-K3). **Only Docker images are usable** at this writing; they depend on several pre-release packages, including [FlashInfer](https://github.com/flashinfer-ai/flashinfer).

## TL;DR

- **2.8T multimodal MoE:** 16 of 896 experts per token, up to 1M tokens, KDA + AttnRes + LatentMoE, native MXFP4 (4-bit) weights.
- **Up to 370 tok/s per user:** 118 tok/s without speculative decoding and **370 tok/s (3.14×)** with DSpark on **16 NVIDIA GB300 NVL72** GPUs.
- **Production features at launch:** speculative decoding, Prefill/Decode disaggregation, agentic KV caching with Mooncake, tool calling, reasoning output, structured output. NVIDIA Hopper/Blackwell and AMD MI355X.
- **Open-source DSpark:** block-diffusion speculative decoding trained with vLLM and [TorchSpec](https://github.com/lightseekorg/TorchSpec), released by Inferact.
- **Hybrid prefix caching:** redesign over recurrent KDA state; now benefits every hybrid linear model.

## Kimi K3's architecture, and how vLLM serves it

**Figure.** Architecture innovations, from the [Kimi K3 release blog](https://www.kimi.com/blog/kimi-k3). Internals: [preview](kimi-k3-preview.md). This post is the practical guide.

### Kimi Delta Attention: a hybrid recurrent + full-attention stack

**What's new:** Most layers are KDA — linear attention with a **fixed-size recurrent state** instead of a growing KV cache — interleaved with **periodic full-attention** layers for exact global recall. That is what makes 1M-token context affordable.

**How vLLM serves it:** One hybrid KV-cache manager holds two memories under one scheduler: paged KV blocks for full-attention layers, compact recurrent-state blocks for KDA. Dedicated KDA backend: **FlashKDA** for Prefill; fused CUDA kernel (or Flash-Linear-Attention/Triton when running speculative decoding) for Decode.

Prefix caching is the hard part. Full-attention stores per-token KV; KDA updates recurrent and convolution state at every token and **cannot** snapshot at every prefix boundary. vLLM decouples large physical KDA state blocks from fine-grained prefix matching, registers snapshots inside those blocks, and **copies before extension** so long shared prompts reuse both KDA state and paged KV. Core vLLM, not a K3 fork.

**Figure.** Hybrid cache: KDA layers interleaved with periodic full attention; recurrent state and paged KV managed together.

### Attention Residuals: learned mixing of residual contributions across depth

**What's new:** Block AttnRes replaces ordinary residual accumulation with **depth-wise attention**: each Transformer sublayer uses a learned pseudo-query to weight RMS-normalized residual states from preceding layer blocks.

**How vLLM serves it:** Triton and CUDA kernels compute depth-wise attention logits, softmax, and hidden-state aggregation in **one fused op**. Residual updates and output RMSNorm fold into the same kernel where supported.

### Stable LatentMoE: quantile-balanced latent-space experts at 16-of-896 sparsity

**What's new:** [LatentMoE](https://research.nvidia.com/labs/nemotron/LatentMoE/) (NVIDIA) projects dispatched activations into a narrower latent dim for routed-expert compute, then projects back — cutting expert-weight bandwidth and all-to-all so more experts fit at similar cost. Kimi K3 [Stable LatentMoE](https://www.kimi.com/blog/kimi-k3) scales to **896 experts, 16 active**, with [Quantile Balancing](https://kexue.fm/archives/11619) instead of heuristic balancing updates.

**How vLLM serves it:** Experts sharded with expert parallelism. Two MoE backends: **TRT-LLM-Gen** for TP > 1; **MegaMoE** for disaggregated/expert-parallel (DEP). Optional Expert-Parallel Load Balancing (EPLB). Weights execute natively in **MXFP4**.

### Chat template: a render program, not a Jinja template

**What's new:** System / user / assistant, multimodal content, tool defs, and tool results must use exact control tokens. K2 used a [Jinja chat template](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/main/chat_template.jinja); K3 uses a **Python program** that builds the prompt **token sequence** directly. Output has distinct regions for reasoning, answer text, and tool calls.

**How vLLM serves it:** Input renderer and streaming output parser in Python and Rust frontends. User- and tool-supplied text treated as ordinary content. Tool calls and structured outputs integrate with [XGrammar](https://xgrammar.mlc.ai/) so structured regions are constrained during decoding and returned as separate reasoning, content, and tool-call fields.

## Built for production

### Ultra-low latency: speculative decoding with DSpark

DSpark from day 0; draft trained with vLLM + TorchSpec for numerical parity between speculator inference and training. Block-diffusion backbone drafts multiple tokens in one parallel pass from Kimi K3 intermediate states; drafting cost stays flat as the block deepens. Low-rank Markov head for intra-block dependency; confidence head predicts accept likelihood. Draft is **MLA-native**, mirroring K3 attention so KV layout stays compatible with advanced KV management and P/D.

**Figure.** DSpark positional acceptance rates across datasets.

SPEED Bench, single-user: **3.14×**, **118 tok/s → 370 tok/s**. Coding / low-entropy: ~**4.73** accepted tokens per step. High-entropy (creative writing): ~**2.61**. Confidence-based scheduling (use the confidence head to prioritize / prune drafts) is **ongoing**. Draft model and inference support are open source.

**Figure.** Lightweight DSpark draft proposes candidates; K3 verifies in one parallel pass.

### Sequence parallelism for TEP Prefill

**Figure.** Sequence parallelism shards token ownership; AttnRes applied per shard; one all-gather rebuilds the batch before the next layer’s QKV projection.

Prefill combines attention tensor parallelism with MoE expert parallelism (**TEP**). Versus pure TP: less communication, whole experts per rank, better expert GEMM shapes.

Naive TEP: **two all-reduces per layer** (after attention `o_proj` and after MoE) — every rank materializes the full batch and redundantly applies AttnRes. Sequence parallelism ([arXiv:2205.05198](https://arxiv.org/abs/2205.05198)): all-reduce after `o_proj` → **reduce-scatter**; AttnRes per shard; MoE all-to-all dispatch/combine; **one all-gather** before next QKV.

Two claimed advantages:

- **Cheaper communication in theory** (reduce-scatter + A2A dispatch + A2A combine + all-gather vs two all-reduces). NCCL reduce-scatter / all-gather are **not** optimized for Prefill message sizes, so they wrote custom kernels **1.7×–4.5×** faster than NCCL, especially at small-to-medium sizes.
- **Sharded AttnRes:** each rank keeps only its token shard. Matters because AttnRes turns the residual into persistent cross-layer state.

Enabled by default when using TP with MegaMoE, or TP + DP + EP. **No extra flags.**

### Large-scale serving: Prefill/Decode disaggregation

High-throughput: expert + data parallelism across nodes, P/D on separate replicas. One validated topology: **TEP8 Prefill → DEP16 Decode**, **NIXL** as KV transfer.

P/D is unforgiving for a hybrid model: recurrent KDA state, full-attention paged KV, and block tables must all arrive. NIXL treats the shared page as two logical views — token-level MLA cache and request-level KDA state (convolution + recurrent). Handshake exchanges MLA/KDA metadata, then **separate transfer descriptors**.

Under heterogeneous TP, hybrid allocator uses different block sizes for Prefill and Decode. NIXL tracks logical-to-physical mapping and **zeroes untransferred tails** so stale data does not leak through padding.

**Figure (GIF).** Prefill/Decode disaggregation flow.

### Reconciling partial block cache hits and KV cache offloading

Fine-grained prefix hits may end **inside** a physical block ([preview](kimi-k3-preview.md)). Offloading then: local GPU hit with a partial tail, then a **longer** prefix in an external store (Mooncake). Full-block hits extend cleanly; a partial tail can **overlap** the remote result.

Scheduler compares exact reusable token lengths from both tiers and takes the **longer** prefix. If remote wins, it releases the block reserved for the shorter local tail and reconciles **all cache groups** to the new length.

Built entirely on existing KV Connector APIs — `MooncakeStoreConnector`, `SimpleCPUOffloadConnector`, and others get multi-tier partial-prefix reuse without model-specific paths. RFC [issue #45702](https://github.com/vllm-project/vllm/issues/45702); PRs [#45939](https://github.com/vllm-project/vllm/pull/45939), [#46384](https://github.com/vllm-project/vllm/pull/46384), [#49502](https://github.com/vllm-project/vllm/pull/49502).

### Agentic serving: smarter cache retention policies

One KDA layer’s state is roughly the MLA cache for a few thousand tokens — large, but **does not grow** with sequence length. For agents at hundreds of thousands to 1M tokens that matters. Caching KDA at every token would exhaust even a distributed pool (each checkpoint ≫ one token’s MLA). Two policies:

#### Interval-based retention

Treat selected positions as checkpoints — e.g. one every **32K** tokens. **Prompt boundaries** are better: the next turn usually replays the previous prompt. vLLM retains those automatically.

`VLLM_PREFIX_CACHE_RETENTION_INTERVAL`: `0` disables periodic checkpoints and keeps **only prompt-end** states (multi-turn). Larger intervals trade recompute for lower cache use.

Introduced for DeepSeek V4 and hybrid SWA in [PR #43447](https://github.com/vllm-project/vllm/pull/43447); K3 / hybrid linear day-0 in [PR #45845](https://github.com/vllm-project/vllm/pull/45845).

**Figure.** MLA caches KV every block; KDA kept only at checkpoints — prompt ends (green) always, fixed-interval (orange) configurable.

#### Marconi-style selective retention

System prompts, repo snapshots, tool specs may be reused **without** aligning to a prompt boundary. [Marconi-style retention (MLSys ’25)](https://mlsys.org/virtual/2025/poster/3260): **cache on the second hit**. First sighting proves the prefix exists; second proves it is shared. One-off prefixes do not crowd the pool.

[PR #37898](https://github.com/vllm-project/vllm/pull/37898); K3 day-0 [PR #47782](https://github.com/vllm-project/vllm/pull/47782).

**Figure (GIF).** Request 1 keeps KDA only at its own prompt end (past the shared prefix) → request 2 gets a KV hit but KDA miss → that second sighting caches state at the prefix boundary → request 3 reuses it.

Together: interval checkpoints structural boundaries; Marconi learns which other prefixes are worth keeping.

## Performance optimizations

The full model barely fits a single DGX B300; **minimum 16 NVIDIA B200/GB200** on that generation. TP is good for interactivity but limits effective KV size; large-scale EP can bottleneck per-user output-token speed on the network. Many of these already appear in the [preview](kimi-k3-preview.md).

### Attention Residuals

Block AttnRes attends over up to **eight** cached block representations plus the current within-block residual — at most **nine** sources. Online-softmax like FlashAttention, but across **depth**, not sequence. One fused kernel (residual update in, optional RMSNorm out). Portable Triton; specialized CUDA on supported Blackwell.

### KDA Decode

**Figure.** Fused KDA Decode: causal convolution, recurrent update, and RMSNorm in one launch.

A KDA layer: input projections, causal 1D conv, QK norm, gate, recurrent update, output gated RMSNorm. On supported configs, post-projection Decode (conv through gated RMSNorm) is **one CUDA kernel**, in-place state updates. Triton fallback otherwise.

### KDA Prefill

Moonshot released [FlashKDA](https://github.com/MoonshotAI/FlashKDA) (CUTLASS). vLLM integrated it (GPU coverage, metadata dtypes, layouts, vendoring). [Shikhar Mishra](https://github.com/Itssshikhar) then published [Flash-Flash-KDA](https://github.com/Itssshikhar/Flash-Flash-KDA) for H100. Validated on GB300 NVL72 within a day; folded into the FlashKDA integration.

### KDA metadata builder

**Figure.** Nsight Systems traces before/after metadata-prep optimization.

DSpark bring-up: K3 first reused the generic GDN metadata builder, which prepared unused FLA metadata and assembled GPU metadata with small eager PyTorch ops. Dedicated K3 KDA builder prunes unused paths and fuses those sequences into Triton. Batch size 1: metadata-prep **870 µs → 34 µs (96%)**; end-to-end DSpark latency **−6%**.

### Low-latency BF16 GEMM

Small-batch latency path: replace generic BF16 GEMM on several linear projections with `skinnyGEMM` — skip shared-memory staging, load activations/weights into registers, CUDA Core FMA (avoid TMA / Tensor Core setup). Microbenchmarks: kernel **8%–100%**; e2e ~**10%** in small-batch settings.

### Low-latency MoE tail fusion

**Figure.** LatentMoE tail: two all-reduces + RMSNorm + latent up-projection + add → three kernels, better comm/compute overlap.

After LatentMoE, routed-expert activations need RMSNorm + up-projection before adding shared-expert output. Normal TP: two all-reduces (or one with concat) and a **replicated** up-projection.

Instead: **reduce-scatter** on shared experts; **all-reduce** on routed (they need to be normalized); column-parallel up-projection on the replicated routed activation; elementwise add onto already-sharded shared output; all-gather via broadcast. About **20%** latency cut in this step; ~**7%–8%** e2e.

## Quality and Performance Benchmarks

### Accuracy and correctness evaluation

Served OpenAI-compatible endpoint; exact configs in the recipes. Maximum reasoning-effort:

| Benchmark | Score |
| --- | ---: |
| GSM8K | 0.976 |
| GPQA-Diamond | 0.939 |
| OCRBench | 0.889 |
| MMMU Pro Vision | 0.818 |

**Caveat:** K3 thinks a lot. A low score is more often a **truncated** answer than a wrong one — raise reasoning effort, set `max_tokens` generously, check for cut-off generations first.

### Serving performance

**Figure.** Decode throughput at batch size 1 on GB300 NVL72, TP8 and TP16.

| Config | tok/s per user (bs=1) |
| --- | ---: |
| TP8, no spec | 111 |
| TP16, no spec | 118 |
| TP8 + DSpark | 331 (~3×) |
| TP16 + DSpark | 370 |

**Figure.** Initial Pareto on GB300 NVL72: high-throughput at **2K+ TPGS** through low-latency at **100+ TPS/user**.

### Reproduce our benchmark

TP8 + DSpark decode throughput:

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
```

Batch size 1, 8K/1K random (no speculative decoding):

```bash
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
```

Batch size 1, SPEED Bench (speculative decoding):

```bash
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

Multi-node, expert-parallel, and vision: [Kimi K3 recipes](https://recipes.vllm.ai/moonshotai/Kimi-K3).

## Important Deployment Tips

1. **Prefix caching:** `--enable-prefix-caching`. Usually on by default in vLLM; **currently off by default for K3** while hybrid-cache design evolves. Pass the flag.
2. **Tool calling:** Validate on your traffic. K3 has occasionally emitted a tool-call format its own parser does not expect → empty `tool_calls`, while clean probes on the same setup parse. Prompt- and run-dependent. Production agents: validate against schema, retry/fall back on empty `tool_calls`, consider strict/structured tool calling.
3. **All-to-all:** `--all2all-backend`. NVLink: `flashinfer_nvlink_one_sided`. RDMA: `deepep_v2`.
4. **MoE backend:** `deep_gemm_mega_moe` for any DEP environment. TP > 1: `flashinfer_trtllm` (FAQ).
5. **Rust frontend:** `VLLM_USE_RUST_FRONTEND=1` — fully supports this model.
6. **ViT parallelism:** `--mm-encoder-tp-mode=data` is **default**. Vision encoder `head_size=12` cannot shard evenly under TP=8. ViT < 1B params vs backbone ~2T, so ViT DP is on by default to avoid encoder all-reduce.

## Kimi K3 vLLM FAQ

### How many GPUs do I need to serve Kimi K3?

At least one **8× B300** (or GB300 NVL72) node; **16× B200** also supported. Most production: multi-node EP + DP over RDMA or NVLink.

### How do I enable DSpark speculative decoding?

The `--speculative-config` JSON above. Roughly triples single-stream Decode on reasoning and coding.

### Which MoE and all-to-all backend should I use?

`deep_gemm_mega_moe` for DEP; `flashinfer_trtllm` for TP > 1. All-to-all: `flashinfer_nvlink_one_sided` (NVLink), `deepep_v2` (RDMA).

### Does Kimi K3 support prefix caching, and is it on by default?

Yes over both full-attention KV and recurrent KDA state. **Not on by default** — pass `--enable-prefix-caching`.

### Does vLLM support Kimi K3 on AMD GPUs?

Yes. ROCm at launch; broader tuning on the roadmap.

### How is this different from the Kimi K3 preview post?

[Preview](kimi-k3-preview.md) = architecture and kernel deep dive (KDA prefix caching, kernels). This post = launch guide: recipes, flags, performance, what is production-ready.

## Roadmap and Future Work

- **RL:** rollout support already added; end-to-end RL training with ecosystem projects next.
- Continuous performance work after day 0.
- **Decode Context Parallelism (DCP):** prototype shows good speedup; early experiments **40% higher throughput than TP8** under selected workloads. Upstream soon.
- Better **EPLB**.
- **Confidence-based scheduling** using DSpark’s confidence head.
- Broader AMD ROCm tuning.

## Quick links

- Model: [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
- DSpark draft: [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark)
- Recipes / Docker: [recipes.vllm.ai/moonshotai/Kimi-K3](https://recipes.vllm.ai/moonshotai/Kimi-K3)
- Kimi K3 technical blog: [kimi.com/blog/kimi-k3](https://www.kimi.com/blog/kimi-k3)
- Design: [preview](kimi-k3-preview.md)

## Acknowledgements

Moonshot AI (architecture ahead of release, KDA-aware caching). Inferact (e2e integration and deployment validation). NVIDIA (fused KDA Decode, KDA Prefill, AttnRes kernels, MXFP4 MoE). AMD (ROCm bring-up). Inference partners including Alibaba Cloud, Baseten, DigitalOcean, Modal. Shikhar (Flash-Flash-KDA). vLLM community. The cache infrastructure now belongs to every hybrid model with a similar architecture.
