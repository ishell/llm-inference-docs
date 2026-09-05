---
source: https://vllm.ai/blog/2026-05-11-vllm-tops-artificial-analysis
lang: en
fetched: 2026-09-04
---

# Artificial Analysis: three models, three bottlenecks; fusions and drafts in main

Chinese: [zh/vllm/blog/performance/artificial-analysis.md](../../../../zh/vllm/blog/performance/artificial-analysis.md)

2026-05-11. **vLLM Team**. Study note; May 2026 DigitalOcean / Artificial Analysis board, not your SLA. Sibling Pareto work: [gpt-oss-optimizations.md](gpt-oss-optimizations.md) / [qwen35-25k-tps.md](../serving/qwen35-25k-tps.md). V3.2 sparse path: [deepseek-v32.md](../architecture/deepseek-v32.md). Later reuse: [deepseek-v4.md](../architecture/deepseek-v4.md). Draft training: [eagle-3-1.md](eagle-3-1.md) / [speculators-v050.md](speculators-v050.md). MRV2: [mrv2.md](../architecture/mrv2.md). System TPS ≠ per-user TPS.

**TL;DR from the page:**

- DigitalOcean [published](https://www.digitalocean.com/blog/how-we-built-fastest-deepseek-minimax-qwen-on-blackwell-ultra) three frontier open-weight deployments on Blackwell Ultra. Engine underneath is open-source vLLM. Claim vs proprietary stacks: same silicon, first on the board.
- DeepSeek V3.2: low-batch **launch-bound**; attention path ~33 kernels → ~10, **1.28×** at bs=1 (85.8→109.3 tok/s, 4×GB200, no MTP). One 8×B300 cc=1: no MTP TP8 **125**; MTP=1 **234** (~90% accept); P/D TP4+TP4+MTP=3 **262**. Router GEMM ~**6%**; indexer TopK one graph, up to **17%** TPOT on 128K Decode.
- MiniMax-M2.5: TorchSpec EAGLE3 + `fuse_minimax_qk_norm`. Ceiling (synthetic 100% accept) TP4 **326 tok/s**.
- Qwen 3.5 397B: missed `allreduce_rms` spent ~half Decode on unfused cross-device reduce; then post-conv fusion + dual-stream. TEP=8 cc=1 **163 tok/s**, cc=256 **6.69→7.33 req/s**.
- Changes in vLLM `main` or in flight. Page headline: DeepSeek V3.2 best per-user output **230 TPS** (>4× most providers); Qwen 3.5 397B first of 12 providers, TTFT under 1 s on 10k-token prompts.

![hero](../../../../assets/vllm/blog/performance/artificial-analysis/01-hero_image.png)

*How vLLM built the leading deployments of DeepSeek V3.2, MiniMax-M2.5, and Qwen 3.5 397B.* (hero on the page)

## How vLLM made it fast

One bottleneck each:

1. **DeepSeek V3.2:** aggressive kernel fusion at low batch (also the foundation of [DeepSeek V4](../architecture/deepseek-v4.md)).
2. **MiniMax-M2.5:** targeted fusion + a custom EAGLE3 draft trained on open-source [TorchSpec](https://github.com/torchspec-project/TorchSpec) and vLLM. Same draft works on M2.7 (architectures identical).
3. **Qwen 3.5 397B:** fusions for the linear-attention and normalization path.

## DeepSeek V3.2: kernel fusion at low batch

At low batch, V3.2 was bound by **kernel launch**, not compute. Each layer issued dozens of tiny kernels (norm, RoPE, quant) that the GPU finished in microseconds; launch cost dominated.

Op fusion across the attention path: Q and KV norm, RoPE for Q and KV, indexer layer-norm + RoPE, FP8 quant, KV-cache writes — collapsed into a pair of fused kernels covering everything outside attention and MoE. Per-layer kernel count ~**33 → ~10**.

![DSv3.2 attention-path fusion](../../../../assets/vllm/blog/performance/artificial-analysis/02-figure1.png)

**Figure 1.** Attention-path fusion: ~33 launches → ~10. **1.28×** at batch size 1.

Fusion alone: **1.28×** at bs=1 (85.8 → 109.3 tok/s on 4× GB200, no MTP). One 8× B300 node at concurrency 1:

- Without MTP (TP=8): **125 tok/s**
- MTP=1 (TP=8): **234 tok/s** (~90% draft acceptance)
- Prefill/Decode disaggregation (TP=4 + TP=4 + MTP=3): **262 tok/s**

Two model-specific kernels after fusion:

- Router GEMM specialized for DSv3 MoE routing dims at small Decode batch — extra **6%** at batch 1 ([#34302](https://github.com/vllm-project/vllm/pull/34302)).
- Sparse-attention indexer TopK: pick algorithm per row from sequence length, all cases in **one CUDA graph**. Up to **17%** per-token latency on 128K-context Decode ([#37421](https://github.com/vllm-project/vllm/pull/37421)).

Same work now underpins DeepSeek V4 (reuses Q RoPE + quant and QK-norm fusions).

![DeepSeek V3.2 Non-Reasoning](../../../../assets/vllm/blog/performance/artificial-analysis/03-figure2.png)

**Figure 2.** DeepSeek V3.2 Non-Reasoning, output speed across providers. Source: [Artificial Analysis](https://artificialanalysis.ai/models/deepseek-v3-2/providers#output-speed), May 2026.

![DeepSeek V3.2 Reasoning](../../../../assets/vllm/blog/performance/artificial-analysis/04-figure3.png)

**Figure 3.** DeepSeek V3.2 Reasoning, output speed across providers. Source: [Artificial Analysis](https://artificialanalysis.ai/models/deepseek-v3-2-reasoning/providers#output-speed), May 2026.

## MiniMax-M2.5: EAGLE3 and more fusion

[Inferact](https://inferact.ai) trained a custom EAGLE3 draft with [TorchSpec](https://github.com/torchspec-project/TorchSpec): FSDP draft training concurrent with vLLM target inference. Draft eats live vLLM hidden states over MiniMax-M2.5-regenerated responses — match the base token distribution, not a generic SFT set.

MRV2 speculative plumbing that made this possible: draft metadata fix for later-position accept ([#38311](https://github.com/vllm-project/vllm/pull/38311)); CUDA graph for draft prefill ([#37588](https://github.com/vllm-project/vllm/pull/37588)).

Fusion: `fuse_minimax_qk_norm` for the model’s non-standard attention norm — Q and K variances reduced across TP ranks **before** the per-channel scale ([#37045](https://github.com/vllm-project/vllm/pull/37045)).

![fuse_minimax_qk_norm](../../../../assets/vllm/blog/performance/artificial-analysis/05-figure4.png)

**Figure 4.** Anatomy of `fuse_minimax_qk_norm` across four tensor-parallel ranks.

With that plus `fuse_norm_quant`, `fuse_act_quant`, `fuse_gemm_comms`, the **ceiling** experiment (synthetic 100% accept, isolates fusion from draft quality):

- **326 tok/s** at concurrency 1 (TP=4, EAGLE3 + 3 speculative tokens).

![MiniMax-M2.5 providers](../../../../assets/vllm/blog/performance/artificial-analysis/06-figure5.png)

**Figure 5.** MiniMax-M2.5, output speed across providers. Source: [Artificial Analysis](https://artificialanalysis.ai/models/minimax-m2-5/providers#output-speed), May 2026.

## Qwen 3.5 397B: linear attention and fusion gaps

Qwen 3.5 uses linear attention with a non-standard attention-block norm. Both fight vLLM’s stock fusion: post-projection convolution is unique to linear-attention models; the norm variant did not match `allreduce_rms`.

Profiler: missed `allreduce_rms` → roughly **half of Decode** on unfused cross-device reduces. Numbers were correct; extra HBM round-trips.

Four pieces:

- Fix `allreduce_rms` to recognize Qwen’s norm variant — ~**5%** TPOT at batch > 1.
- Kernel-level qk-norm + RoPE path.
- Post-conv fusion ([#37813](https://github.com/vllm-project/vllm/pull/37813)) for the linear-attention architecture.
- Dual-stream overlap of independent compute branches.

![Qwen 3.5 fusion](../../../../assets/vllm/blog/performance/artificial-analysis/07-figure6.png)

**Figure 6.** Qwen 3.5 397B kernel fusion work in vLLM.

With TP=8 + expert parallelism, production deployment:

- **163 tok/s** at concurrency 1 (TEP=8, post-conv fusion)
- **7.33 req/s** at concurrency 256, from **6.69 req/s** baseline (**+10%**)

Shipped in vLLM `main`.

![Qwen 3.5 providers](../../../../assets/vllm/blog/performance/artificial-analysis/08-figure7.png)

**Figure 7.** Qwen 3.5 397B, output speed across providers. Source: [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-5-397b-a17b/providers#output-speed), May 2026.

## What this means / open-source default

DSv3.2 attention-path fusions, MiniMax EAGLE3 training recipes, Qwen 3.5 fusions: upstream or on the way. Teams on current vLLM get the same speedups.

The page’s framing: historically the fastest inference stacks were proprietary. On these Artificial Analysis boards, the fastest inference they measured is open source.

## Acknowledgements (from the page)

Inferact, DigitalOcean, NVIDIA, Red Hat, and the vLLM open-source community.
