---
source: https://vllm.ai/blog/2026-07-13-eagle-3-amd-instinct
lang: en
fetched: 2026-09-04
---

# EAGLE3 on Instinct: Quark MXFP4, then a served acceptance length

Chinese: [zh/vllm/blog/performance/eagle3-amd.md](../../../../zh/vllm/blog/performance/eagle3-amd.md)

2026-07-13. **Larry Li, Chao Li, Haichen Zhang, Chun Fang, Andy Luo, Spandan Tiwari, and Ashish Sirasao** (AMD Quark). Study note; MI355X / InferenceX benches, not your SLA. Accept math: [spec-decode.md](spec-decode.md). CUDA-side EAGLE: [p-eagle.md](p-eagle.md) / [eagle-3-1.md](eagle-3-1.md). Later five-path ROCm survey: [spec-decode-amd.md](spec-decode-amd.md). Hidden-state export: [extract-hidden-states.md](../architecture/extract-hidden-states.md). ROCm attention: [rocm-attention.md](../architecture/rocm-attention.md). Draft families: [parallel-drafting.md](parallel-drafting.md).

**TL;DR from the page:**

- Three pieces on Instinct: train an EAGLE3 draft with vLLM in the loop, AMD Quark MXFP4/FP8 for target and draft, serve on ROCm/vLLM. Benchmarked with InferenceX on **MI355X**.
- Kimi-K2.5 1K/1K: **1.69–2.00×** vs matching no-spec baselines. MiniMax-M2.5: **1.38–1.79×**. MiniMax-M3 draft they trained: SPEED-Bench average acceptance length **2.80**.
- Draft precision need not match verify — quant hits **draft bandwidth**. BF16 and FP8 Kimi sweeps use **different vLLM builds and MML**; not a controlled precision comparison.
- Random-prompt sweeps are throughput microbenchmarks. Re-measure acceptance length on your model.

## Why speculative decoding and EAGLE3

Prefill can be fast; Decode still walks one target-model step per token. For MoE / attention-heavy targets (Kimi-K2.5, MiniMax-M2.5) that sequential loop caps serving TPS.

Speculative decoding is lossless relative to the target: a lighter draft proposes several tokens; the target verifies them in one pass. Greedy: matching prefix is accepted. Sampling: accept or correct from target vs draft probabilities. At the first rejection the verifier emits a correction and drafting resumes; if all drafts land, the verifier emits one **bonus** token.

**Conditional acceptance rate** = P(accept this draft position | earlier positions accepted). **Acceptance length** = tokens emitted per verification cycle. Higher AL can cut target steps; realized TPS still pays drafting + verification overhead.

![greedy speculative decoding](../../../../assets/vllm/blog/performance/eagle3-amd/01-figure1.png)

**Figure 1.** Greedy speculative decoding with γ=5: target accepts an α=3-token prefix, rejects the first mismatch, discards later drafts, emits a correction → α+1=4 tokens. If all γ drafts are accepted, the extra token is a target-generated bonus.

EAGLE → EAGLE2 → EAGLE3: feature-level drafting, then better accept, then multi-layer features from the target (low / mid / high) plus training-time testing. Not an unrelated small LM. Production point: more generation TPS, same target output behavior after verify.

Other families named on the page: small drafts, MTP, Medusa, DFlash, DSpark.

## AMD Quark MXFP4

MXFP4 = OCP Microscaling 4-bit float: small blocks share a scale. Memory near INT4, better numerics. Instinct **MI350X / MI355X** have native FP4 matrix acceleration — MXFP4 weights map onto the silicon and ease MoE Decode bandwidth / capacity.

[AMD Quark](https://quark.docs.amd.com/latest/) ships **day-0** MXFP4 (and FP8) checkpoints when a major model lands, ready on ROCm/vLLM. Examples: [amd/Kimi-K2.5-MXFP4](https://huggingface.co/amd/Kimi-K2.5-MXFP4), [amd/MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4). Those checkpoints are the **target** for both EAGLE3 draft training and speculative serving. Execution path: supported MXFP4 + **AITER MoE** kernels. Speculative decoding does not change the target distribution: every draft token is verified.

## Training EAGLE3 drafts with vLLM

A high-acceptance draft is a systems problem. vLLM sits in the training loop, not only at serve. Running example: MiniMax-M3 EAGLE3 trained by the Quark team on Instinct. The Kimi-K2.5 and MiniMax-M2.5 drafts in the inference charts are **community HF drafts**, not trained here.

![vLLM-centric EAGLE3 pipeline](../../../../assets/vllm/blog/performance/eagle3-amd/02-figure2.png)

**Figure 2.** One vLLM-on-ROCm runtime: synthesize on-policy data (Stage 1), stream low/mid/high hidden states (Stage 2), FSDP2 cold-start of a single-layer EAGLE3 head (Stage 3), in-loop serve-eval by measured acceptance length (Stage 4), export and EAGLE3 serve (Stage 5).

1. **On-policy synthesis.** Stand up the Quark MXFP4 target as vLLM-ROCm. Chat via `/v1/chat/completions` with the exact serving template; raw `/v1/completions` (template-bypassed) for non-chat / OOD robustness. Same engine and template as later serve.

2. **Hidden-state extraction.** Draft conditions on target internals (low / mid / high + `fc_norm`), not an unrelated small model. Three modes: **online** (target co-located), **offline** (dump to disk), **streaming** (live serve → trainer, no dump). Streaming is what makes a **420B MXFP4 MoE** target practical on one node. Same door as [extract-hidden-states.md](../architecture/extract-hidden-states.md).

3. **Cold-start FSDP2.** Single-layer EAGLE3 head from scratch; TTT loss and position-decay under FSDP2. Verifier is the Quark MXFP4 target — draft learns the activation space it will see at deploy.

4. **Serve-eval in the loop.** Training loss overstates real accept. Periodically export, serve under vLLM speculative decoding, pick the checkpoint by **served acceptance length**.

5. **Export.** Hugging Face format, vLLM-ready draft directory, ROCm EAGLE speculative decoding.

### SPEED-Bench: MiniMax-M3 draft

Acceptance length (AL) = mean tokens emitted per target verification step. AL = 1 is one emitted token per verify, before drafting overhead.

| Domain | AL |
| --- | ---: |
| Coding | 3.32 |
| Math | 3.14 |
| RAG | 3.12 |
| Multilingual | 3.04 |
| Reasoning | 2.89 |
| STEM | 2.86 |
| Summarization | 2.86 |
| Humanities | 2.71 |
| QA | 2.55 |
| Writing | 2.33 |
| Roleplay | 2.01 |
| **Average** | **2.80** |

Strongest on structured / technical domains; still AL **2.01–2.33** on writing / roleplay. Prompt 1K → 32K: AL essentially flat (**2.69 → 2.65**). At three speculative tokens, first / second / third positions accepted about **76% / 56% / 43%** (cumulative).

![MiniMax-M3 AL vs context](../../../../assets/vllm/blog/performance/eagle3-amd/03-figure3.png)

**Figure 3.** MiniMax-M3 EAGLE3 acceptance length vs input length on SPEED-Bench. Dashed AL=1 is one emitted token per verification cycle.

Draft: [amd/MiniMax-M3-EAGLE3.1](https://huggingface.co/amd/MiniMax-M3-EAGLE3.1) against [amd/MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4):

```bash
export VLLM_ROCM_USE_AITER=1
vllm serve amd/MiniMax-M3-MXFP4 --trust-remote-code --tensor-parallel-size 8 \
--block-size 128 --attention-backend TRITON_ATTN --moe-backend emulation \
--speculative-config '{"method":"eagle3","model":"amd/MiniMax-M3-EAGLE3.1","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}'
```

## End-to-end stack they claim

- **Target:** day-0 MXFP4/FP8 + ROCm/vLLM.
- **Draft:** EAGLE3 training (this work for M3), FP8/MXFP4 with Quark, ROCm/vLLM.
- **Glue:** on-policy data, hidden extract, serve-eval, export, speculative serve — all through vLLM.

EAGLE3 draft-training support on Instinct is **planned for the next AMD Quark release**; quantization workflows are in the released toolkit.

## Acceleration results

Only **1K/1K** (ISL=1024, OSL=1024) in this section. Speedup = EAGLE3 TPS / matching no-spec TPS from the **same** vLLM build and MML. MML = `--max-model-len` (prompt + generated). These are random-prompt throughput microbenchmarks, not application benches.

### Kimi-K2.5: BF16 and Quark FP8 drafts

Hardware: MI355X, TP=4, random prompts, `num_prompts=10 × concurrency`, `num_warmups=2 × concurrency`, 10 seeds per cell (arithmetic mean).

| Path | Docker | MML |
| --- | --- | ---: |
| BF16 | `vllm/vllm-openai-rocm:v0.19.0` | 2248 |
| FP8 | `vllm/vllm-openai-rocm:nightly-fb1ac806c55a6dc96fe92261b80c8550e9c39d2f` | 2304 |

Target: [amd/Kimi-K2.5-MXFP4](https://huggingface.co/amd/Kimi-K2.5-MXFP4). BF16 draft: [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3). FP8 draft: [amd/kimi-k2.5-eagle3-fp8](https://huggingface.co/amd/kimi-k2.5-eagle3-fp8) (Quark FP8 workflow; shares the target’s BF16 LM head). FP8 draft dispatches `RowWiseTorchFP8ScaledMMLinearKernel` (`torch._scaled_mm` / hipBLASLt row-wise scaled FP8 GEMM), **not** AITER preshuffled FP8.

![Kimi-K2.5 EAGLE3 throughput](../../../../assets/vllm/blog/performance/eagle3-amd/04-figure4.png)

**Figure 4.** Kimi-K2.5 output tok/s/GPU at 1K/1K on MI355X TP=4. BF16 **1.69–1.90×**, Quark FP8 **1.76–2.00×** vs each path’s own no-spec baseline. Largest relative gain at low concurrency. Builds and MML differ — not a precision bake-off.

### MiniMax-M2.5 BF16 EAGLE3

Image: `vllm/vllm-openai-rocm:nightly-4eafc729285e459a5fc96efd6f7b313b155cad48`. Target: [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5). Draft: [thoughtworks/MiniMax-M2.5-Eagle3](https://huggingface.co/thoughtworks/MiniMax-M2.5-Eagle3), BF16, `num_speculative_tokens=3`, `draft_tensor_parallel_size=1`. 1K/1K random, TP=4 + expert parallelism, five seeds (mean). Matching no-spec baseline from the same build.

![MiniMax-M2.5 EAGLE3 throughput](../../../../assets/vllm/blog/performance/eagle3-amd/05-figure5.png)

**Figure 5.** MiniMax-M2.5 output tok/s/GPU at 1K/1K on MI355X TP=4. Largest relative gain at low concurrency.

Sweep summary: Kimi-K2.5 **1.69–2.00×**, MiniMax-M2.5 **1.38–1.79×**.

## Acknowledgements / resources (from the page)

AMD Quark, ROCm and vLLM contributors, InferenceX maintainers, EAGLE3 research community. Special thanks: Chang Liu, Xinjun Niu, Wei Luo, Lin Zhao.

- [EAGLE3 project](https://github.com/SafeAILab/EAGLE) / [paper](https://arxiv.org/abs/2503.01840)
- [SPEED-Bench](https://arxiv.org/abs/2604.09557)
- [InferenceX](https://github.com/SemiAnalysisAI/InferenceX)
- [AMD Quark](https://github.com/amd/Quark) / [vLLM](https://github.com/vllm-project/vllm)
