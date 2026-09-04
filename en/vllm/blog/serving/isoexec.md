---
source: https://vllm.ai/blog/2026-08-21-isoexec
lang: en
fetched: 2026-09-04
---

# IsoExec: one execution contract for trainer and engine

Chinese: [zh/vllm/blog/serving/isoexec.md](../../../../zh/vllm/blog/serving/isoexec.md)

2026-08-21. Alexander Jiang and the SkyRL Team. Repo: [zanderjiang/SkyRL-IsoExec](https://github.com/zanderjiang/SkyRL-IsoExec). Earlier “two model copies, matched kernels”: [bitwise-rl.md](bitwise-rl.md). Pause / weight APIs that this loop still sits on: [native-rl.md](native-rl.md). Study note.

SkyRL × vLLM × Megatron. One **8×H100** node, synchronous **Qwen3.5-35B-A3B** DAPO: mean rollout-versus-training logprob gap on contract-covered regions **below 1e-6**, **25%** wall-clock versus the SkyRL baseline then (50 steps).

## TL;DR

On-policy RL assumes rollout and training evaluate the **same** policy. In practice the two engines differ: model definitions, kernels, batch shapes, parallelism layouts. Floating-point arithmetic is non-associative, so those differences change token probabilities even when the “same” policy is running. New algorithms, harness/environment changes, and kernel/hardware work all become hard to debug.

**IsoExec** is a cross-framework unified execution abstraction. Two pieces:

1. An **execution contract** that specifies and enforces the rounding-sensitive details across engines.
2. A **unified model** with aligned, batch-invariant kernels that are bitwise consistent across training and rollout.

Implemented in SkyRL with vLLM and Megatron. On one 8×H100, synchronous Qwen3.5-35B-A3B DAPO, average end-to-end rollout-versus-training logprob difference **below \(10^{-6}\)** with **25%** overhead vs the then-current SkyRL baseline over **50** steps.

Contributions named on the page:

- **Unified execution contract:** one numerical contract across training and inference; zero contract-covered mismatch, low debug cost when algorithms, environments, or kernels change.
- **Parallelism-invariant kernels:** numerics preserved across tensor, expert, and sequence parallelism.
- **Chunkwise-parallel recurrent (CPR) Gated DeltaNet:** align training, Prefill, and recurrent Decode without serializing long-sequence forwards.

## Introduction

RL executes the same policy twice: the rollout engine samples a token under \(\mu\); the trainer later recomputes its log probability under \(\pi\) with the same parameters. Synchronous on-policy training assumes \(\mu = \pi\). Systems make that hard because

\[
(a+b)+c \neq a+(b+c).
\]

Stacks mix inference engines (vLLM, SGLang) with trainers (Megatron, FSDP). Different kernels, batch shapes, execution modes (training, Prefill, Decode), and distributed layouts → different reduction orders → different token distributions.

ByteDance [VeXact](https://arxiv.org/abs/2605.14220): mismatch alone can destabilize REINFORCE and GRPO, distort advantage-weighted loss before a KL estimator reacts, and make IS / rejection fixes calibration-sensitive. [Fireworks](https://fireworks.ai/blog/frontier-lab-training-infrastructure-as-a-service): a GLM-5.2 run with train–inference KL around **0.013** clipped about **45%** of tokens and reward collapsed around **step 20**; a bitwise-aligned run had **zero** clipped tokens and stayed stable.

Prior slices of the problem:

- [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) — batch invariance: neither other batch elements nor batch size should change one element’s computation.
- [vLLM × TorchTitan](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference) ([bitwise-rl.md](bitwise-rl.md)) — import matched kernels into both engines; still **two aligned model copies**.
- [Zero Train–Inference Mismatch for Linear Attention and Async RL](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/) — one model definition between TorchTitan and vLLM; parity for [Gated DeltaNet](https://arxiv.org/pdf/2412.06464) by using the recurrent form for all forwards and keeping the chunked kernel for backward.
- [Tree-Based Invariant Kernels (TBIK)](https://arxiv.org/abs/2511.17826) — bitwise consistency under different parallelism configs.

IsoExec: an execution contract plus a unified model. The contract captures rounding-sensitive choices (kernel, accumulation dtype, reduction order) in a framework-independent form and enforces them in each runtime. The unified model uses kernels validated bitwise-consistent across training, Prefill, and Decode, and still plugs into vLLM’s scheduler / KV manager / CUDA graph capture and Megatron’s training stack.

## Unified execution contract

The contract declares every bit-relevant execution choice both runtimes must specify identically.

```jsonc
"ExecutionContract": {
  "cases": [ ... ],        // logprob computations: trainer_fwd, engine_decode, etc.
  "composition": [ ... ],  // (region, case) -> implementation + pinned constants
  "claims": { ... },       // topology invariance, state invalidation, tolerances
  "identities": { ... }    // semantic / numerical_policy / deployment digests
}
```

Each token-logprob computation is a **case** (rollout `engine_prefill`, trainer `trainer_fwd`, …). Forward operators are partitioned into **regions**: spans of arithmetic implemented by one kernel, possibly fused. For every `(region, case)` pair, **composition** picks the implementation and the constants it is pinned to — accumulation / boundary dtypes, split-K and split-KV partition counts, anything that can change bits. A region is tested for bitwise exactness across cases **before** its implementations may be registered.

Example:

```jsonc
"composition": [
  {
    "region": ["gdn.core", "gdn.gating", "norms.l2"],
    "cases": ["trainer_fwd", "trainer_fwd_no_autograd", "engine_prefill", "engine_decode"],
    "impl": {"id": "native_fused_sigmoid", "version": 1, "arch": "sm90"}
  },
  {
    "region": ["moe.combine"],
    "cases": ["engine_prefill", "engine_decode"],       // the trainer side is its own entry
    "impl": {"id": "pik_leaf_tree", "version": 2, "arch": "sm90"},
    "constants": {"leaves": 8, "leaf_dtype": "fp32"},
    "discharge": {"kind": "equivalence_proof", "ref": "gates/ep_invariant_combine"} // proved equivalence
  }
]
```

**Claims** are the conditions under which those guarantees hold, enforced at runtime. A topology claim lists the parallel sizes for which a reduction tree is proven bitwise invariant. Installing a kernel, the adapter compares the runtime’s actual parallel size with that list and **rejects an unproven size**.

**Identities** are SHA-256 digests of the serialized contract, used to verify trainer ↔ rollout agreement:

- `semantic` — same logical model.
- `numerical_policy` — every execution choice that can affect numerics (implementations, versions).
- `deployment` — settings proven **not** to affect bits (memory sizing, transport). The contract does **not** require these to match.

Matching `semantic` and `numerical_policy`, plus adapter enforcement, is the page’s criterion that both sides run the same verified numerical policy on covered regions.

Local figures (copyright remains with the original site; study copies):

![unified execution abstraction](../../../../assets/vllm/blog/serving/isoexec/01-unified_execution_abstraction.png)

IsoExec’s unified execution contract across training and inference runtimes.

A per-runtime **contract adapter** installs the contract and implementations and enforces them: binds each composition entry to the framework’s extension points (e.g. which attention kernel), then monitors installed kernels, declared claims, and cross-process identity digests.

## Unified model

SkyRL’s unified model: batch-invariant GEMM, attention, and normalization, plus deterministic MoE routing and combine. Bitwise-consistent across tensor / expert / sequence-parallel layouts, and a chunkwise-parallel recurrent algorithm for GDN hybrids.

Experiments applied the abstraction to **zero contract-covered mismatch** on dense (**MiMo-7B**), MLA MoE (**GLM-4.7-Flash**), hybrid (**Qwen3.5-9B**), and hybrid MoE (**Qwen3.5-35B-A3B**). Code: [SkyRL-IsoExec](https://github.com/zanderjiang/SkyRL-IsoExec).

### Parallelism-invariant kernels

Trainer and engine want different layouts. The trainer must fit optimizer state, activations, gradients, and (for MoE) distributed expert weights. The rollout engine wants KV capacity without hurting Decode latency.

For forward numerics with fixed inputs and weights, six axes:

- **DP** — batch partitioning; batch-invariant kernels keep each sample’s numerics.
- **PP** — whole layers move across devices; reductions inside a layer are unsplit when boundary dtypes are fixed.
- **TP** — splits contraction reductions across ranks. **Changes bits** unless the tree is fixed.
- **EP** — distributes expert compute and changes how expert outputs combine. **Changes bits.**
- **SP** — row-parallel reductions go from all-reduce to reduce-scatter. **Changes bits.**
- **CP** — splits the attention reduction across the sequence dimension. **Changes bits.** (Invariance here is a next step, not claimed.)

[TBIK](https://arxiv.org/abs/2511.17826) fixed a global reduction tree across row-parallel GEMMs and the cross-GPU reduction for TP-invariant inference. IsoExec keeps the fixed-tree idea but applies it along **K**. `pik` divides K into \(G\) contiguous **leaves**. Each leaf uses deterministic Tensor Core MMA with **FP32** accumulation. The contract pins the rank-to-leaf mapping and the binary arithmetic schedule; **NCCL moves partials** — no custom communication kernels.

![pik figure](../../../../assets/vllm/blog/serving/isoexec/02-pik_figure.png)

The fixed binary reduction tree `pik` uses to keep numerics across parallelism layouts.

Same principle for EP and SP:

- **EP:** combine expert outputs in a **fixed routing order**, not rank order.
- **SP:** reuse the non-SP reduction tree; each rank keeps its own output slice instead of gathering the full result. Trainer logits are bitwise identical with SP on or off.

### Chunkwise-parallel recurrent (CPR) GDN

Linear attention is harder: training and inference use different algorithms. Existing GDN stacks use a **chunkwise-parallel** form for training and Prefill, and a **recurrent** form for Decode. Mathematically the same; rounding is not. Comparing [FLA](https://github.com/fla-org/flash-linear-attention)’s chunkwise-parallel kernel to vLLM’s fused recurrent kernel: mean per-element absolute difference about **\(1.7 \times 10^{-2}\)** , max **0.25**.

The [TorchTitan write-up](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/) ran the recurrent form for rollout Prefill **and** the trainer forward, chunkwise only for backward. That removes GDN mismatch and serializes Prefill / trainer forward in sequence length: about **2–3×** slower on math workloads and about **5×** on a terminal-agent workload — not a full-job plan. Recurrent-everywhere on their table is **4×+** Prefill.

**CPR** keeps recurrence as the main function but evaluates it in parallel across chunks:

- Training / Prefill: like chunkwise-parallel, a first pass computes the recurrent state at every chunk boundary; a parallel recurrent scan then fills outputs inside each chunk.
- Decode: recurrent form, but **resync the hidden state every \(C\) decoded tokens** (\(C\) = chunk size), so Prefill, training, and Decode share a rounding schedule.

Per-layer cost on **H100**, \(C=64\). Trainer and rollout engine use their production TP layouts and kernels. Ratios vs the native mixed implementation for that stage (smaller is better):

| Stage | Shape | Native mixed | Chunkwise everywhere | Recurrent everywhere | CPR |
| --- | --- | --- | --- | --- | --- |
| Bitwise exact | — | No | Yes | Yes | **Yes** |
| Trainer forward + backward | 1 × 10,240 tokens | 5.177 ms | 5.177 ms (1.00×) | 22.863 ms (4.42×) | **7.386 ms (1.43×)** |
| Rollout-engine Prefill | 5 × 2,048 tokens | 0.844 ms | 0.844 ms (1.00×) | 3.639 ms (4.31×) | **1.412 ms (1.67×)** |
| Rollout-engine Decode | 256 sequences × 1 token | 0.0612 ms | 2.2374 ms (36.6×) | 0.0612 ms (1.00×) | **0.0846 ms (1.38×)** |

## Results

IsoExec vs SkyRL’s native stack on one **8×H100**, **Qwen3.5-35B-A3B** on DAPO-Math-17k, **synchronous** RL. Same setup otherwise. IsoExec: **25%** end-to-end overhead vs the native SkyRL stack (vLLM + Megatron) under the highest-throughput synchronous-RL configuration they evaluated.

![result logprob diff](../../../../assets/vllm/blog/serving/isoexec/03-result_logprob_diff.png)

Rollout-versus-training absolute logprob differences: native SkyRL vs IsoExec.

Across **50** steps, mean pre-update rollout-versus-training absolute logprob difference:

| | Native | IsoExec |
| --- | ---: | ---: |
| Mean | \(1.648 \times 10^{-2}\) | \(6.744 \times 10^{-7}\) |
| Std | \(4.035 \times 10^{-2}\) | \(6.821 \times 10^{-7}\) |
| Average per-step max | 5.073 | \(7.358 \times 10^{-6}\) |

Mean on covered regions is **below 1e-6**.

![result time](../../../../assets/vllm/blog/serving/isoexec/04-result_time.png)

Average RL step timing over those 50 steps.

| Metric | Native | IsoExec | Overhead |
| --- | ---: | ---: | ---: |
| Generation | 591.3 s | 776.6 s | **31.3%** |
| Policy training | 498.6 s | 591.3 s | **18.6%** |
| Full RL step | 1224.6 s | 1534.0 s | **25.3%** |

Headline **25%** is that full-step **25.3%**.

![result reward](../../../../assets/vllm/blog/serving/isoexec/05-result_reward.png)

Pass@16 and raw reward over 50 steps. No meaningful reward lift from killing contract-covered mismatch on this short run — too short to see a stability dividend.

## Next steps

- **Blackwell support**
- **Context parallelism invariance**
- **Sparse attention**
- **Block-FP8 MoE**

## Acknowledgements

[Alexander Jiang](https://www.linkedin.com/in/akj2) and the SkyRL team. Thanks to [Charlie Ruan](https://www.charlieruan.com/), [Sumanth Hegde](https://sumanthrh.com/about/), [Eric Tang](https://erictang000.github.io/), [Philipp Moritz](https://www.linkedin.com/in/philipp-moritz-61419682), [Yichuan Wang](https://yichuan-w.github.io/), [Mayank Mishra](https://www.mayank.site/), and [Lingxiao Ma](https://xysmlx.github.io/) for discussions.
