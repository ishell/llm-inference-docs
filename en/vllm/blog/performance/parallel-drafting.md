---
source: https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting
lang: en
fetched: 2026-09-04
---

# Parallel All the Way Down: Beyond Single-Token Generation with Speculative Decoding

Chinese: [zh/vllm/blog/performance/parallel-drafting.md](../../../../zh/vllm/blog/performance/parallel-drafting.md)

2026-07-28. **Alexandre Marques, Megan Flynn, Helen Zhao, Krishna Teja Chitty Venkata, Chibueze Ukachi (Red Hat AI)**. Study note. Speculators + vLLM support for three parallel drafters: [P-EAGLE](https://arxiv.org/abs/2602.01469), [DFlash](https://arxiv.org/abs/2602.06036), [DSpark](https://arxiv.org/abs/2607.05147). Checkpoints in the [Speculators Collection](https://huggingface.co/collections/RedHatAI/speculator-models).

**Errata (2026-07-29), also a section at the end:** the plots in Figure 1 were updated. Original numbers did not match the reported benchmarking conditions (bad environment). **Relative** ranking between models stayed consistent; conclusions in the post are unchanged. The markdown extract does **not** print TPS / ITL / OTPS for those curves — do not invent them from the PNGs.

P-EAGLE detail and B200 tables: [p-eagle.md](p-eagle.md). Adaptive DSpark verification budget: [dspark-adaptive.md](dspark-adaptive.md). Accept math still [spec-decode.md](spec-decode.md). Later EAGLE attention-drift fix: [eagle-3-1.md](eagle-3-1.md). Hidden export for training: [extract-hidden-states.md](../architecture/extract-hidden-states.md).

Verification is still **rejection sampling**. Speculative decoding keeps the verifier's output distribution exactly; quality is mathematically the same as standard decoding. This post changes **how the draft is grown**, not the accept math.

## 1. Introduction

Speculative decoding is a core serving optimization against memory-bandwidth limits. Validate several candidate tokens in one verifier forward; production systems get substantial speedups.

As serving infrastructure moved, classic speculative stacks hit a structural ceiling in **how draft tokens are generated**. This post is Speculators + vLLM open-source support for three parallel drafting algorithms: P-EAGLE, DFlash, DSpark.

![compare interactivity qwen38b math](../../../../assets/vllm/blog/performance/parallel-drafting/01-compare_interactivity_qwen38b_math.png)

![compare interactivity qwen330b humaneval](../../../../assets/vllm/blog/performance/parallel-drafting/02-compare_interactivity_qwen330b_humaneval.png)

![compare interactivity gemma431b humaneval](../../../../assets/vllm/blog/performance/parallel-drafting/03-compare_interactivity_gemma431b_humaneval.png)

**Figure 1.** Parallel drafting (P-EAGLE, DFlash, DSpark) vs autoregressive EAGLE-3 on three workloads. Checkpoints: [Speculators Collection](https://huggingface.co/collections/RedHatAI/speculator-models) on RedHatAI HuggingFace. **Use the 2026-07-29 updated plots**; see Errata. The post itself does not tabulate the curve values.

## 2. The limits of recursive drafting

[EAGLE](https://arxiv.org/abs/2401.15077) and [MTP](https://arxiv.org/abs/2404.19737) changed the game: the speculator reads the verifier's hidden states instead of guessing from surface text, and acceptance rates jumped.

[EAGLE-3](https://arxiv.org/abs/2503.01840) still **auto-regressive drafts**. A sequence of candidates means **one forward per draft token**.

Two production trade-offs:

- **Constraints on model size.** Drafting cost scales linearly with speculation length, so speculators stay tiny so they do not eat the time the verifier just saved.
- **Complex operational tuning.** Linear scaling caps K in practice. Optimal speculation length becomes a knob engineering has to retune per use case and live server load.

![ar vs parallel](../../../../assets/vllm/blog/performance/parallel-drafting/04-ar_vs_parallel.jpg)

**Figure 2.** Parallel drafting: several draft tokens in one step. Auto-regressive drafting: one draft token per step.

## 3. The shift to parallel drafting

Parallel drafting drops sequential execution from the draft phase. Predict a whole candidate **block** concurrently. Flatten drafting to one forward, and proposal latency is **decoupled** from how many tokens you speculate.

Two consequences the page names:

- **Capacity for expressiveness.** The speculator runs once per block, so you can use larger, deeper draft architectures. More context, higher acceptance, without a sequential latency tax.
- **Simplified parameter tuning.** Drafting cost no longer tracks block length, so you stop hyper-tuning K against fluctuating load.

The idea is older: [Medusa](https://arxiv.org/abs/2401.10774) and [PARD](https://arxiv.org/abs/2504.18583). P-EAGLE, DFlash, and DSpark add **deep verifier-state conditioning** — the insight that made EAGLE work — on top of parallel execution.

## 4. Under the hood: inference and training architecture

All three consume verifier hidden and emit draft tokens in parallel. They take different routes. Figure 3 is the side-by-side.

![diagram](../../../../assets/vllm/blog/performance/parallel-drafting/05-diagram.jpg)

**Figure 3.** P-EAGLE ingests verifier hidden as speculator **inputs**. DFlash **projects** hidden into the speculator **KV-cache**. DSpark keeps a DFlash backbone and adds sequential correction plus a confidence estimator.

Shared training problem: any parallel speculator does next-K prediction at **every** token position. Sequence length N, lookahead K: naively computing loss on the full matrix blows memory and compute. Each algorithm sparsifies differently.

### P-EAGLE

Builds on EAGLE: verifier hidden as input features. Instead of consuming them token-by-token, it maps them across several future positions and emits a whole candidate sequence in one parallel step.

Train-time trick in **this** post: **draft block sparsification** — drop tokens along the lookahead dimension K with a **decaying rate**, concentrate loss on the nearest tokens, prune distant future positions. (The dedicated [P-EAGLE post](p-eagle.md) describes a sequence-partition algorithm for long N; keep the two write-ups separate.)

### DFlash

Routes verifier features differently. Rather than standard inputs, it **projects** hidden and **injects them into the speculator KV-cache**. Attention is tightly conditioned on the verifier state **without** growing the input sequence. Candidate block via **block diffusion**.

Train-time: **sequence length sparsification**. Do not compute block loss at every position along N; pick random **anchor** points and compute block predictions only there. Saves GPU memory, keeps representative coverage. Same family of idea as [speculators-v050.md](speculators-v050.md).

### DSpark

DFlash parallel backbone plus two extras:

1. A lightweight **autoregressive correction head**, so future tokens condition more strongly on past tokens. Parallel throughput plus sequential coherence.
2. A **confidence head** that scores draft tokens **before** the verifier, and forwards only those likely to be accepted. Parallel drafting can propose many tokens cheaply; the verifier still pays for all of them. Confidence cuts wasted verify compute. The serving-side budget that uses this head: [dspark-adaptive.md](dspark-adaptive.md).

## 5. Inference performance

Figure 1 is the comparison vs EAGLE-3. Three model × algorithm pairs:

| Model | Algorithm | Use case | Hardware |
| --- | --- | --- | --- |
| Qwen3-8B | [P-EAGLE](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.peagle) | Math reasoning (GSM8k) | 1×A100 |
| Qwen3-30B-A3B | [DFlash](https://huggingface.co/RedHatAI/Qwen3-30B-A3B-speculator.dflash) | Coding (HumanEval) | 2×A100 |
| gemma-4-31B-it | [DSpark](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.dspark) | Coding (HumanEval) | 2×A100 |

The page's claim after the errata: in all three, parallel drafting **significantly** beats EAGLE-3; relative ranking held after the environment fix. Absolute numbers were **not** printed in the post body. Performance varies by model, task, and hardware — the authors ask you to bench your own workload.

## 6. Production serving with vLLM and Speculators

Speculators is the training/eval ecosystem, wired into vLLM. Launch a parallel-backed speculative engine by passing config at init. DFlash example from the page:

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 2 \
  --reasoning-parser qwen3 \
  --speculative-config '{
    "model": "RedHatAI/Qwen3-30B-A3B-speculator.dflash",
    "num_speculative_tokens": 7,
    "method": "dflash"
  }'
```

P-EAGLE in vLLM is `method: eagle3` plus `"parallel_drafting": true` (see [p-eagle.md](p-eagle.md)). DSpark method name on this page's sibling is `"dspark"` ([dspark-adaptive.md](dspark-adaptive.md)).

Block-level parallel drafting: the pipeline is parallel all the way down. Lossless vs standard decoding via rejection sampling.

## 7. Get started

The page calls parallel drafting fully supported, open-source, and production-ready at the time of writing. Train your own via documented pathways; bench natively in vLLM.

- Repo: [Speculators](http://github.com/vllm-project/speculators)
- Pre-trained: [Speculators Collection on HuggingFace](https://huggingface.co/collections/RedHatAI/speculator-models)
- Training guides: [Speculator tutorials](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/tutorials/index.md)

## Errata (2026-07-29)

Figure 1 plots were updated on **2026-07-29**. Original plot numbers were inconsistent with the reported benchmarking conditions because of an **erroneous environment setup**. Relative behavior between models was consistent; **conclusions in the blog are not changed**. No replacement numeric table was added in the errata text.
