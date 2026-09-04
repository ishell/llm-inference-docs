---
source: https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus
lang: en
fetched: 2026-09-04
---

# Speculative decoding on AMD GPUs: five draft paths

Chinese: [zh/vllm/blog/performance/spec-decode-amd.md](../../../../zh/vllm/blog/performance/spec-decode-amd.md)

2026-08-23. **AMD and Embedded LLM** (acknowledgements on the page). Study note; benches on **MI300X / MI355X**, ROCm, not your SLA. Snapshot in the disclaimer: vLLM `0.23.1rc1.dev1120+g0f0f28b53`, ROCm/HIP `7.2.53211`. Original page: https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus

Accept math is still [spec-decode.md](spec-decode.md). How the draft is grown: [parallel-drafting.md](parallel-drafting.md) (P-EAGLE / DFlash / DSpark) and [p-eagle.md](p-eagle.md). DSpark confidence budget (not on in these AMD runs): [dspark-adaptive.md](dspark-adaptive.md). Later EAGLE attention-drift fix: [eagle-3-1.md](eagle-3-1.md). Training hidden export: [extract-hidden-states.md](../architecture/extract-hidden-states.md). ROCm attention backends on the same GPUs: [rocm-attention.md](../architecture/rocm-attention.md).

**TL;DR from the page:** speculative decoding lets vLLM verify several drafted tokens in one target-model pass. Output-token throughput moved with drafting method and proposal length `N`, and also with model family, draft checkpoint, workload, and acceptance. Upper end of the measured range: **2.87×** DFlash on `gemma-4-26B-A4B-it`, **2.83×** Gemma 4 MTP on the same target, **2.68×** DFlash on Kimi-K2.5. Some sweeps were near-flat or below the non-speculative baseline. This is **how to turn the five methods on and measure them on ROCm**, not new accept math.

## Introduction

Serving still starts from standard autoregressive decoding: generate one token, append it, generate the next. Simple, and the loop advances one committed token at a time because outputs are left-to-right.

Speculative decoding [[1]](#ref-1) keeps that output behavior and splits the loop into **draft** and **verify**. A lightweight draft component proposes candidate future tokens; the original model, as **target**, checks them before they are committed. When several drafts survive, several output tokens come out of one target verification step.

This post reviews the autoregressive baseline and the draft-and-verify process, then five drafting approaches that differ in what they take from the target and whether candidates are sequential, autoregressive, parallel, or hybrid: **Native MTP**, **Gemma 4 MTP**, **EAGLE-3**, **DFlash**, **DSpark**. Then CLI, pretrained drafts, measurements on Instinct **MI300X** and **MI355X**, tuning, and a short training sketch.

## The autoregressive decoding baseline

Each Decode step produces and commits one new token. Four output tokens → four sequential Decode steps:

1. `context` → model → T1
2. `context + T1` → model → T2
3. `context + T1 T2` → model → T3
4. `context + T1 T2 T3` → model → T4

The generated token is appended and becomes input for the next step. Long generations make this token-by-token loop dominate latency and cap serving TPS.

The question behind speculative decoding: can we keep the original model's output behavior while reducing how often generation advances by only one token at a time?

Proposal is separated from verification. A draft component proposes several candidates; the target verifies them before commit.

## Core idea of speculative decoding

The original model is not replaced. It stays the target and still owns the final output. A faster proposal stage sits in front.

- **Draft:** propose several candidate future tokens. They are not committed yet.
- **Verify:** the target evaluates that candidate sequence in **one** verification pass.

Verification is left-to-right. Each draft token is checked at its position. Accepted tokens are committed. The first rejection stops the rest of that proposal: later candidates are discarded, and the target provides the token at the rejected position. Generation continues from the updated sequence.

One round: draft proposes T1 T2 T3 T4; target accepts T1 and T2, rejects T3, discards T4; commit is T1, T2, and a **replacement token** from the target. Several accepts → several output tokens from one verification. A reject still lets the target decide the next token. Lossless relative to the target.

![figure 01](../../../../assets/vllm/blog/performance/spec-decode-amd/01-figure-01.svg)

**Figure 1.** Draft proposes candidate future tokens; the target verifies them before output tokens are committed.

### A simple accept/reject example

Green boxes in Figure 2 are draft tokens that survive verification. Red is the first rejected draft. Gray is a later draft that is discarded. Blue in the output comes from the target, not the draft.

![figure 02](../../../../assets/vllm/blog/performance/spec-decode-amd/02-figure-02.svg)

**Figure 2.** Left-to-right verification. First two drafts accepted; rejected position uses a target token; remaining candidate discarded.

Prompt: `The weather today is`. Draft proposes: `sunny`, `and`, `warm`, `outside`.

| | pos 1 | pos 2 | pos 3 | pos 4 |
| --- | --- | --- | --- | --- |
| draft proposes | sunny | and | warm | outside |
| model verifies | ✓ | ✓ | ✗ | stop |
| commit | sunny | and | **clear** (target) | — |

`sunny` and `and` are accepted. At the third position the draft said `warm`; the target selected `clear`. `outside` is discarded because it follows the first rejection. The next round starts from `The weather today is sunny and clear`.

## How the drafting methods work

Same draft-and-verify shell; different draft components. Main differences: what information arrives from the target; how it is folded into drafting; sequential vs parallel candidates.

Three buckets (architecture of the **draft**, not the target family). A target can have native MTP **and** separately trained EAGLE-3 / DFlash / DSpark drafts.

- **Native MTP modules.** Built into the target; model-native auxiliary prediction path; sequential candidates.
- **Separate MTP drafters.** Separate checkpoint paired with a specific target; target activations and shared KV during inference; sequential candidates. Gemma 4 MTP lives here.
- **Dedicated target-conditioned draft networks.** Speculators trained for a target: EAGLE-3 (autoregressive from target hidden), DFlash (parallel blocks from target hidden), DSpark (DFlash backbone + light causal correction + confidence prefix).

Depending on the method, the draft may see: a hidden from the target; hidden from several selected layers; the target KV cache; or a fusion of several representations.

### Native MTP

Multi-Token Prediction: model-native mechanisms that predict beyond the immediate next token. In vLLM, native MTP is available when the target includes a compatible auxiliary prediction component [[2]](#ref-2). Architecture varies by family; all of them give an auxiliary path for future tokens.

First speculative step: MTP combines a target hidden with information from the current token → first draft token. Later steps: newly drafted token + hidden from the previous MTP step → next candidate. After `N` candidates, the target verifies them together.

Common fusion: target (or previous MTP) hidden **plus** the embedding of a shifted input token or the latest draft token → fusion / projection → auxiliary prediction layer → draft logits. Hidden carries the preceding sequence; embedding names the latest token. Combined along the hidden dimension.

`num_speculative_tokens` and **physical** MTP depth are not the same. When `N` exceeds the checkpoint’s prediction depth, vLLM reuses the MTP path with extra forwards. Larger `N` → more candidates and more sequential drafting.

Native MTP is tied to the target architecture. Shared components (often) keep extra memory modest. Multiple speculative tokens still draft sequentially before verification.

### Gemma 4 MTP

Gemma 4 ships a **separately packaged** MTP draft paired with a specific target [[3]](#ref-3). Own checkpoint, still tightly connected at inference.

The draft uses activations the target already produced and **shares the target KV cache**, so it does not re-process the accepted prefix on its own.

Layer count in the draft is still separate from configured `N`. Several candidates → sequential generation through the paired MTP component, then one target verification.

### EAGLE-3

Dedicated draft network trained for a specific target [[4]](#ref-4). Own execution path, conditioned on target internals.

During the target forward, EAGLE-3 records hidden from **three** Transformer stages: near the beginning, around the middle, near the end. Concatenate + project → one fused target feature. That fused vector is combined with the embedding of the sampled token, then concatenated/projected into the EAGLE-3 draft decoder.

- Fused target feature: accepted sequence, seen at several stages of the target forward.
- Sampled-token embedding: the token from which drafting continues.

EAGLE-3 drafts **autoregressively**. First draft token: fused feature + sampled-token embedding. After a draft token is produced, its embedding feeds the next drafting stage. Later speculative positions have no target hidden yet (the target has not processed them), so EAGLE-3 uses the previous **draft-component output** to continue.

Later draft tokens depend on earlier drafted tokens along the proposal. More speculative tokens → more sequential drafting before verification. Attention-drift at longer `N` is the [eagle-3-1.md](eagle-3-1.md) story; this AMD post uses `method: eagle3` as shipped.

### DFlash

Dedicated draft network, but unlike MTP and EAGLE-3 it predicts a **whole block of future positions in parallel** [[5]](#ref-5). Family write-up: [parallel-drafting.md](parallel-drafting.md).

Each draft block starts with an **anchor**: a known token produced or confirmed by the target. DFlash does not predict the anchor; it is the starting point for the masked positions that follow. In later rounds this is typically the extra target token returned by the previous verification.

Example block of length 7: position 0 is `anchor`; positions 1–6 are `mask`. One DFlash forward predicts all masked positions together: output is `anchor, draft1 … draft6`.

Like EAGLE-3, DFlash first fuses hidden from several target layers (concatenate + projection → fused target context). The use is different. EAGLE-3 concatenates that fusion with the sampled-token embedding at the **input** of an autoregressive draft net. DFlash converts fused target context into extra **Key and Value** representations available in **every** layer of the draft network. Queries from the masked draft positions attend to both target-derived K/V and K/V from the draft block itself. Target context stays available throughout the draft net, not only at the input.

After the block is generated, the target verifies all proposed tokens in one pass. Left-to-right: accept until the first rejection; remaining candidates discarded; target token replaces the first rejected draft. All masked positions are predicted together in **one** draft-network forward (`draft1 … draft4` together, not `draft1 → draft2 → …`). A later position is **not** conditioned on the sampled output of an earlier position in the same pass — later-position quality then depends on checkpoint and workload, especially at longer blocks.

### DSpark

DSpark extends parallel drafting with two extra mechanisms [[6]](#ref-6):

1. A lightweight sequential head that introduces dependence between tokens **inside** the draft block.
2. Confidence-based selection of the prefix submitted for target verification.

Backbone: a modified DFlash model. One parallel forward produces a hidden state and base logits for every draft position, with the same target-context conditioning as DFlash.

A fully parallel drafter never sees tokens selected at earlier positions in the same block. When several continuations are plausible, combinations can be inconsistent: both “of course” and “no problem” may be reasonable, but independent position-wise predictions could emit “of problem.”

After the backbone, a lightweight **Markov head** selects left-to-right. For each position `k` it uses the immediately preceding selected token to produce a small bias added to the backbone’s base logits → adjusted distribution for that position. The heavy draft net still runs once; only the Markov head walks the block.

The design also includes a **confidence head** that can shorten the prefix sent to the target. **That feature was not active in the vLLM path used for these experiments**, so the numbers below are parallel backbone + Markov correction only. Adaptive verification budget on a later CUDA path: [dspark-adaptive.md](dspark-adaptive.md).

Target still verifies the proposed sequence in one pass; commit left-to-right until the first rejection.

### Summary of the drafting methods

Figure 3 is the side-by-side: draft component shape, which target information it uses, sequential vs parallel. In all five, the target still verifies once; acceptance is left-to-right until the first rejected draft.

![figure method summary](../../../../assets/vllm/blog/performance/spec-decode-amd/03-figure-method-summary.svg)

**Figure 3.** Draft structure and token-generation patterns for the five methods.

| Method | Draft component | Target-model information | How draft tokens are generated |
| --- | --- | --- | --- |
| Native MTP | Model-native auxiliary MTP path | Target or previous-MTP hidden + current draft-token info | Sequentially, reusing the MTP path |
| Gemma 4 MTP | Separate MTP checkpoint paired with the target | Target activations + shared target KV | Sequentially through the paired MTP component |
| EAGLE-3 | Dedicated autoregressive draft network | Early / mid / late target hidden, fused | Sequentially; each drafted token influences the next |
| DFlash | Dedicated parallel draft network | Fused target hidden as extra K/V in every draft layer | All candidate positions in one parallel forward |
| DSpark | DFlash-style parallel net + lightweight Markov head | Same target-conditioned info as the parallel net | One parallel forward, then light sequential adjustment |

## How to enable speculative decoding in vLLM

Configured through `--speculative-config`. Differences: method name, whether a separate draft checkpoint is required, and `num_speculative_tokens`. Current vLLM method values named on the page: `mtp`, `eagle3`, `dflash`, `dspark`.

| Method | Separate draft checkpoint | Typical configuration |
| --- | --- | --- |
| Native MTP | No | `"method": "mtp"`, `"num_speculative_tokens": <N>` |
| Gemma 4 MTP | Yes | `"method": "mtp"`, `"model": "<matching-assistant>"`, `"num_speculative_tokens": <N>` |
| EAGLE-3 | Yes | `"method": "eagle3"`, `"model": "<matching-speculator>"`, `"num_speculative_tokens": <N>` |
| DFlash | Yes | `"method": "dflash"`, `"model": "<matching-speculator>"`, `"num_speculative_tokens": <N>` |
| DSpark | Yes | `"method": "dspark"`, `"model": "<matching-speculator>"`, `"num_speculative_tokens": <N>` |

Native MTP omits `model` (draft is inside the target):

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "mtp",
    "num_speculative_tokens": <N>
  }'
```

Gemma 4 MTP, EAGLE-3, DFlash, DSpark point `model` at a checkpoint trained for that target:

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "<method>",
    "model": "<matching-draft-checkpoint>",
    "num_speculative_tokens": <N>
  }'
```

Gemma 4 assistant checkpoints still use the MTP path even though they arrive through `model`. vLLM connects the assistant to the target and lets it share target KV.

Before enabling: installed vLLM supports the method and architecture; draft checkpoint matches target and method; `num_speculative_tokens` is compatible with the checkpoint; the model card supports the intended hardware and inference backend.

### Memory considerations

Native MTP does not load a separate draft checkpoint and may share the embedding table or output head with the target. Gemma 4 MTP, EAGLE-3, DFlash, and DSpark load extra draft weights — reserve GPU headroom. Overhead depends on draft size, precision, tensor-parallel layout, and runtime buffers.

## Where to find the pretrained draft models

Hugging Face publishers named on the page:

| Draft-model publisher | Methods | Representative models and targets |
| --- | --- | --- |
| Google | Gemma 4 MTP | Assistant checkpoints for Gemma 4 E2B, E4B, 12B, 26B-A4B, and 31B [[7]](#ref-7) |
| LightSeek Foundation | EAGLE-3 and EAGLE-3.1 | EAGLE drafts for Kimi-K2.5, Kimi-K2.6, Kimi-K2.7-Coder, including standard and MLA variants [[8]](#ref-8) |
| Red Hat AI | EAGLE-3, DFlash, DSpark | Llama, Qwen, Gemma, GPT-OSS, GLM, Nemotron, Mistral; suffixes `-speculator.eagle3`, `-speculator.dflash`, `-speculator.dspark` [[9]](#ref-9) |
| Z-Lab | DFlash | Qwen3 / Qwen3.5 / Qwen3.6, Gemma 4, Kimi, MiniMax, GPT-OSS, Llama; names generally `<target>-DFlash` [[10]](#ref-10) |
| DeepSeek AI | EAGLE-3, DFlash, DSpark | DeepSpec: all three methods for Qwen3-4B / 8B / 14B and Gemma 4 12B. Examples: `eagle3_qwen3_8b_ttt7`, `dflash_qwen3_8b_block7`, `dspark_qwen3_8b_block7` [[11]](#ref-11) |
| Inferact | EAGLE-3 and DSpark | `Inferact/MiniMax-M3-EAGLE3` (and GQA variants), `Inferact/Kimi-K3-DSpark` [[12]](#ref-12) |

## Experimental setup and measurements

After enabling, the practical question is whether extra drafting work improves end-to-end serving. Candidates need not be correct at every position; the target checks before commit. Performance depends on how many proposals are accepted and whether saved target Decode work outweighs draft + verify cost.

They evaluate on **task-grounded** benchmarks, not random token sequences. Acceptance depends on structure and predictability of real outputs.

Main indicators:

- Output-token throughput and speedup over the non-speculative baseline.
- Mean accepted length and draft-token acceptance rates, where available.
- Model quality relative to the non-speculative baseline.

### Models and experiment coverage

Check mark = results exist for that target–method pair; dash = not in this sweep. Publisher of the draft is in the cell.

| Target model | Native MTP | Gemma 4 MTP | EAGLE-3 | DFlash | DSpark |
| --- | --- | --- | --- | --- | --- |
| `google/gemma-4-26B-A4B-it` | — | ✓ Google | ✓ Red Hat AI | ✓ Z-Lab | — |
| `google/gemma-4-31B-it` | — | ✓ Google | ✓ Red Hat AI | ✓ Z-Lab | ✓ Red Hat AI |
| `Qwen/Qwen3-8B` | — | — | ✓ Red Hat AI | ✓ Z-Lab | ✓ DeepSeek |
| `Qwen/Qwen3.5-27B` | ✓ Built-in | — | — | ✓ Z-Lab | — |
| `Qwen/Qwen3.5-122B-A10B` | ✓ Built-in | — | — | ✓ Z-Lab | — |
| `Qwen/Qwen3.6-27B` | ✓ Built-in | — | — | ✓ Z-Lab | — |
| `Qwen/Qwen3.6-35B-A3B` | ✓ Built-in | — | — | ✓ Z-Lab | — |
| `moonshotai/Kimi-K2.5` | — | — | ✓ LightSeek | ✓ Z-Lab | — |
| `MiniMaxAI/MiniMax-M3-MXFP8` | — | — | ✓ Inferact | — | — |

Read each result inside its test config: architecture, active-parameter count, draft size, workload, and serving conditions all move the number.

### Throughput measurements

Generated tokens per second vs a standard autoregressive baseline; sweep speculative token count `N` to see how speculation depth moves end-to-end serving TPS.

**Figure 4** on the original page is an interactive Plotly bar chart (selector by target model; hover for speedup and selected `N`). Not reproduced here. The numeric claims below are the ones written in the post body.

### Main observations

Measurements varied by target, method, workload, and `N`.

**`gemma-4-26B-A4B-it`.** Largest measured throughput ratios in the sweep: **2.74×** and **2.62×** for Gemma 4 MTP on GSM8K and MBPP; **2.87×** and **2.79×** for DFlash on MATH500 and HumanEval. EAGLE-3 ranged **2.11×–2.27×** across the four datasets.

**`gemma-4-31B-it`.** Gemma 4 MTP reached **2.00×** on GSM8K and **1.99×** on MBPP. DFlash reached **2.34×** on MATH500 and **2.05×** on HumanEval. EAGLE-3 and DSpark were also above baseline on all four evaluated datasets (exact ratios not written). Best `N` varied by workload.

**`Qwen3-8B`.** DSpark **1.15×** (MATH500) to **1.63×** (GSM8K). DFlash **1.08×–1.27×**. EAGLE-3 above baseline on GSM8K, HumanEval, and MBPP; its largest measured MATH500 value stayed **below** baseline.

**`Qwen3.5-27B`, `Qwen3.5-122B-A10B`, `Qwen3.6-27B`.** Maximum measured native-MTP values in the sweeps were higher than the corresponding maximum DFlash values. Largest ratio in this group: **2.20×** for Qwen3.5-122B-A10B on MATH500. Native-MTP `N` at that maximum ranged **N=4 to N=7**, depending on model and dataset.

**`Qwen3.6-35B-A3B`.** DFlash **1.77×–2.06×**, largest at **N=7** on each of the four datasets. Native MTP **1.28×–1.49×**, largest at **N=6**. Same family as Qwen3.6-27B, different ranking — results can move between models in one family.

**`MiniMax-M3-MXFP8`.** EAGLE-3 reached **2.09×** on HumanEval at **N=4**. (This target ran on MI355X; see disclaimer.)

**`Kimi-K2.5`.** EAGLE-3 up to **2.33×**; DFlash up to **2.68×**. Largest EAGLE-3 values generally at **N=4**; largest DFlash at **N=7**.

Across experiments, the `N` tied to the largest measured throughput was not constant. Sequential methods: throughput often rose over the first few `N` then plateaued. DFlash and DSpark: **N=7** was frequently among the higher-throughput settings; larger `N` did not consistently help.

These observations are for the hardware, software, target, draft checkpoint, workload, and sweep in this study.

## Tuning considerations

Treat speculative decoding as a runtime optimization, not one `N` that works for every workload. Best `num_speculative_tokens` depends on how many proposals are accepted and whether avoided target Decode outweighs draft + verify cost.

A model-card recommendation is a start; pick the final setting from representative workloads and end-to-end measurements. Useful signals: throughput, mean accepted length, overall acceptance rate, **per-position** acceptance rate.

A larger proposal window gives more chances to commit several tokens in one verification. Acceptance often drops at later draft positions. Extra candidates then add work and little accept → TPS flattens or regresses.

### Start from a supported configuration

Native MTP: **N=1** is the conservative start (least extra sequential drafting):

```json
{"method": "mtp", "num_speculative_tokens": 1}
```

After correctness and stability, sweep 2, 3, 4, 5, 6, 7.

In these measurements, native-MTP `N` at largest throughput:

- **Qwen3.5-27B:** N=5 for GSM8K and MATH500; N=4 for HumanEval and MBPP; N=3 for MT-Bench.
- **Qwen3.5-122B-A10B:** N=7 across the four listed reasoning and code datasets.
- **Qwen3.6-27B:** N=4 or N=5.
- **Qwen3.6-35B-A3B:** throughput increased through N=6.

Gemma 4 MTP and EAGLE-3 also add sequential drafting as `N` grows. A short sweep is useful even when the checkpoint recommends a config. In these Gemma 4 and EAGLE-3 runs, measured TPS generally increased over the first few `N` then plateaued.

DFlash: start from proposal lengths the checkpoint recommends or supports. Many DFlash checkpoints train with a fixed `block_size`. Example:

```text
block_size = 16
num_speculative_tokens = 15
```

First position is the confirmed anchor; the other 15 are draft candidates. That is the **maximum supported** proposal length, not necessarily the highest-TPS setting. The post suggests testing smaller values: `N = 3, 7, 11, 15`. Across their DFlash experiments, **N=7** was frequently among the higher-throughput settings; for some workloads the largest measured TPS was at **N=11**.

DSpark: `num_speculative_tokens` is how many candidates each speculative round generates. In these vLLM experiments the **full** configured proposal was submitted for verification, so compare N=3 vs N=7 (and so on) with end-to-end TPS.

### Monitor acceptance behavior

| Signal | What it shows |
| --- | --- |
| Throughput | End-to-end serving vs non-speculative baseline |
| Mean accepted length | Draft tokens committed per speculative round, on average |
| Overall acceptance rate | Fraction of proposed draft tokens accepted |
| Per-position acceptance rate | Whether later positions in the proposal still pay |

Per-position accept is the knob for `N`. If the first few positions accept often and later ones contribute almost nothing, shrinking `num_speculative_tokens` can raise TPS by skipping useless draft work.

Read acceptance **with** throughput. Cheap drafts can beat baseline even at a lower accept rate. High accept does not guarantee higher TPS if the draft is expensive.

### Match the sweep to the workload

GSM8K and MATH500: medium or deeper `N` often had the higher measured TPS in these sweeps. Native MTP on Qwen3.5-122B-A10B increased through N=7. DFlash highs frequently at N=7 or N=11.

HumanEval and MBPP: moderate `N` often among the higher-throughput settings. Code has local structure, but formatting, identifiers, and implementation choices can still diverge a plausible continuation.

### Example tuning workflow

1. Start from a supported / recommended checkpoint config.
2. Benchmark with representative prompts and generation settings.
3. Record throughput, mean accepted length, and acceptance rates.
4. Sweep several smaller and larger proposal lengths.
5. Pick on the metric that matters for the intended workload. Here, end-to-end serving TPS was the primary selection metric.

The winner need not have the longest proposal, the highest accept rate, or the largest mean accepted length. Trade drafting cost, verification cost, accepted tokens, and the metric you actually care about.

## Training a speculator for a new target model

This guide does not cover training in depth. Workflow below is a sketch from vLLM Speculators and DeepSpec [[13]](#ref-13) [[14]](#ref-14) [[15]](#ref-15). Hidden export through vLLM: [extract-hidden-states.md](../architecture/extract-hidden-states.md).

1. Prepare representative prompts.
2. Generate responses with the **exact** target model.
3. Choose a hidden-state generation mode.
4. Collect the required target hidden states.
5. Train the speculator.
6. Test acceptance and serving throughput.

Prompts should match the expected workload (chat, math, code, tool use, multilingual). Keep a held-out eval set. Tokenizer, chat template, thinking mode, and generation config should match deployment. Applying the target tokenizer or chat template to **existing** responses does not make the data target-specific; the responses themselves must come from the target.

### Choose how to obtain hidden states

| Training mode | How it works | Main consideration |
| --- | --- | --- |
| Online | Hidden generated by a running vLLM server when needed, then discarded | Avoids a large disk cache; needs resources for target inference and training at once |
| Offline | Hidden generated and stored before training | Frees all GPUs for training afterward; needs substantial storage |
| Hybrid | Hidden generated and cached in the first epoch, then reused | Pays generation cost once without a separate preprocess stage |

Mode changes where hidden come from; the rest of training is largely the same.

A vLLM server can run the target and expose hidden from the layers the drafting method needs. Custom layer picks must match the speculator-training config.

- EAGLE-3: selected-layer hidden for autoregressive drafting [[4]](#ref-4).
- DFlash: target features to train a parallel block predictor [[16]](#ref-16).
- DSpark: light sequential and confidence heads on a DFlash-style net [[6]](#ref-6).
- MTP: fine-tunes the target’s own MTP component — the target must already have compatible MTP layers [[13]](#ref-13).

After training, inspect the checkpoint and serve it with the target in vLLM. Training loss is not enough: measure accepted length, acceptance rate, draft latency, GPU memory, and end-to-end serving TPS. Weak accept on a workload → change prompt mix or training config and repeat. Same target, same generation mode, same representative workload.

## Summary

Draft proposes; target verifies; nothing is committed until the target says so.

Five drafting approaches differ in how they use target information and whether candidates are sequential, parallel, or parallel-plus-light sequential correction.

Experiments: selected Gemma, Qwen, MiniMax, and Kimi models on Instinct **MI300X** and **MI355X**, ROCm. Measured TPS moved with target, draft checkpoint, workload, `N`, and serving config.

Some settings were small or **below** the non-speculative baseline. Several model–workload combinations were **above 2×**. Upper end written on the page: **2.87×** DFlash on `gemma-4-26B-A4B-it`, **2.83×** Gemma 4 MTP on the same target, **2.68×** DFlash on Kimi-K2.5.

`N` mattered. Increasing `num_speculative_tokens` sometimes helped for the first few settings, then plateaued or fell. Checkpoint recommendations are starting points; pick a deploy config from representative workload measurements and acceptance metrics.

## Future work

- Non-learned approaches such as n-gram speculation and suffix decoding, especially for repeated-token workloads (code editing, agentic loops).
- Broader eval across concurrency, prompt/output lengths, batch sizes, and sampling settings.
- How speculator training data moves accept across code, math, chat, multilingual, tool use, and structured output.
- Deeper profiling of draft generation, target verification, KV-cache behavior, graph execution, and scheduling.

## References

1. <a id="ref-1"></a> vLLM, Speculative Decoding — https://docs.vllm.ai/en/latest/features/speculative_decoding/
2. <a id="ref-2"></a> vLLM, MTP Speculative Decoding — https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/
3. <a id="ref-3"></a> Google Developers Blog, Multi-token prediction in Gemma 4 — https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/
4. <a id="ref-4"></a> EAGLE-3, *Scaling up Inference Acceleration…* — https://arxiv.org/pdf/2503.01840
5. <a id="ref-5"></a> Z-Lab DFlash GitHub — https://github.com/z-lab/dflash
6. <a id="ref-6"></a> DSpark paper — https://arxiv.org/pdf/2607.05147
7. <a id="ref-7"></a> Google Gemma 4 collection — https://huggingface.co/collections/google/gemma-4
8. <a id="ref-8"></a> LightSeek Foundation models — https://huggingface.co/lightseekorg/models
9. <a id="ref-9"></a> Red Hat AI Speculator Models — https://huggingface.co/collections/RedHatAI/speculator-models
10. <a id="ref-10"></a> Z-Lab DFlash collection — https://huggingface.co/collections/z-lab/dflash
11. <a id="ref-11"></a> DeepSeek-AI DeepSpec — https://huggingface.co/collections/deepseek-ai/deepspec
12. <a id="ref-12"></a> Inferact models — https://huggingface.co/Inferact/models
13. <a id="ref-13"></a> vLLM Speculators, Training a Speculator — https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/
14. <a id="ref-14"></a> vLLM Speculators GitHub — https://github.com/vllm-project/speculators
15. <a id="ref-15"></a> DeepSeek-AI DeepSpec GitHub — https://github.com/deepseek-ai/DeepSpec
16. <a id="ref-16"></a> DFlash paper — https://arxiv.org/pdf/2602.06036

## Appendix: interactive heatmap (not copied)

The original appendix is an **interactive HTML heatmap** over **9** targets × methods × experiments (per-position acceptance by proposal length `N`; each row also shows measured speedup and output tok/s). That widget is CSS/JS on the page. This note does not dump it. Per-position accept **percentages** were not printed as a static table in the cleaned extract — do not invent them. Use the original page to hover a cell.

Nine targets (same coverage table as above):

1. `google/gemma-4-26B-A4B-it`
2. `google/gemma-4-31B-it`
3. `Qwen/Qwen3-8B`
4. `Qwen/Qwen3.5-27B`
5. `Qwen/Qwen3.5-122B-A10B`
6. `Qwen/Qwen3.6-27B`
7. `Qwen/Qwen3.6-35B-A3B`
8. `moonshotai/Kimi-K2.5`
9. `MiniMaxAI/MiniMax-M3-MXFP8`

Each heatmap row, as described on the page: per-position acceptance by `N`, plus that run’s **speedup** and **output tok/s**.

**Baseline tok/s printed in the cleaned captions** (all `google/gemma-4-26B-A4B-it`; same four baselines reused across Gemma 4 MTP / EAGLE-3 / DFlash captions):

| Dataset | Baseline output tok/s |
| --- | ---: |
| GSM8K | 2,344 |
| MATH500 | 2,181 |
| HumanEval | 1,854 |
| MBPP | 2,163 |

**Numeric speedups written in the post** (relisted; nothing beyond the body). Summary also names **2.83×** Gemma 4 MTP on `gemma-4-26B-A4B-it` as an upper-end example (the GSM8K/MBPP pair in “Main observations” is 2.74× / 2.62×).

| Target | Method | Written speedup | Notes |
| --- | --- | --- | --- |
| gemma-4-26B-A4B-it | Gemma 4 MTP | 2.74× GSM8K, 2.62× MBPP; 2.83× named in Summary | |
| gemma-4-26B-A4B-it | DFlash | 2.87× MATH500, 2.79× HumanEval | |
| gemma-4-26B-A4B-it | EAGLE-3 | 2.11×–2.27× across four datasets | |
| gemma-4-31B-it | Gemma 4 MTP | 2.00× GSM8K, 1.99× MBPP | |
| gemma-4-31B-it | DFlash | 2.34× MATH500, 2.05× HumanEval | |
| gemma-4-31B-it | EAGLE-3, DSpark | above baseline, four datasets | exact × not written |
| Qwen3-8B | DSpark | 1.15× MATH500 … 1.63× GSM8K | |
| Qwen3-8B | DFlash | 1.08×–1.27× | |
| Qwen3-8B | EAGLE-3 | above baseline on GSM8K / HumanEval / MBPP; MATH500 **below** baseline | exact × not written |
| Qwen3.5-27B / 122B-A10B / Qwen3.6-27B | Native MTP vs DFlash | native-MTP max > DFlash max; **2.20×** Qwen3.5-122B-A10B MATH500 | native MTP N=4–7 |
| Qwen3.6-35B-A3B | DFlash | 1.77×–2.06× at N=7 | |
| Qwen3.6-35B-A3B | Native MTP | 1.28×–1.49× at N=6 | |
| MiniMax-M3-MXFP8 | EAGLE-3 | 2.09× HumanEval at N=4 | MI355X |
| Kimi-K2.5 | EAGLE-3 | up to 2.33×, generally N=4 | |
| Kimi-K2.5 | DFlash | up to 2.68×, N=7 | |

## Acknowledgements

Hongxia Yang and Peng Sun (AMD); Pin Siang Tan, Jun Kang Chow, and Ye Hur Cheong (Embedded LLM).

## Disclaimer

Measurements on AMD Instinct™ MI300X and MI355X with the configurations below.

**Hardware**

- Hardware 1: **8×** AMD Instinct™ **MI300X** (gfx942) with **2×** AMD EPYC™ **9654** 96-Core.
- Hardware 2: **8×** AMD Instinct™ **MI355X** (gfx950) with **2×** AMD EPYC™ **9575F** 64-Core. Used for the **MiniMax-M3-MXFP8** experiment.

**Software**

Ubuntu **22.04.5** LTS, ROCm/HIP runtime **7.2.53211**, vLLM **0.23.1rc1.dev1120+g0f0f28b53**, PyTorch **2.11.0+gitd0c8b1f**, Transformers **5.13.1**, Python **3.12.13**.

Server manufacturers may vary configurations. Performance may vary with configuration, software, vLLM version, and drivers / optimizations.
