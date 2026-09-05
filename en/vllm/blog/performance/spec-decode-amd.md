---
source: https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus
lang: en
fetched: 2026-09-05
---

# Speculative decoding on AMD GPUs: five draft paths

Chinese: [zh/vllm/blog/performance/spec-decode-amd.md](../../../../zh/vllm/blog/performance/spec-decode-amd.md)

2026-08-23. **AMD and Embedded LLM** (acknowledgements on the page). Study note; benches on **MI300X / MI355X**, ROCm, not your SLA. Snapshot in the disclaimer: vLLM `0.23.1rc1.dev1120+g0f0f28b53`, ROCm/HIP `7.2.53211`. Original page: https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus

Accept math is still [spec-decode.md](spec-decode.md). How the draft is grown: [parallel-drafting.md](parallel-drafting.md) (P-EAGLE / DFlash / DSpark) and [p-eagle.md](p-eagle.md). DSpark confidence budget (not on in these AMD runs): [dspark-adaptive.md](dspark-adaptive.md). Later EAGLE attention-drift fix: [eagle-3-1.md](eagle-3-1.md). Training hidden export: [extract-hidden-states.md](../architecture/extract-hidden-states.md). ROCm attention backends on the same GPUs: [rocm-attention.md](../architecture/rocm-attention.md).

**TL;DR from the page:** speculative decoding lets vLLM verify several drafted tokens in one target-model pass. Output-token throughput moved with drafting method and proposal length `N`, and also with model family, draft checkpoint, workload, and acceptance. Upper end of the measured range: **2.87×** DFlash on `gemma-4-26B-A4B-it`, **2.83×** Gemma 4 MTP on the same target, **2.68×** DFlash on Kimi-K2.5. Some sweeps were near-flat or below the non-speculative baseline. This is **how to turn the five methods on and measure them on ROCm**, not new accept math.

## Introduction

Large language models support a wide range of applications, but serving them at scale still needs careful optimization. The baseline used by most LLM serving systems is standard autoregressive decoding: generate one token, append it, generate the next. Simple and reliable, and the loop still advances one committed token at a time because outputs must be produced left-to-right.

Speculative decoding [[1]](#ref-1) keeps that output behavior and splits the loop into **draft** and **verify**. A lightweight draft component proposes candidate future tokens; the original model, as **target**, checks them before they are committed. When several drafts survive, several output tokens come out of one target verification step.

This post explores how speculative decoding works in vLLM and shares measurements from their test environment. It first reviews the autoregressive baseline and the draft-and-verify process, then five drafting approaches that differ in what they take from the target and whether candidates are sequential, autoregressive, parallel, or hybrid: **Native MTP**, **Gemma 4 MTP**, **EAGLE-3**, **DFlash**, **DSpark**. Then how to enable the methods they tested, measurements on Instinct **MI300X** and **MI355X** with ROCm, and practical tuning and observability.

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

Several organizations publish pretrained drafts on Hugging Face. Google ships MTP assistants for Gemma 4; Z-Lab maintains a DFlash collection. Red Hat AI covers EAGLE-3, DFlash, and DSpark; DeepSeek's DeepSpec collection matches all three methods. LightSeek focuses on EAGLE drafts for Kimi; Inferact publishes drafts for MiniMax and Kimi.

Publishers named on the page:

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

They evaluate model quality and serving performance on **task-grounded** benchmarks, not random token sequences. Acceptance depends on the structure and predictability of actual outputs, so task prompts give a more representative view of practical performance.

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

**Figure 4** on the original page is an interactive Plotly bar chart of measured output throughput by method and experiment, with the non-speculative baseline as a reference. Use the selector to switch target models; hover over bars to see speedup and selected proposal length `N`. Not reproduced here. The numeric claims below are the ones written in the post body.

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

Observability therefore matters. A model-card recommendation is a start; pick the final setting from representative workloads and end-to-end measurements. Useful signals: throughput, mean accepted length, overall acceptance rate, **per-position** acceptance rate.

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

DFlash: start from proposal lengths the checkpoint recommends or supports. Many DFlash checkpoints train with a fixed `block_size`. For example, when `block_size = 16`, the maximum proposal length is normally:

```text
num_speculative_tokens = 15
```

because the first position is the confirmed anchor and the remaining 15 positions are draft candidates. That is the **maximum supported** proposal length, not necessarily the highest-TPS setting. In practice it is useful to test smaller values:

```text
N = 3, 7, 11, 15
```

Across their DFlash experiments, **N=7** was frequently among the higher-throughput settings; for some workloads the largest measured TPS was at **N=11**.

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

This guide does not cover speculator training in depth. The workflow below summarizes practical considerations from vLLM Speculators and DeepSpec [[13]](#ref-13) [[14]](#ref-14) [[15]](#ref-15). Hidden export through vLLM: [extract-hidden-states.md](../architecture/extract-hidden-states.md).

A typical workflow is:

1. Prepare representative prompts.
2. Generate responses with the **exact** target model.
3. Choose a hidden-state generation mode.
4. Collect the required target hidden states.
5. Train the speculator.
6. Test acceptance and serving throughput.

### Prepare representative prompts

Start with prompts that reflect the expected workload: chat, mathematics, code generation, tool use, or multilingual tasks. Keep a separate evaluation set.

Responses used for training should be generated by the exact target the speculator will support. Tokenizer, chat template, thinking mode, and generation config should match the intended deployment. The vLLM documentation emphasizes that applying the target tokenizer or chat template to **existing** responses does not make the data target-specific; the responses themselves must come from the target.

### Choose how to obtain hidden states

The speculator receives internal hidden states from the target during training. The vLLM Speculators workflow supports three ways to provide them:

| Training mode | How it works | Main consideration |
| --- | --- | --- |
| Online | Hidden generated by a running vLLM server when needed, then discarded | Avoids a large disk cache; needs resources for target inference and training at once |
| Offline | Hidden generated and stored before training | Frees all GPUs for training afterward; needs substantial storage |
| Hybrid | Hidden generated and cached in the first epoch, then reused | Pays generation cost once without a separate preprocess stage |

The selected mode changes where the hidden states come from; the remaining training workflow is largely the same.

### Collect target-model information

A vLLM server can run the target and expose hidden from the layers the drafting method needs. When custom target layers are chosen, the same layer selections must also be used in the speculator-training configuration.

The information collected depends on the method:

- EAGLE-3 uses hidden states from selected target layers for autoregressive drafting [[4]](#ref-4).
- DFlash uses target features to train a network that predicts a block of future positions in parallel [[16]](#ref-16).
- DSpark adds lightweight sequential and confidence heads to a DFlash-style draft network [[6]](#ref-6).
- MTP training fine-tunes the target’s own MTP component and therefore requires a target that already contains compatible MTP layers [[13]](#ref-13).

### Train and test the speculator

The speculator configuration must match the target’s hidden size, vocabulary, tokenizer, and selected target layers. Method-specific settings such as draft-network depth, block size, sequence length, and learning rate must also be selected.

After training, inspect the checkpoint and serve it together with the target in vLLM. Training loss alone is not enough; the important measurements are accepted length, acceptance rate, draft latency, GPU memory, and end-to-end serving TPS. The vLLM Speculators tutorial covers the path from data preparation and hidden-state extraction to checkpoint testing and serving.

When acceptance is weak for a particular workload, the prompt mixture or training configuration can be adjusted and the process repeated. The main principle is to use the same target, generation mode, and representative workload the speculator is expected to support.

## Summary

This post treats speculative decoding in vLLM as a draft-and-verify approach for LLM serving. A draft component proposes candidate future tokens; the target evaluates the proposal before any tokens are committed.

Five drafting approaches — native MTP, Gemma 4 MTP, EAGLE-3, DFlash, and DSpark — differ mainly in how they use information from the target and whether candidates are sequential, parallel, or parallel plus lightweight sequential correction.

The experiments covered selected Gemma, Qwen, MiniMax, and Kimi models on Instinct **MI300X** and **MI355X** with ROCm. Measured throughput varied across target models, draft checkpoints, workloads, proposal lengths, and serving configurations.

Across the tested configurations, some settings produced smaller changes or throughput **below** the non-speculative baseline, while several model–workload combinations produced ratios **above 2×**. Examples at the upper end of the observed range: **2.87×** DFlash on `gemma-4-26B-A4B-it`, **2.83×** Gemma 4 MTP on the same target, **2.68×** DFlash on Kimi-K2.5.

Proposal length was also an experimental variable. Increasing `num_speculative_tokens` sometimes increased throughput over the first few settings; larger values could plateau or fall. Checkpoint recommendations can provide starting points, but representative workload measurements and acceptance metrics are needed when selecting a deployment configuration.

## Future work

Future benchmarking could include non-learned approaches such as n-gram speculation and suffix decoding, particularly for workloads with repeated token patterns such as code editing and agentic loops.

Broader evaluation across concurrency levels, prompt and output lengths, batch sizes, and sampling settings would also help show how speculative decoding behaves under different serving conditions.

Another useful direction is to study how speculator training data affects acceptance across code, mathematics, chat, multilingual prompts, tool use, and structured output. That could give clearer guidance when choosing or training a draft checkpoint for a specific workload.

Finally, deeper profiling of draft generation, target verification, KV-cache behavior, graph execution, and scheduling would help explain the performance differences observed across target models and workloads.

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

## Appendix: per-position acceptance heatmap (not copied) + nine serve recipes

The appendix focuses on acceptance by draft position. The original page is an interactive HTML widget: pick a target model, drafting method, and experiment to view one larger per-position acceptance heatmap. Rows are proposal lengths `N`; columns are draft positions; darker cells indicate higher acceptance. Each row also includes measured speedup and output throughput for context. **That widget is CSS/JS on the page. This note does not paste Plotly, heatmap HTML, or invent a static per-position percentage table.** Hover a cell on the original post.

**MAL** means mean accepted length (draft tokens committed per speculative round, on average). **AR** means acceptance rate (fraction of proposed draft tokens accepted). Each heatmap row’s small print is typically `speedup | tok/s` plus `MAL … | AR …%`.

### Baseline output tok/s printed in heatmap captions

These numbers are the non-speculative baselines printed in the appendix heatmap captions, not the per-position cells. MiniMax-M3-MXFP8 ran on MI355X; the others used this study’s MI300X configuration.

| Target | GSM8K | MATH500 | HumanEval | MBPP |
| --- | ---: | ---: | ---: | ---: |
| `google/gemma-4-26B-A4B-it` | 2,344 | 2,181 | 1,854 | 2,163 |
| `google/gemma-4-31B-it` | 1,631 | 1,365 | 1,228 | 1,519 |
| `Qwen/Qwen3-8B` | 3,698 | 3,530 | 3,226 | 3,268 |
| `Qwen/Qwen3.5-27B` | 1,555 | 1,500 | 1,256 | 1,418 |
| `Qwen/Qwen3.5-122B-A10B` | 1,494 | 1,446 | 1,105 | 1,459 |
| `Qwen/Qwen3.6-27B` | 1,521 | 1,514 | 1,481 | 1,495 |
| `Qwen/Qwen3.6-35B-A3B` | 2,275 | 2,235 | 2,193 | 2,258 |
| `moonshotai/Kimi-K2.5` | 324 | 310 | 301 | 311 |
| `MiniMaxAI/MiniMax-M3-MXFP8` | 2,086 | 2,468 | 2,317 | 2,277 |


### Example `vllm serve` commands used in the experiments

The source wraps nine targets in a `<details>` block: baseline first, then each method they actually ran. CLI is copied as shipped. Gemma 4 MTP assistants still use the MTP path even when they arrive through `model`, and often omit an extra `"method"` field. `num_speculative_tokens` here is the recipe example, not necessarily the throughput-sweep winner.

### `google/gemma-4-26B-A4B-it`

Baseline:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve google/gemma-4-26B-A4B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --chat-template /app/vllm/examples/tool_chat_template_gemma4.jinja \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768
```

Gemma 4 MTP:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve google/gemma-4-26B-A4B-it \
  --tensor-parallel-size 2 \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --chat-template /app/vllm/examples/tool_chat_template_gemma4.jinja \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --speculative-config '{"model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}'
```

EAGLE-3:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve google/gemma-4-26B-A4B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --chat-template /app/vllm/examples/tool_chat_template_gemma4.jinja \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.8 \
  --speculative-config '{"model":"RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3","num_speculative_tokens":1,"method":"eagle3"}'
```

DFlash:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve google/gemma-4-26B-A4B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --attention-backend triton_attn \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --chat-template /app/vllm/examples/tool_chat_template_gemma4.jinja \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.8 \
  --speculative-config '{"method":"dflash","model":"z-lab/gemma-4-26B-A4B-it-DFlash","num_speculative_tokens":15,"attention_backend":"triton_attn"}'
```

### `google/gemma-4-31B-it`

Baseline:

```bash
vllm serve google/gemma-4-31B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --chat-template /app/vllm/examples/tool_chat_template_gemma4.jinja \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768
```

Gemma 4 MTP:

```bash
vllm serve google/gemma-4-31B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --chat-template /app/vllm/examples/tool_chat_template_gemma4.jinja \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --speculative-config '{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":1}'
```

EAGLE-3:

```bash
vllm serve google/gemma-4-31B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --speculative-config '{"model":"RedHatAI/gemma-4-31B-it-speculator.eagle3","num_speculative_tokens":3,"method":"eagle3"}'
```

DFlash:

```bash
vllm serve google/gemma-4-31B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --attention-backend triton_attn \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"dflash","model":"z-lab/gemma-4-31B-it-DFlash","num_speculative_tokens":15,"attention_backend":"triton_attn"}'
```

DSpark:

```bash
vllm serve google/gemma-4-31B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --attention-backend triton_attn \
  --language-model-only \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"model":"RedHatAI/gemma-4-31B-it-speculator.dspark","num_speculative_tokens":7,"method":"dspark","attention_backend":"triton_attn"}'
```

### `Qwen/Qwen3-8B`

Baseline:

```bash
vllm serve Qwen/Qwen3-8B \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

EAGLE-3:

```bash
vllm serve Qwen/Qwen3-8B \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"model":"RedHatAI/Qwen3-8B-Thinking-speculator.eagle3","num_speculative_tokens":5,"method":"eagle3"}'
```

DFlash:

```bash
vllm serve Qwen/Qwen3-8B \
  --trust-remote-code \
  --max-num-batched-tokens 16384 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"model":"z-lab/Qwen3-8B-DFlash-b16","method":"dflash","num_speculative_tokens":7}'
```

DSpark:

```bash
vllm serve Qwen/Qwen3-8B \
  --trust-remote-code \
  --max-num-batched-tokens 16384 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"model":"deepseek-ai/dspark_qwen3_8b_block7","method":"dspark","num_speculative_tokens":11}'
```

### `Qwen/Qwen3.5-27B`

Baseline:

```bash
vllm serve Qwen/Qwen3.5-27B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 32768
```

Native MTP:

```bash
vllm serve Qwen/Qwen3.5-27B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 32768 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

DFlash:

```bash
vllm serve Qwen/Qwen3.5-27B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 32768 \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-27B-DFlash","num_speculative_tokens":15}'
```

### `Qwen/Qwen3.5-122B-A10B`

Baseline:

```bash
vllm serve Qwen/Qwen3.5-122B-A10B \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-num-batched-tokens 32768
```

Native MTP:

```bash
vllm serve Qwen/Qwen3.5-122B-A10B \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-num-batched-tokens 32768 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":7}'
```

DFlash:

```bash
vllm serve Qwen/Qwen3.5-122B-A10B \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-num-batched-tokens 32768 \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-122B-A10B-DFlash","num_speculative_tokens":15}'
```

### `Qwen/Qwen3.6-27B`

Baseline:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve Qwen/Qwen3.6-27B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 32768
```

Native MTP:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve Qwen/Qwen3.6-27B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 32768 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

DFlash:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve Qwen/Qwen3.6-27B \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 32768 \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":15}'
```

### `Qwen/Qwen3.6-35B-A3B`

Baseline:

```bash
VLLM_ROCM_USE_AITER=1 \
vllm serve Qwen/Qwen3.6-35B-A3B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --mm-encoder-tp-mode data \
  --max-num-batched-tokens 16384
```

Native MTP:

```bash
VLLM_ROCM_USE_AITER=1 \
vllm serve Qwen/Qwen3.6-35B-A3B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --mm-encoder-tp-mode data \
  --max-num-batched-tokens 16384 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
```

DFlash:

```bash
VLLM_ROCM_USE_AITER=1 \
vllm serve Qwen/Qwen3.6-35B-A3B \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --mm-encoder-tp-mode data \
  --max-num-batched-tokens 16384 \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":15}'
```

### `moonshotai/Kimi-K2.5`

Baseline:

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
vllm serve moonshotai/Kimi-K2.5 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --language-model-only \
  --reasoning-parser kimi_k2 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2
```

EAGLE-3:

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
vllm serve moonshotai/Kimi-K2.5 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --language-model-only \
  --reasoning-parser kimi_k2 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --speculative-config '{"model":"lightseekorg/kimi-k2.5-eagle3-mla","method":"eagle3","num_speculative_tokens":3}'
```

DFlash:

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
vllm serve moonshotai/Kimi-K2.5 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --language-model-only \
  --reasoning-parser kimi_k2 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --speculative-config '{"model":"z-lab/Kimi-K2.5-DFlash","method":"dflash","num_speculative_tokens":7}'
```

### `MiniMaxAI/MiniMax-M3-MXFP8`

Baseline:

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
VLLM_ROCM_USE_AITER_MOE=1 \
vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --tensor-parallel-size 8 \
  --block-size 128 \
  --attention_config.indexer_kv_dtype fp8 \
  --linear-backend emulation \
  --attention-backend TRITON_ATTN \
  --language-model-only \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3
```

EAGLE-3:

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
VLLM_ROCM_USE_AITER_MOE=1 \
vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --tensor-parallel-size 8 \
  --block-size 128 \
  --attention_config.indexer_kv_dtype fp8 \
  --linear-backend emulation \
  --attention-backend TRITON_ATTN \
  --language-model-only \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3 \
  --speculative-config '{"method":"eagle3","model":"Inferact/MiniMax-M3-EAGLE3","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}'
```

## Acknowledgements

Thanks to everyone who contributed to this collaboration, including Hongxia Yang and Peng Sun from AMD, and Pin Siang Tan, Jun Kang Chow, and Ye Hur Cheong from Embedded LLM.

## Disclaimer

Measurements were run on AMD Instinct™ MI300X and MI355X platforms using the configurations below.

**Hardware Configuration**

- Hardware 1: **8×** AMD Instinct™ **MI300X** GPUs (gfx942) with **2×** AMD EPYC™ **9654** 96-Core Processor.
- Hardware 2: **8×** AMD Instinct™ **MI355X** GPUs (gfx950) with **2×** AMD EPYC™ **9575F** 64-Core processors. This platform was used for the **MiniMax-M3-MXFP8** experiment.

**Software Configuration**

Ubuntu **22.04.5** LTS, ROCm/HIP runtime **7.2.53211**, vLLM **0.23.1rc1.dev1120+g0f0f28b53**, PyTorch **2.11.0+gitd0c8b1f**, Transformers **5.13.1**, Python **3.12.13**.

Server manufacturers may vary configurations, yielding different results. Performance may vary based on configuration, software, vLLM version, and the use of the latest drivers and optimizations.
