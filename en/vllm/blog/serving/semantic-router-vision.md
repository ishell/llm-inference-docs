---
source: https://vllm.ai/blog/2026-05-28-vllm-sr-vision-encoder-hardening
lang: en
fetched: 2026-09-04
---

# Vision signals: not a bigger encoder — Candle missed the reference

Chinese: [zh/vllm/blog/serving/semantic-router-vision.md](../../../../zh/vllm/blog/serving/semantic-router-vision.md)

2026-05-28. **David Shrader, Huamin Chen, Xunzhuo Liu, Bowei He, and the vLLM Semantic Router Team**. Repo: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router). Launch: [semantic-router.md](semantic-router.md). Spine: [Iris](semantic-router-iris.md) / [signal-decision](semantic-router-signal.md). Follows Athena’s `multi-modal-embed-small`: [athena](semantic-router-athena.md). Contract around signals: [themis](semantic-router-themis.md). Do not confuse with the in-engine [router.md](router.md). Cosines and probe scores are **their** PR-branch measurements. Until merge: validation trail, not a production guarantee.

Siblings: [modular](semantic-router-modular.md), [amd](semantic-router-amd.md), [mom-amd](semantic-router-mom-amd.md), [session](semantic-router-session.md), [fusion](semantic-router-fusion.md), [micro-agent](semantic-router-micro-agent.md), [mom](semantic-router-mom.md).

Most routers start from a prompt and pick an endpoint. VSR’s bet: extract signals, compose them into decisions, make the path observable **before** the serving model. Iris moved that past a fixed domain classifier. Athena pushed it toward a system-level intelligence layer for MoM and agentic deployments.

Local figures (copyright remains with the original site; study copies):

![hero](../../../../assets/vllm/blog/serving/semantic-router-vision/01-hero.png)

**Figure 1.** The next boundary is multimodal: the image can be the decisive evidence.

Once an image, screenshot, scan, or document page is in the request, the router is no longer reasoning over a prompt alone. A text-only router routes a **partial** request. The important step is not “add an image encoder.” It is turning visual evidence into a trustworthy VSR **signal** in the same decision fabric as text.

The deployed multimodal path around `multi-modal-embed-small` looked **confidently wrong**. First guess: the compact encoder was too weak. Actual issue: the **Rust/Candle path did not match the PyTorch reference** for the same model.

## Multimodal routing is not image classification

Text-only routing already does more than topic matching. Signals are independent observations; decisions compose them with priority and boolean logic; plugins / model refs say what happens next. That is how VSR can say “security-sensitive code review gets a stronger reasoning model and jailbreak checks” instead of “computer science goes to the coding model.”

Multimodal keeps that shape. The unit of analysis becomes the **full request**. Text can be generic while the image carries the decision:

| Request evidence | Text-only router sees | Multimodal router should see |
|---|---|---|
| "Summarize this" + passport image | Generic summarization | Identifier document, PII risk, restricted handling |
| "What does this show?" + chest X-ray | Vague visual question | Clinical image, medical-domain policy, capable VLM target |
| "Find the bug" + code screenshot | Coding request | Code artifact, possible secret leakage, security review path |
| Medical prompt + unrelated car image | Medical text | Out-of-domain visual evidence, clarification or rejection path |

The innovation is not that VSR can compute an image embedding. It is that the embedding becomes a **typed signal** beside text intent, PII, jailbreak, domain, semantic similarity, plugins, and model selection. Prompt-level routing becomes **request-level policy**.

![policy layer](../../../../assets/vllm/blog/serving/semantic-router-vision/02-policy-layer.png)

**Figure 2.** Image evidence joins the same Signal-Decision fabric as text.

If a text signal is wrong, the policy picks the wrong model or skips a plugin. If a vision signal is **anti-correlated**, the router can be confidently wrong and still leave a clean audit trail for the wrong decision. Reference parity is a **control-plane invariant**, not model-quality hygiene.

## When the vision signal was confidently wrong

Not a small accuracy drop. **11-image** probe, three verticals, **21** candidate labels: the deployed `multi-modal-embed-small` (mmes) path ranked the **wrong vertical highest on 9 of 11** images. Medical X-rays scored closer to semiconductor candidates than to medical candidates. Identifier documents did not reliably land near identifier anchors.

That is an **82% inversion** rate. Anti-correlated, not merely noisy.

![inversion heatmap](../../../../assets/vllm/blog/serving/semantic-router-vision/03-inversion-heatmap.png)

**Figure 3.** Inversion heatmap: the deployed path ranks the wrong vertical first.

A weak classifier usually looks uncertain. An inverted one looks confident in the wrong direction. For a multimodal policy layer that can be worse than **no** image signal.

Surface that exposed it: image-modality routing around `multi-modal-embed-small`, including the E2E routing profile in [PR #1881](https://github.com/vllm-project/semantic-router/pull/1881). Real images through the Candle binding made the gap visible.

## The tempting explanation: upgrade the encoder

Hypothesis: compact encoder too weak. The team was already looking at SigLIP2 and larger `multi-modal-embed-large` (mmEL). Direct tests on the same 21-candidate probe:

- SigLIP2-base: **10/10**
- SigLIP-base through Hugging Face Transformers: **10/10**
- mmEL (vision tower based on SigLIP2): **10/10**
- The mmes model card loaded through the **PyTorch reference** path: **10/10**

![encoder eliminated](../../../../assets/vllm/blog/serving/semantic-router-vision/04-encoder-eliminated.png)

**Figure 4.** Encoder family eliminated: the same mmes checkpoint is fine on the PyTorch reference path.

So the encoder family was not the root. Even the “failing” mmes model behaved when loaded through the reference.

Side learning they keep: larger SigLIP2-so400m showed stronger out-of-distribution rejection on this probe (an accidentally included car-engine image). May matter later for defensive routing if memory allows a larger vision tower. **Not** the production inversion bug.

## The reference check that changed the investigation

Same mmes model, same passport fixture, two paths.

PyTorch reference: cosine **0.7204** against the relevant passport anchor. Deployed Candle-binding path: **0.1576**. A **5–8×** magnitude gap on the same model and fixture.

![diagnostic gap](../../../../assets/vllm/blog/serving/semantic-router-vision/05-diagnostic-gap.png)

**Figure 5.** Same checkpoint, same passport: 0.7204 vs 0.1576.

After that, stop asking “which encoder.” Ask: where does production diverge from the reference? For multimodal routing, **reference comparison should be the first diagnostic**. The embedding is policy evidence. Opposite orientation → every downstream layer can be logically correct and operationally wrong.

## What was actually broken

Drift in the **Candle path**, not the weights. Three cuts:

1. **Pooling head wrong.** `SigLIPVisionEncoder::forward` in `candle-binding/src/model_architectures/embedding/multimodal_embedding.rs` was doing BERT-style mean + Linear + tanh. SigLIP uses an attentional probe pooling head. [PR #1927](https://github.com/vllm-project/semantic-router/pull/1927) mirrors SigLIP multi-head attention pooling in Candle binding.

2. **Normalization incomplete.** Go image loader produced CHW float32 pixels in `[0, 1]`. SigLIP expects per-channel `(x - 0.5) / 0.5`. [PR #1928](https://github.com/vllm-project/semantic-router/pull/1928) applies that in the Rust encoder path.

3. **Preprocess residual.** Old Go-side resize: 4-tap bilinear. PyTorch reference: PIL-style via `SiglipProcessor`. [PR #1943](https://github.com/vllm-project/semantic-router/pull/1943) moves decode, resize, and CHW float32 conversion into Rust (`image` crate, Catmull-Rom) to approximate PIL bicubic + antialias.

![hardening arc](../../../../assets/vllm/blog/serving/semantic-router-vision/06-hardening-arc.png)

**Figure 6.** Three PRs: pooling, normalization, preprocess-in-Rust.

Easy to miss in a cross-language stack. Go, Rust FFI, Candle, and PyTorch can each look reasonable and still break the route end to end.

## Validation status

Numbers below: PR branch stack for [#1927](https://github.com/vllm-project/semantic-router/pull/1927), [#1928](https://github.com/vllm-project/semantic-router/pull/1928), [#1943](https://github.com/vllm-project/semantic-router/pull/1943). Until all three merge: **branch-stack validation**, not released production behavior.

Three-vector isolation on the canonical passport fixture (`inrule_identifier_passport.jpg`):

| Comparison | Cosine | Max abs diff | What it isolates |
|---|---:|---:|---|
| Python vs Candle-PIL | **0.999989** | 0.000911 | Model-forward only |
| Candle-PIL vs Candle-Go | **0.999916** | 0.001992 | Preprocessing only |
| Python vs Candle-Go | **0.999902** | 0.002120 | Full branch-stack pipeline |

First row: Rust model-forward matches PyTorch at fp32-level noise. Remaining drift after the first two fixes lived in preprocessing — why moving preprocess across the FFI boundary matters.

**20-image** corpus (identifier, ambient, code, adversarial, OOD):

- Cosine: min **0.999557**, mean **0.999919**, max **0.999978**
- **20 / 20** images at cosine **>= 0.999** vs PyTorch reference
- Pre-fix preprocessing cosine on the canonical fixture: **0.990145**

![corpus alignment](../../../../assets/vllm/blog/serving/semantic-router-vision/07-corpus-alignment.png)

**Figure 7.** Branch-stack corpus: min cosine 0.999557 vs the Python reference.

The method matters as much as the final cosine: compare production to reference, split model-forward drift from preprocess drift, then make serving use the same preprocess semantics as tests.

## What this unlocks

Once the vision path is trustworthy, images are first-class evidence, not side-channel metadata. Not merely “image requests go to an image model.” Text and image in the same fabric:

| Combined signal pattern | Example decision |
|---|---|
| Clinical text + clinical image + PHI/PII signal | Route to a protected medical VLM path with privacy plugins enabled |
| Generic text + identifier image | Block, redact, or route to an identity-document handling policy before model invocation |
| Code/security prompt + code screenshot | Route to a security-specialized model and keep jailbreak checks on the original request |
| In-domain text + out-of-domain image | Ask for clarification or reject the image evidence instead of forcing a bad route |

Iris made decisions composable. Athena added a stronger model stack, selection, memory, replay, richer signals. Multimodal extends the same architecture from language-only control to **request-level** control.

Public demo named: [shrader.dev](https://shrader.dev). At post time it shows the **text-routing** version of the policy pattern: domain relevance, privacy-sensitive routing, blocked outcomes **before** model invocation. Policy shape before images are added.

![cyclotron demo](../../../../assets/vllm/blog/serving/semantic-router-vision/08-cyclotron-demo.png)

**Figure 8.** Text-routing demo of the same policy shape (shrader.dev).

Text path also shows a latency property that matters once images join: classifier signals can run concurrently through `runSignalDispatchers`, so wall-clock is bounded by the **slowest enabled classifier**, not the sum. Representative trace on the page: full classification decision **~1.3s on CPU**.

![parallel dispatch](../../../../assets/vllm/blog/serving/semantic-router-vision/09-parallel-dispatch.png)

**Figure 9.** Parallel signal dispatch: wall clock follows the slowest classifier (~1.3s CPU in their trace).

Multimodal is not a separate product path. Same policy engine, larger evidence surface. Image and text should be extracted, validated, composed, replayed, and audited through the same routing semantics. If VSR routes on visual evidence, the vision path has to be boringly reliable: match the reference, survive Go/Rust/Candle, stay testable as policies get more expressive.

## What comes next

Land and review the hardening PRs; keep the validation corpus in the loop. Make reference-driven checks normal for multimodal serving. Then architectural:

- expose image-derived signals in the same decision layer as text
- keep multimodal decisions visible in replay, metrics, debugging
- make model selection aware of policy fit **and** modality capability
- preserve high-fidelity inspection for safety-critical signals (PII, jailbreak)
- extend the fabric toward agentic workflows: tool calls, memory writes, model invocations through one decision layer

Text routing was the first control surface. Multimodal is the next. Goal: not a one-off visual classifier beside the router, but every meaningful part of a request available to the same programmable routing brain.

![next steps](../../../../assets/vllm/blog/serving/semantic-router-vision/10-next-steps.png)

**Figure 10.** Next: image signals in the same decision layer, replay, modality-aware selection.

Getting started: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router), live demo [shrader.dev](https://shrader.dev).

## Acknowledgments

Huamin Chen for the mmEL pointer that broke the encoder-upgrade misdiagnosis; maintainer reviews on [#1927](https://github.com/vllm-project/semantic-router/pull/1927), [#1928](https://github.com/vllm-project/semantic-router/pull/1928), [#1943](https://github.com/vllm-project/semantic-router/pull/1943); invitation to write this up. Broader maintainer team: multi-modal classifier work this arc plugs into, the `multi-modal-embed-small` model card, and the Candle-binding integration.
