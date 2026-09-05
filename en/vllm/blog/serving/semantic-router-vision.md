---
source: https://vllm.ai/blog/2026-05-28-vllm-sr-vision-encoder-hardening
lang: en
fetched: 2026-09-04
---

# Vision signals: not a bigger encoder — Candle missed the reference

Chinese: [zh/vllm/blog/serving/semantic-router-vision.md](../../../../zh/vllm/blog/serving/semantic-router-vision.md)

2026-05-28. **David Shrader, Huamin Chen, Xunzhuo Liu, Bowei He, and the vLLM Semantic Router Team**. Repo: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router). Launch: [semantic-router.md](semantic-router.md). Spine: [Iris](semantic-router-iris.md) / [signal-decision](semantic-router-signal.md). Follows Athena’s [`multi-modal-embed-small`](semantic-router-athena.md). Candle kernel family: [modular](semantic-router-modular.md). Later: [themis](semantic-router-themis.md), [session](semantic-router-session.md), [fusion](semantic-router-fusion.md), [micro-agent](semantic-router-micro-agent.md). Do not confuse with the in-engine [router.md](router.md). Cosines and inversion rates are **their** probe / PR-branch numbers. Until merge, PR validation — not a production guarantee.

Siblings: [amd](semantic-router-amd.md), [mom-amd](semantic-router-mom-amd.md), [mom](semantic-router-mom.md), [halugate](halugate.md).

Most routers take a prompt and pick an endpoint. VSR’s bet: extract signals, compose them into decisions, make the path observable and programmable **before** the serving model. That started with text. The next boundary is multimodal: once an image / screenshot / scan / page is in the request, the router is reasoning over **request evidence**, not a prompt alone. The image may be what makes the request clinical, regulated, security-sensitive, out of domain, or worth a stronger VLM. Text-only routing is routing a **partial** request.

The important step is not “add an image encoder.” It is turning visual evidence into a **trustworthy VSR signal** in the same fabric as text. A deployed path around `multi-modal-embed-small` looked **confidently wrong**. First guess: compact encoder too weak. Actual issue: the **Rust/Candle path did not match the PyTorch reference** for the same weights.

Local figures (copyright remains with the original site; study copies):

![hero](../../../../assets/vllm/blog/serving/semantic-router-vision/01-hero.png)

**Figure 1.** Visual evidence has to become a typed signal, not a side-channel embedding.

## Multimodal routing is not image classification

Text-only routing already does more than topic matching. Signals are independent observations; decisions compose them with priority and boolean logic; plugins / model refs say what happens next. That is how “security-sensitive code review gets a stronger model and jailbreak checks” instead of “computer science → coding model.”

Multimodal keeps the same shape; the unit of analysis becomes the **full request**. Text can be generic while the image carries the decisive evidence:

| Request evidence | Text-only router sees | Multimodal router should see |
| --- | --- | --- |
| "Summarize this" + passport image | Generic summarization | Identifier document, PII risk, restricted handling |
| "What does this show?" + chest X-ray | Vague visual question | Clinical image, medical-domain policy, capable VLM |
| "Find the bug" + code screenshot | Coding request | Code artifact, possible secret leakage, security review |
| Medical prompt + unrelated car image | Medical text | Out-of-domain visual evidence, clarify or reject |

The image embedding becomes a **typed signal** beside text intent, PII, jailbreak, domain, similarity, plugins, model selection. Prompt-level routing → **request-level policy**.

![policy layer](../../../../assets/vllm/blog/serving/semantic-router-vision/02-policy-layer.png)

**Figure 2.** Same Signal-Decision fabric; larger evidence surface.

If a text signal is wrong, you route to the wrong model or skip a plugin. If a vision signal is **anti-correlated**, the router can be confidently wrong with a clean audit trail for the wrong decision. Reference parity is a **control-plane invariant**.

## When the vision signal was confidently wrong

11-image probe, three verticals, 21 candidate labels: deployed `multi-modal-embed-small` (mmes) ranked the **wrong vertical** highest on **9 of 11**. Medical X-rays closer to semiconductor candidates than medical. Identifier documents did not land near identifier anchors. **82% inversion** — anti-correlated, not merely noisy.

![inversion heatmap](../../../../assets/vllm/blog/serving/semantic-router-vision/03-inversion-heatmap.png)

**Figure 3.** Inverted rankings: confidence in the wrong direction.

A weak classifier usually looks uncertain. An inverted one looks sure. For a policy layer that can be worse than **no** image signal. Surface that exposed it: image-modality routing around mmes, including the E2E profile in [PR #1881](https://github.com/vllm-project/semantic-router/pull/1881). Real images through the Candle binding made the gap visible.

## Tempting explanation: upgrade the encoder

Natural first hypothesis: compact encoder not strong enough. The team was already looking at SigLIP2 and larger `multi-modal-embed-large` (mmEL). Direct test on the same 21-candidate probe:

- SigLIP2-base: **10/10**
- SigLIP-base via Hugging Face Transformers: **10/10**
- mmEL (vision tower based on SigLIP2): **10/10**
- mmes via the **PyTorch reference** path: **10/10**

![encoder eliminated](../../../../assets/vllm/blog/serving/semantic-router-vision/04-encoder-eliminated.png)

**Figure 4.** The family was fine; even “failing” mmes was fine on the reference loader.

Sideline: larger SigLIP2-so400m showed stronger OOD rejection in this probe (suppressed an accidental car-engine image more aggressively). Maybe useful later if memory allows a larger tower. **Not** the production inversion bug.

## The reference check that changed the investigation

Same mmes, same passport fixture, two paths. PyTorch reference cosine **0.7204** vs the passport anchor. Deployed Candle-binding: **0.1576**. A **5–8×** magnitude gap on the same model and fixture.

![diagnostic gap](../../../../assets/vllm/blog/serving/semantic-router-vision/05-diagnostic-gap.png)

**Figure 5.** Same weights, two loaders: the production path had drifted.

After that, stop asking “which encoder?” Ask: where does production diverge from the model-card reference? For multimodal routing, **reference comparison should be the first diagnostic**. The embedding is policy evidence, not only retrieval.

## What was actually broken

Implementation drift in Candle, not the weights. Three cuts:

1. **Pooling head.** `SigLIPVisionEncoder::forward` in `candle-binding/src/model_architectures/embedding/multimodal_embedding.rs` was doing BERT-style mean + Linear + tanh. SigLIP uses an attentional probe. [PR #1927](https://github.com/vllm-project/semantic-router/pull/1927) mirrors SigLIP multi-head attention pooling.
2. **Normalization.** Go image loader produced CHW float32 in `[0, 1]`. SigLIP expects `(x - 0.5) / 0.5`. [PR #1928](https://github.com/vllm-project/semantic-router/pull/1928) applies that in the Rust encoder path.
3. **Preprocess residual.** Old Go resize: 4-tap bilinear. PyTorch reference: PIL-style via `SiglipProcessor`. [PR #1943](https://github.com/vllm-project/semantic-router/pull/1943) moves decode, resize, CHW float32 into Rust (`image` crate, Catmull-Rom ≈ PIL bicubic + antialias).

![hardening arc](../../../../assets/vllm/blog/serving/semantic-router-vision/06-hardening-arc.png)

**Figure 6.** Pooling, normalization, then preprocess across the FFI boundary.

Easy to miss in a cross-language stack: Go, Rust FFI, Candle, and PyTorch can each look reasonable while the e2e route is broken.

## Validation status

Numbers below are from the **PR branch stack** for #1927 / #1928 / #1943. Until all three merge, read as branch-stack validation, not released production behavior.

Three-vector isolation on the canonical passport fixture (`inrule_identifier_passport.jpg`):

| Comparison | Cosine | Max abs diff | What it isolates |
| --- | ---: | ---: | --- |
| Python vs Candle-PIL | **0.999989** | 0.000911 | Model-forward only |
| Candle-PIL vs Candle-Go | **0.999916** | 0.001992 | Preprocessing only |
| Python vs Candle-Go | **0.999902** | 0.002120 | Full branch-stack pipeline |

First row: Rust model-forward can match PyTorch at fp32-level noise. Remaining drift after the first two fixes lived in preprocessing — why moving preprocess across the FFI boundary matters.

20-image corpus (identifier, ambient, code, adversarial, OOD):

- Cosine: min **0.999557**, mean **0.999919**, max **0.999978**
- **20 / 20** images at cosine ≥ 0.999 vs PyTorch
- Pre-fix preprocessing cosine on the canonical fixture: **0.990145**

![corpus alignment](../../../../assets/vllm/blog/serving/semantic-router-vision/07-corpus-alignment.png)

**Figure 7.** Isolation method: split model-forward drift from preprocess drift.

## What this unlocks

Once the vision path is trustworthy, images are first-class evidence, not side-channel metadata. Not only “image requests → image model”:

| Combined signal pattern | Example decision |
| --- | --- |
| Clinical text + clinical image + PHI/PII | Protected medical VLM path; privacy plugins on |
| Generic text + identifier image | Block, redact, or identity-document policy **before** invocation |
| Code/security prompt + code screenshot | Security-specialized model; jailbreak checks on the original request |
| In-domain text + OOD image | Clarify or reject the image evidence instead of forcing a bad route |

Iris made decisions composable. Athena made the router more strategic. Multimodal extends language-only control to **request-level** control.

Public demo: [shrader.dev](https://shrader.dev) — today the **text-routing** version of the policy pattern (domain relevance, privacy-sensitive routing, blocked outcomes before invocation). Shows the policy shape before images are added.

![cyclotron demo](../../../../assets/vllm/blog/serving/semantic-router-vision/08-cyclotron-demo.png)

**Figure 8.** Text policy demo of the same control shape.

Classifier signals can run concurrently through `runSignalDispatchers`; wall-clock is bounded by the **slowest** enabled classifier, not the sum. Representative trace: full classification decision ~**1.3 s** on CPU (theirs).

![parallel dispatch](../../../../assets/vllm/blog/serving/semantic-router-vision/09-parallel-dispatch.png)

**Figure 9.** Parallel signal dispatch; wall clock eats the slowest classifier.

Multimodal is the same policy engine with a larger evidence surface. Image and text signals should be extracted, validated, composed, replayed, and audited through the same routing semantics. If VSR routes on visual evidence, the vision path has to be boringly reliable.

## What comes next

Land and review the hardening PRs; keep the validation corpus in the loop. Then: expose image-derived signals in the same decision layer; keep multimodal decisions visible in replay / metrics / debug; model selection aware of policy fit **and** modality capability; high-fidelity inspection for PII and jailbreak; extend toward agentic workflows where tool calls, memory writes, and invocations share one decision layer.

![next steps](../../../../assets/vllm/blog/serving/semantic-router-vision/10-next-steps.png)

**Figure 10.** Text was the first control surface; multimodal is the next — not a one-off visual classifier beside the router.

- Repo: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
- Live demo: [shrader.dev](https://shrader.dev)

## Acknowledgments

Huamin Chen for the mmEL pointer that broke the encoder-upgrade misdiagnosis; maintainer reviews on #1927 / #1928 / #1943; invitation to write this up; broader maintainer work on multi-modal classifiers, the `multi-modal-embed-small` card, and Candle-binding.
