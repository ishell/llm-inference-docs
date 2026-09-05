---
source: https://vllm.ai/blog/2025-12-14-halugate
lang: en
fetched: 2026-09-04
---

# HaluGate: the tool was right; the model still lied

Chinese: [zh/vllm/blog/serving/halugate.md](../../../../zh/vllm/blog/serving/halugate.md)

2025-12-14. **vLLM Semantic Router Team**. Repo: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router). Docs: [vllm-semantic-router.com](https://vllm-semantic-router.com). Plugin on [Iris](semantic-router-iris.md); signal type on [signal-decision](semantic-router-signal.md). Launch: [semantic-router.md](semantic-router.md). Classifier kernel: [modular LoRA](semantic-router-modular.md). Do not confuse with in-engine structured decode, or with the P/D [router](router.md). Slack `#semantic-router`. Study note; latency and F1 numbers are theirs.

**TL;DR from the page:** conditional token-level hallucination detection against tool / RAG context. No LLM-as-judge. Three models on native Rust/Candle: Sentinel (~**12 ms** CPU, **96.4%** val acc) → Detector + NLI Explainer. Full path **76–162 ms** when detection runs. Token-level hallucinated-class F1 alone **59%**; a unified 5-class head was **21.7%** F1 — they keep the two-model ensemble. Not intrinsic hallucination; no context → unverified headers, not a silent pass.

Local figures (copyright remains with the original site; study copies):

![halugate 0](../../../../assets/vllm/blog/serving/halugate/01-halugate-0.png)

## The problem: hallucinations block production

The page calls hallucination the single biggest barrier to production LLMs. Same pattern across industries: **legal** (fabricated case citations), **healthcare** (incorrect drug interactions), **finance** (invented financial data), **customer service** (non-existent policies). Plausible, authoritative text that fails under scrutiny.

The hard case is not obvious nonsense. It is *subtle fabrications inside otherwise accurate responses* — errors that need domain expertise or external verification. For enterprises, that uncertainty is a liability.

## The scenario: tools work, the model does not

Typical function-calling interaction on the page:

> **User**: "When was the Eiffel Tower built?"
>
> **Tool Call**: `get_landmark_info("Eiffel Tower")`
>
> **Tool Response**: `{"name": "Eiffel Tower", "built": "1887-1889", "height": "330 meters", "location": "Paris, France"}`
>
> **LLM Response**: "The Eiffel Tower was **built in 1950** and stands at **500 meters** tall in Paris, France."

The tool was correct. Two of the model's "facts" are **extrinsic hallucinations** that contradict the provided context.

Why this mode is nasty:

- **Users trust it** because they saw the tool call.
- **Traditional filters miss it** — no toxic or harmful content.
- **Evaluation is expensive** if another LLM is the judge.

The page's question: detect this automatically, in real time, with millisecond-class latency.

## The insight: function calling as ground truth

Modern function-calling APIs already supply grounding context. Factual questions trigger tools — DB lookups, APIs, document retrieval. Those results are treated as semantically equivalent to RAG retrieved documents.

**Figure.** Grounding is already in the API flow: tool output as context, user as question, assistant as claims.

No separate retrieval stack. No GPT-4 judge. Three components from the existing flow:

| Component | Source | Purpose |
| --- | --- | --- |
| **Context** | Tool message content | Ground truth for verification |
| **Question** | User message | Intent understanding |
| **Answer** | Assistant response | Claims to verify |

The question becomes: **is the answer faithful to the context?**

## Why not LLM-as-judge?

The obvious extra-LLM verifier has production problems:

| Approach | Latency | Cost | Explainability |
| --- | --- | --- | --- |
| GPT-4 as judge | 2–5 seconds | $0.01–0.03/request | Low (black box) |
| Local LLM judge | 500 ms–2 s | GPU compute | Low |
| **HaluGate** | **76–162 ms** | **CPU only** | **High (token-level + NLI)** |

LLM judges also: **position bias**, **verbosity bias**, **self-preference**, **inconsistency** (same input, different judgments). They wanted faster, cheaper, more explainable.

## HaluGate: a conditional two-stage pipeline

Efficiency vs precision. Not every query pays for token-level detection.

![halugate 1](../../../../assets/vllm/blog/serving/halugate/02-halugate-1.png)

**Figure.** Two-stage pipeline: Sentinel first, then Detector + NLI only when the prompt is fact-seeking.

### Stage 1: HaluGate Sentinel (prompt classification)

Not every query needs hallucination detection:

| Prompt | Needs fact-check? | Reason |
| --- | --- | --- |
| "When was Einstein born?" | Yes | Verifiable fact |
| "Write a poem about autumn" | No | Creative task |
| "Debug this Python code" | No | Technical assistance |
| "What's your opinion on AI?" | No | Opinion request |
| "Is the Earth round?" | Yes | Factual claim |

Token-level detection on creative writing or code review is wasteful and can false-positive ("your poem contains unsupported claims").

**Why pre-classification matters:** token-level detection scales linearly with context length. **4K**-token RAG context → ~**125 ms**; **16K** tokens → ~**365 ms**. In production workloads where ~**35%** of queries are non-factual, pre-classification claims a **72.2% efficiency gain** — skip expensive detection for creative, coding, and opinion queries.

[HaluGate Sentinel](https://huggingface.co/llm-semantic-router/halugate-sentinel) is a ModernBERT-based classifier: *does this prompt warrant factual verification?*

![halugate 2](../../../../assets/vllm/blog/serving/halugate/03-halugate-2.png)

**Figure.** Sentinel binary: FACT_CHECK_NEEDED vs not.

Training mix on the page:

**Fact-check needed (positive):**

- **Question answering:** SQuAD, TriviaQA, Natural Questions, HotpotQA
- **Truthfulness:** TruthfulQA (common misconceptions)
- **Hallucination benchmarks:** HaluEval, FactCHD
- **Information-seeking dialogue:** FaithDial, CoQA
- **RAG:** neural-bridge/rag-dataset-12000

**No fact-check needed (negative):**

- **Creative writing:** WritingPrompts, story generation
- **Code:** CodeSearchNet docstrings, programming tasks
- **Opinion / instruction:** Dolly non-factual, Alpaca creative

Binary classification: **96.4%** validation accuracy, **~12 ms** inference via native Rust/Candle.

### Stage 2: token-level detection + NLI explanation

For fact-seeking prompts, a two-model detection pipeline.

#### Token-level hallucination detection

Not a sentence-level "hallucinated / not" label. **Token-level** marks *which* tokens are unsupported by context.

![halugate 3](../../../../assets/vllm/blog/serving/halugate/04-halugate-3.png)

**Figure.** Per-answer-token labels: 0 = supported, 1 = hallucinated.

Architecture:

```text
Input: [CLS] context [SEP] question [SEP] answer [SEP]
                                          ↓
                              ModernBERT Encoder
                                          ↓
                    Token Classification Head (Binary per token)
                                          ↓
              Label: 0 = Supported, 1 = Hallucinated (for answer tokens only)
```

Design choices:

- **Answer-only classification:** only answer-segment tokens, not context or question
- **Span merging:** consecutive hallucinated tokens merged into spans
- **Confidence thresholding:** configurable; default **0.8** for precision/recall

#### NLI explanation layer

Knowing *that* is not enough. NLI classifies each detected span against the context:

![halugate 4](../../../../assets/vllm/blog/serving/halugate/05-halugate-4.png)

**Figure.** Per-span NLI: CONTRADICTION / NEUTRAL / ENTAILMENT.

| NLI label | Meaning | Severity | Action |
| --- | --- | --- | --- |
| **CONTRADICTION** | Claim conflicts with context | 4 (High) | Flag as error |
| **NEUTRAL** | Claim not supported by context | 2 (Medium) | Flag as unverifiable |
| **ENTAILMENT** | Context supports the claim | 0 | Filter false positive |

**Why the ensemble:** token-level detection alone **59% F1** on the hallucinated class — nearly half of hallucinations missed, about one-third of flags false positives. A unified 5-class model (SUPPORTED / CONTRADICTION / FABRICATION / etc.) got only **21.7% F1** — token-level classification cannot distinguish *why*. Two-stage: LettuceDetect-style recall, NLI for precision and explainability.

## Integration with Signal-Decision

HaluGate is a new signal type and a plugin on the [signal-decision](semantic-router-signal.md) spine. Ships with [Iris](semantic-router-iris.md).

### `fact_check` as a signal type

Alongside keyword, embedding, and domain: `fact_check` is first-class.

![halugate 5](../../../../assets/vllm/blog/serving/halugate/06-halugate-5.png)

**Figure.** `fact_check` conditions a decision; the `hallucination` plugin attaches to that decision.

> **Note on the page:** even frontier models show hallucination variance between releases. [GPT-5.2's system card](https://cdn.openai.com/pdf/3a4153c8-c748-4b71-8e31-aecbde944f8d/oai_5_2_system-card.pdf) is cited as showing a measurable hallucination delta vs previous versions — continuous verification regardless of model sophistication.

```yaml
decisions:
  - name: "factual-query-with-verification"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "fact_check"
          name: "needs_fact_check"
        - type: "domain"
          name: "general"
    plugins:
      - type: "hallucination"
        configuration:
          enabled: true
          use_nli: true
          hallucination_action: "header"
```

### Request–response context propagation

Classification is **request time**; detection is **response time**. State has to cross that boundary.

![halugate 6](../../../../assets/vllm/blog/serving/halugate/07-halugate-6.png)

**Figure.** `RequestContext` carries classification, tool context, then detection results.

```yaml
RequestContext:
  # Classification results (set at request time)
  FactCheckNeeded: true
  FactCheckConfidence: 0.87

  # Tool context (extracted at request time)
  HasToolsForFactCheck: true
  ToolResultsContext: "Built 1887-1889, 330 meters..."
  UserContent: "When was the Eiffel Tower built?"

  # Detection results (set at response time)
  HallucinationDetected: true
  HallucinationSpans: ["1950", "500 meters"]
  HallucinationConfidence: 0.92
```

Those numbers in the YAML sketch are the page's worked example (Eiffel), not a benchmark mean.

### The `hallucination` plugin

Per-decision configuration:

```yaml
plugins:
  - type: "hallucination"
    configuration:
      enabled: true
      use_nli: true  # Enable NLI explanations

      # Action when hallucination detected
      hallucination_action: "header"  # "header" | "body" | "block" | "none"

      # Action when fact-check needed but no tool context
      unverified_factual_action: "header"

      # Include detailed info in response
      include_hallucination_details: true
```

| Action | Behavior |
| --- | --- |
| `header` | Add warning headers, pass response through |
| `body` | Inject warning into response body |
| `block` | Return error response, don't forward LLM output |
| `none` | Log only, no user-visible action |

## Response headers

Detection results as HTTP headers for downstream policy:

```http
HTTP/1.1 200 OK
Content-Type: application/json
x-vsr-fact-check-needed: true
x-vsr-hallucination-detected: true
x-vsr-hallucination-spans: 1950; 500 meters
x-vsr-nli-contradictions: 2
x-vsr-max-severity: 4
```

Unverified factual (tools not available):

```http
HTTP/1.1 200 OK
x-vsr-fact-check-needed: true
x-vsr-unverified-factual-response: true
x-vsr-verification-context-missing: true
```

Headers enable: **UI disclaimers**, **human review queues**, **audit logging**, **conditional blocking** of high-severity contradictions.

## Three paths

![halugate 7](../../../../assets/vllm/blog/serving/halugate/08-halugate-7.png)

**Figure.** Path 1 skip, Path 2 unverified headers, Path 3 full detection.

| Path | Condition | Latency added | Action |
| --- | --- | --- | --- |
| **Path 1** | Non-factual prompt | ~12 ms (classifier only) | Pass through |
| **Path 2** | Factual + no tools | ~12 ms | Add warning headers |
| **Path 3** | Factual + tools available | 76–162 ms | Full detection + headers |

## Model architecture

Three models:

![halugate 8](../../../../assets/vllm/blog/serving/halugate/09-halugate-8.png)

**Figure.** Sentinel / Detector / Explainer, all ModernBERT-base family.

### HaluGate Sentinel: binary prompt classification

**Architecture:** ModernBERT-base + LoRA adapter + binary classification head

**Training:**

- **Base:** `answerdotai/ModernBERT-base`
- **Fine-tuning:** LoRA (rank=16, alpha=32, dropout=0.1)
- **Data:** 50,000 samples from 14 datasets
- **Loss:** CrossEntropy with class weights (imbalance)
- **Optimization:** AdamW, lr=2e-5, 3 epochs

**Inference:**

- **Input:** raw prompt text
- **Output:** (class_id, confidence)
- **Latency:** ~12 ms on CPU

LoRA: only **2.2%** of parameters updated (**3.4M** of **149M**).

### HaluGate Detector: token-level binary classification

**Architecture:** ModernBERT-base + token classification head

**Input format:**

```text
[CLS] The Eiffel Tower was built in 1887-1889 and is 330 meters tall.
[SEP] When was the Eiffel Tower built?
[SEP] The Eiffel Tower was built in 1950 and is 500 meters tall. [SEP]
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    Answer tokens (classification targets)
```

**Output:** binary label (0=Supported, 1=Hallucinated) per answer token

**Post-processing:**

1. Filter to answer segment only
2. Confidence threshold (default: 0.8)
3. Merge consecutive hallucinated tokens into spans
4. Return spans with confidence scores

### HaluGate Explainer: three-way NLI

**Architecture:** ModernBERT-base fine-tuned on NLI

**Input format:**

```text
[CLS] The Eiffel Tower was built in 1887-1889. [SEP] built in 1950 [SEP]
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^       ^^^^^^^^^^^^^^^
                    Premise (context)                Hypothesis (span)
```

**Output:**

- **ENTAILMENT** (0): context supports the claim
- **NEUTRAL** (1): cannot be determined from context
- **CONTRADICTION** (2): context conflicts with claim

**Severity mapping:**

| NLI label | Severity score | Interpretation |
| --- | ---: | --- |
| ENTAILMENT | 0 | Likely false positive — filter out |
| NEUTRAL | 2 | Claim is unverifiable |
| CONTRADICTION | 4 | Direct factual error |

## Why native Rust / Candle

All three models via **Candle** (Hugging Face's Rust ML framework) with CGO bindings to Go:

![halugate 9](../../../../assets/vllm/blog/serving/halugate/10-halugate-9.png)

**Figure.** In-process Candle; no Python sidecar.

| Aspect | Python (PyTorch) | Native (Candle) |
| --- | --- | --- |
| **Cold start** | 5–10 s | <500 ms |
| **Memory** | 2–4 GB per model | 500 MB–1 GB per model |
| **Latency** | +50–100 ms overhead | Near-zero overhead |
| **Deployment** | Python runtime required | Single binary |
| **Scaling** | GIL contention | True parallelism |

No separate Python service, sidecars, or model servers — in-process. Same Candle door as [modular LoRA](semantic-router-modular.md).

### Latency breakdown

Measured production-pipeline components (their numbers):

| Component | P50 | P99 | Notes |
| --- | ---: | ---: | --- |
| Fact-check classifier | 12 ms | 28 ms | ModernBERT inference |
| Tool context extraction | 1 ms | 3 ms | JSON parsing |
| Hallucination detector | 45 ms | 89 ms | Token classification |
| NLI explainer | 18 ms | 42 ms | Per-span classification |
| **Total overhead** | **76 ms** | **162 ms** | When detection runs |

They call **76–162 ms** negligible vs typical LLM generation (**5–30 seconds**), so practical for synchronous request processing. Trust their measurement; not your SLA.

## Configuration reference

```yaml
# Model configuration
hallucination_mitigation:
  # Stage 1: Prompt classification
  fact_check_model:
    model_id: "models/halugate-sentinel"
    threshold: 0.6  # Confidence threshold for FACT_CHECK_NEEDED
    use_cpu: true

  # Stage 2a: Token-level detection
  hallucination_model:
    model_id: "models/halugate-detector"
    threshold: 0.8  # Token confidence threshold
    use_cpu: true

  # Stage 2b: NLI explanation
  nli_model:
    model_id: "models/halugate-explainer"
    threshold: 0.9  # NLI confidence threshold
    use_cpu: true

# Signal rules for fact-check classification
fact_check_rules:
  - name: needs_fact_check
    description: "Query contains factual claims that should be verified"
  - name: no_fact_check_needed
    description: "Query is creative, code-related, or opinion-based"

# Decision with hallucination plugin
decisions:
  - name: "verified-factual"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "fact_check"
          name: "needs_fact_check"
    plugins:
      - type: "hallucination"
        configuration:
          enabled: true
          use_nli: true
          hallucination_action: "header"
          unverified_factual_action: "header"
          include_hallucination_details: true
```

Thresholds on the page: Sentinel **0.6**, Detector **0.8**, NLI **0.9**.

## Offline evaluation framework

Same pipeline can score models offline. Feed benchmark datasets through detection instead of intercepting live requests.

![halugate 10](../../../../assets/vllm/blog/serving/halugate/11-halugate-10.png)

**Figure.** HaluGate as a hallucination scorer on QA / RAG datasets.

Workflow:

1. **Load dataset:** TriviaQA, Natural Questions, HotpotQA, or custom enterprise context–question pairs
2. **Generate responses:** model under test, with provided context
3. **Detect:** (context, query, response) through HaluGate Detector
4. **Classify severity:** HaluGate Explainer on each flagged span
5. **Aggregate:** hallucination rates, contradiction ratios, per-category breakdowns

The page does not publish a HaluGate-scored leaderboard of named LLMs.

## Limitations and scope

Targets **extrinsic hallucinations** — tool / RAG context is the ground. Known limits:

### What HaluGate cannot detect

| Limitation | Example | Reason |
| --- | --- | --- |
| **Intrinsic hallucinations** | Model says "Einstein was born in 1900" without any tool call | No context to verify against |
| **No-context scenarios** | User asks a factual question, no tools defined | Missing ground truth |

### Transparent degradation

Fact-seeking but no tool context: flag as "unverified factual", do not silently pass:

```http
x-vsr-fact-check-needed: true
x-vsr-unverified-factual-response: true
x-vsr-verification-context-missing: true
```

## Acknowledgments

- **Token-level architecture:** [LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) (KRLabs) — ModernBERT-based hallucination detection
- **NLI:** [tasksource/ModernBERT-base-nli](https://huggingface.co/tasksource/ModernBERT-base-nli)
- **Training datasets:** TruthfulQA, HaluEval, FaithDial, RAGTruth, and other public benchmarks

## Conclusion

Claims on the page:

- **Conditional verification:** skip non-factual, verify factual
- **Token-level precision:** which claims are unsupported
- **Explainable results:** NLI says *why*
- **Zero-latency integration:** native Rust, no Python sidecars (their slogan; measured overhead is the table above)
- **Actionable transparency:** headers for downstream policy

Next time the LLM calls a tool, gets accurate data, and still lies — they want HaluGate to catch it before users do.

**Resources:** [signal-decision post](https://blog.vllm.ai/2025/11/19/signal-decision.html) (note: [semantic-router-signal.md](semantic-router-signal.md)), [GitHub](https://github.com/vllm-project/semantic-router), [docs](https://vllm-semantic-router.com). Slack `#semantic-router`.
