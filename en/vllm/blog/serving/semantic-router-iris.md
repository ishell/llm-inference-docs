---
source: https://vllm.ai/blog/2026-01-05-vllm-sr-iris
lang: en
fetched: 2026-09-04
---

# vLLM Semantic Router v0.1 Iris: The First Major Release

Chinese: [zh/vllm/blog/serving/semantic-router-iris.md](../../../../zh/vllm/blog/serving/semantic-router-iris.md)

2026-01-05. **vLLM Semantic Router Team**. Repo: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router). Launch: [semantic-router.md](semantic-router.md). Signal / decision spine: [semantic-router-signal.md](semantic-router-signal.md). LoRA on the classifier kernel: [semantic-router-modular.md](semantic-router-modular.md). HaluGate write-up: [halugate.md](halugate.md). Do not confuse with the in-engine [router.md](router.md). Community counts and architecture diagrams are a release snapshot.

Later notes: [mom-amd](semantic-router-mom-amd.md), [athena](semantic-router-athena.md), [vision](semantic-router-vision.md), [session](semantic-router-session.md), [themis](semantic-router-themis.md), [fusion](semantic-router-fusion.md), [micro-agent](semantic-router-micro-agent.md), [mom](semantic-router-mom.md). Also on AMD routing, between launch and this post: [semantic-router-amd.md](semantic-router-amd.md).

[vLLM Semantic Router](https://github.com/vllm-project/semantic-router) calls itself **system-level intelligence** for Mixture-of-Models (MoM): it sits between users and models, pulls signals from requests, responses, and context, and routes — model selection, safety (jailbreak, PII), semantic cache, hallucination detection. Background remains the [launch post](semantic-router.md).

v0.1, codename **Iris**, is the first major release. From the September 2025 experiment to this post: **600+** PRs merged, **300+** issues, **50+** engineers. Early 2026 they call the result a production-ready semantic routing platform.

Local figures (copyright remains with the original site; study copies):

![iris 0](../../../../assets/vllm/blog/serving/semantic-router-iris/01-iris-0.png)

## Why Iris?

In Greek myth, Iris (Ἶρις) is the messenger between gods and mortals, traveling the rainbow. The page uses that for v0.1: **a bridge between users and diverse AI models**, routing across providers and architectures.

![iris 1](../../../../assets/vllm/blog/serving/semantic-router-iris/02-iris-1.png)

## What’s new in v0.1

### 1. Architecture: signal → decision → plugin chain

**Before:** one classifier into **14** MMLU domains; jailbreak, PII, and semantic cache statically orchestrated.

**Now:** **Signal-Decision Driven Plugin Chain**. From 14 fixed categories to unbounded routing decisions. Internals: [semantic-router-signal.md](semantic-router-signal.md).

![iris 2](../../../../assets/vllm/blog/serving/semantic-router-iris/03-iris-2.png)

Six signal types from a query:

- **Domain:** MMLU-trained classification, LoRA-extensible
- **Keyword:** fast, interpretable regex
- **Embedding:** semantic similarity on neural embeddings
- **Factual:** fact-check classification for hallucination detection
- **Feedback:** user satisfaction / dissatisfaction
- **Preference:** user-defined preferences

Signals feed a decision engine: AND/OR with priority. Formerly static jailbreak, PII, and semantic cache become per-decision **plugins**:

| Plugin | Purpose |
| --- | --- |
| `semantic-cache` | Cache similar queries for cost |
| `jailbreak` | Prompt injection |
| `pii` | Sensitive information |
| `hallucination` | Real-time hallucination detection |
| `system_prompt` | Inject custom instructions |
| `header_mutation` | Mutate HTTP headers for metadata |

New signals, plugins, and model-selection algorithms without changing the spine.

### 2. Performance: modular LoRA

With the **Hugging Face Candle** team, they refactored the router’s inference kernel. Previously each classification task loaded and ran independently — cost grew linearly with the number of tasks. Details: [semantic-router-modular.md](semantic-router-modular.md).

![iris 3](../../../../assets/vllm/blog/serving/semantic-router-iris/04-iris-3.png)

**The breakthrough:** LoRA shares base-model compute across classification tasks:

| Approach | Workload | Scalability |
| --- | --- | --- |
| Before | N full model forwards | O(n) |
| After | 1 base pass + N light LoRA adapters | O(1) + O(n×ε) |

> **Note:** ε is the cost of one LoRA forward relative to the full base — typically ε ≪ 1, so the extra overhead is small.

They claim a significant latency drop and multi-task classification on the same input.

### 3. Safety: HaluGate

Request-time safety already had jailbreak and PII. v0.1 adds **HaluGate** — a three-stage hallucination pipeline on **responses**. Write-up: [halugate.md](halugate.md).

**Stage 1: Sentinel.** Binary: does this query need factual verification (creative writing and code do not).

**Stage 2: Detector.** Token-level: which tokens in the response are unsupported by the provided context.

**Stage 3: Explainer.** NLI: *why* each flagged span is a problem (CONTRADICTION vs NEUTRAL).

![iris 4](../../../../assets/vllm/blog/serving/semantic-router-iris/05-iris-4.png)

Hooks into function calling: tool results are ground truth. Verdicts ride HTTP headers; downstream blocks or labels.

### 4. UX: one-command install

**Local:**

```bash
pip install vllm-sr
```

![iris 7](../../../../assets/vllm/blog/serving/semantic-router-iris/06-iris-7.png)

The package includes core dependencies for quickstart. After install, `vllm-sr init` writes a default `config.yaml`; then fill `providers`:

```yaml
providers:
  models:
    - name: "openai/gpt-oss-120b"       # Local vLLM endpoint
      endpoints:
        - endpoint: "localhost:8000"
          protocol: "http"
      access_key: "your-vllm-api-key"
    - name: "openai/gpt-4"              # External provider
      endpoints:
        - endpoint: "api.openai.com"
          protocol: "https"
      access_key: "sk-xxxxxx"
  default_model: "openai/gpt-oss-120b"
```

Config docs: [installation](https://vllm-semantic-router.com/docs/installation/).

**Kubernetes:**

```bash
helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router
```

Helm charts with what they call sensible defaults and a long customization list.

**Dashboard:** a web console — routing policies, model config, an interactive chat playground to watch routing live. Routing flows, latency distributions, classification thresholds, all in the browser.

### 5. Ecosystem

**Inference frameworks:**

- [vLLM Production Stack](https://github.com/vllm-project/production-stack) (note: [production-stack.md](production-stack.md)) — reference stack: Helm, request routing, KV offload
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) — datacenter-scale distributed inference, multi-GPU / multi-node, disaggregated P/D
- [llm-d](https://github.com/llm-d/llm-d) — Kubernetes-native distributed inference across NVIDIA, AMD, Google TPU, Intel XPU
- [vLLM AIBrix](https://github.com/vllm-project/aibrix) (note: [aibrix.md](aibrix.md)) — GenAI infrastructure building blocks for scalable LLM serving

**API gateways:**

- [Envoy AI Gateway](https://github.com/envoyproxy/ai-gateway) — unified generative-AI access on Envoy Gateway
- [Istio](https://github.com/istio/istio) — service mesh: traffic, security, observability

### 6. MoM family

A suite of small models trained for semantic routing:

![iris 6](../../../../assets/vllm/blog/serving/semantic-router-iris/07-iris-6.png)

| Model | Purpose |
| --- | --- |
| `mom-domain-classifier` | MMLU-based domain classification |
| `mom-pii-classifier` | PII |
| `mom-jailbreak-classifier` | Prompt injection |
| `mom-halugate-sentinel` | Fact-check classification |
| `mom-halugate-detector` | Token-level hallucination |
| `mom-halugate-explainer` | NLI explanation |
| `mom-toolcall-sentinel` | Tool-selection classification |
| `mom-toolcall-verifier` | Tool-call verification |
| `mom-feedback-detector` | User feedback |
| `mom-embedding-x` | Semantic embeddings |

Claim on the page: trained and optimized for Semantic Router, consistent across routing scenarios. Later MoM notes: [mom](semantic-router-mom.md), [mom-amd](semantic-router-mom-amd.md).

### 7. Responses API

OpenAI **Responses API** (`/v1/responses`) with in-memory conversation state:

- **Stateful conversations:** `previous_response_id` chaining
- **Multi-turn context:** context kept across turns
- **Routing continuity:** intent-classification history follows the conversation

Routing for agent frameworks and multi-turn apps.

### 8. Tool selection

Tool management for agentic workflows:

- **Semantic tool filtering:** drop irrelevant tools before they reach the LLM
- **Context-aware selection:** conversation history and task requirements
- **Fewer tokens:** smaller catalogs, faster and cheaper inference

The launch post already warned about tool-catalog bloat; Iris makes filtering a first-class feature.

## Looking ahead: v0.2

v0.1 is the foundation. Enhancements listed for v0.2 (later release note: [athena](semantic-router-athena.md)):

![iris 5](../../../../assets/vllm/blog/serving/semantic-router-iris/08-iris-5.png)

**Signal-decision**

- More signal types
- Better accuracy on existing signals
- Signal Composer: a composition layer for complex extraction

**Model selection**

![iris 8](../../../../assets/vllm/blog/serving/semantic-router-iris/09-iris-8.png)

- ML: KNN, KMeans, MLP, SVM, matrix factorization
- Advanced: Elo, RouterDC, AutoMix, hybrids
- Graph-based: model-relationship graphs
- Size-aware: model size vs task complexity

**Out-of-box plugins**

- Memory: persistent conversation memory
- Router Replay: debug and replay routing decisions and feedback

**Multi-turn**

- Responses API: stateful backends such as Redis, Milvus, Memcached
- Context engineering: compression and memory
- RL: preference-driven model selection

**MoM**

- Pre-train a base with a longer context window for signal extraction
- Post-train an SLM for human-preference signals
- Migrate existing models onto self-trained replacements

**Safety**

- Jailbreak detection on tool calling
- Multi-turn guardrails across sessions
- Higher-precision hallucination detection

**Tool management**

- Tool completion: auto-complete definitions and calls from intent
- Finer relevance filtering

**UX and operations**

- Dashboard visualization and management
- Helm charts: more config and deployment patterns

**Evaluation**

- Router evaluation frameworks with the RouterArena team

## Acknowledgments

The page calls this a global collaboration. Named organizations: **Red Hat**, **IBM Research**, **AMD**, **Hugging Face**, and others unnamed.

Committer list (as printed):

*Senan Zedan, samzong, Liav Weiss, Asaad Balum, Yehudit, Noa Limoy, JaredforReal, Abdallah Samara, Hen Schwartz, Srinivas A, carlory, Yossi Ovadia, Jintao Zhang, yuluo-yx, cryo-zd, OneZero-Y, aeft*

Plus **50+** contributors.

## Get started

```bash
pip install vllm-sr
vllm-sr init
```

Then edit `providers` in `config.yaml` as above. On Kubernetes, the Helm line.

## Join the community

The page invites companies wiring intelligent routing into infrastructure, researchers on semantic understanding, and individual developers who care about open-source AI.

**Ways to contribute (four bullets on the page)**

- **Organizations:** integrations, sponsorship, engineering time
- **Researchers:** papers, algorithms, benchmarks
- **Developers:** PRs, issues, docs, community plugins
- **Community:** use cases, tutorials, translations, answering questions

A typo fix counts. Links:

- Docs: [vllm-semantic-router.com](https://vllm-semantic-router.com)
- GitHub: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
- Models: [Hugging Face](https://huggingface.co/llm-semantic-router)
- Slack: [vLLM Slack](https://vllm-dev.slack.com/archives/C09CTGF8KCN)

Closing line: *The rainbow bridge is now open. Welcome to Iris.*
