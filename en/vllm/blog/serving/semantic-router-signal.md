---
source: https://vllm.ai/blog/2025-11-19-signal-decision
lang: en
fetched: 2026-09-04
---

# Signal-Decision Driven Architecture: Reshaping Semantic Routing at Scale

Chinese: [zh/vllm/blog/serving/semantic-router-signal.md](../../../../zh/vllm/blog/serving/semantic-router-signal.md)

2025-11-19. **vLLM Semantic Router Team**. Repo: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router). Launch: [semantic-router.md](semantic-router.md). Ships in v0.1: [semantic-router-iris.md](semantic-router-iris.md). Classifier LoRA: [semantic-router-modular.md](semantic-router-modular.md). Later: [athena](semantic-router-athena.md), [session](semantic-router-session.md), [themis](semantic-router-themis.md), [mom](semantic-router-mom.md). Do not confuse with the in-engine [router.md](router.md). This post’s signal catalog is **three** types (keyword / embedding / domain); Iris later lists six.

Siblings: [amd](semantic-router-amd.md), [mom-amd](semantic-router-mom-amd.md), [vision](semantic-router-vision.md), [fusion](semantic-router-fusion.md), [micro-agent](semantic-router-micro-agent.md).

Earlier Semantic Router versions classified a query into one of **14** MMLU domain categories, then routed to a corresponding model. That worked for basic cases. Production enterprise traffic showed the limit.

Worked example on the page: “I need urgent help reviewing a security vulnerability in my authentication code.” A classifier calls it “computer science” and sends it to a general coding model. It misses:

- **urgency** that wants immediate attention
- **security** sensitivity that wants specialized expertise and jailbreak protection
- **code review** intent that benefits from reasoning
- **authentication** complexity that needs careful analysis

The constraint: classification-based routing captures **one** dimension of intent — the domain — and ignores the rest of the signals in the query.

This post introduces **Signal-Decision Architecture**: from 14 fixed categories to unbounded routing decisions. Multi-dimensional signal extraction, AND/OR decision logic with priority, built-in plugin orchestration.

Local figures (copyright remains with the original site; study copies):

![signal 0](../../../../assets/vllm/blog/serving/semantic-router-signal/01-signal-0.png)

## The problem: why classification-based routing doesn’t scale

Previous pipeline:

```text
User Prompt → MMLU Domain Classification → Model Selection
```

### Single-dimensional analysis

Only domain / subject matter. It cannot capture:

- **Urgency:** “urgent”, “immediate”, “critical”
- **Security sensitivity:** “vulnerability”, “exploit”, “breach”
- **Intent types:** code review, architecture design, troubleshooting
- **Complexity:** simple FAQ vs complex reasoning
- **Compliance:** PII handling, regulatory constraints

**Real impact on the page:** a medical query about “urgent patient data breach” reaches a medical model but lacks PII protection and security filtering — potentially violating HIPAA.

### Fixed category constraint

Limited to 14 predefined MMLU categories (math, physics, computer science, business, …). Impossible to:

- create custom categories for specific business domains
- define fine-grained routing rules inside a domain
- scale beyond academic subject classification

**Real impact:** an enterprise with **50+** specialized use cases (legal contracts, financial compliance, medical diagnostics, code security audits) cannot express routing inside 14 categories.

### Inflexible logic

Cannot combine conditions or implement complex strategies:

- no AND/OR: “expert model only when urgent **AND** security-related”
- no priority when multiple conditions match
- no conditional plugins based on signal combinations

**Real impact:** cannot do layered strategies such as “high-priority security issues get reasoning + jailbreak protection, while general questions get cached responses.”

![signal](../../../../assets/vllm/blog/serving/semantic-router-signal/02-signal.png)

## Introducing Signal-Decision Architecture

Separate signal extraction from routing decisions. Flexible decision engine with built-in plugin orchestration.

### Architecture overview

![signal 1](../../../../assets/vllm/blog/serving/semantic-router-signal/03-signal-1.png)

Three innovations named:

1. **Multi-Signal Extraction** — several dimensions of intent at once
2. **Decision Engine** — AND/OR plus priority-based selection
3. **Plugin Chain** — built-in intelligence for caching, security, optimization

### Complete request flow

![signal 2](../../../../assets/vllm/blog/serving/semantic-router-signal/04-signal-2.png)

## Core concepts

### Signals: multi-dimensional prompt analysis

This post extracts **three** complementary signal types. Each uses a different technique.

![signal 3](../../../../assets/vllm/blog/serving/semantic-router-signal/05-signal-3.png)

#### Keyword signals: interpretable pattern matching

Regex-based terms or phrases. **Human-interpretable** — you can see which keywords fired.

Technical approach:

- compiled regex for efficient matching
- AND/OR boolean operators
- case-sensitive and case-insensitive modes
- **no model inference** (zero ML overhead)

Advantage: transparency for compliance auditing and production debugging.

Use cases named: urgency markers (“urgent”, “immediate”, “asap”, “critical”); security keywords (“vulnerability”, “exploit”, “breach”, “CVE”); compliance terms (“HIPAA”, “GDPR”, “PII”, “confidential”); intent patterns (“code review”, “architecture design”, “troubleshooting”).

#### Embedding signals: scalable semantic understanding

Neural embeddings, semantic similarity between the query and candidate phrases. Intent beyond exact keywords.

Technical approach:

- pre-computed embeddings for candidate phrases (offline)
- runtime query embedding with lightweight models (e.g. sentence-transformers)
- cosine similarity with configurable thresholds
- aggregation: **max** (any match), **mean** (average similarity), **any** (threshold-based)

Advantage: scales to thousands of candidate phrases. Adding a pattern does not require retraining — add phrases and compute embeddings.

Use cases: intent paraphrase (“I need help” → “technical support request”); “How do I fix this bug?” ≈ “debugging assistance”; cross-lingual routing with multilingual embeddings; typos, abbreviations, informal language.

#### Domain signals: dataset-driven classification

MMLU-trained classifiers for academic / professional domain. Custom expansion via **LoRA**. Details: [semantic-router-modular.md](semantic-router-modular.md).

Technical approach:

- fine-tuned classification on MMLU (**14** base categories)
- custom domain expansion via LoRA adapters
- multi-label classification
- confidence scoring

Advantage: enterprises can add **private** domain categories without retraining the whole model. Examples on the page:

- Healthcare: `medical_imaging`, `clinical_trials`, `pharmaceutical_research`
- Finance: `risk_modeling`, `algorithmic_trading`, `regulatory_compliance`
- Legal: `contract_law`, `intellectual_property`, `litigation_support`

![signal 4](../../../../assets/vllm/blog/serving/semantic-router-signal/06-signal-4.png)

Use cases: domain-expert models (math → math-expert); domain-appropriate policies (medical → PII protection); specialized knowledge bases (legal → legal retrieval); domain-specific plugins (code → syntax validation).

### Signal comparison

| Signal type | Technique | Interpretability | Scalability | Extensibility |
| --- | --- | --- | --- | --- |
| Keyword | Regex matching | High (transparent rules) | Medium (manual patterns) | Manual addition |
| Embedding | Neural embeddings | Low (black-box similarity) | High (thousands of phrases) | Add phrases dynamically |
| Domain | MMLU + LoRA | Medium (domain labels) | Medium (14+ categories) | LoRA adapters for custom domains |

### Why three signal types?

Complementary, not redundant:

- **Keyword** — fast, interpretable matching for known patterns
- **Embedding** — semantic variation, large phrase sets
- **Domain** — academic datasets, domain-specific expertise

All three at once is the point.

### Decisions: flexible routing logic

Each decision has:

**Signal combination.** AND (high precision) / OR (high recall).

**Priority.** Integer conflict resolution. Higher wins. Enables layered strategies.

**Model reference.** Which model, optional LoRA adapter. Reasoning mode and effort level.

**Plugin chain.** Ordered list: semantic caching, jailbreak detection, PII protection, system prompt injection, header mutation.

#### Decision evaluation flow

![signal 5](../../../../assets/vllm/blog/serving/semantic-router-signal/07-signal-5.png)

Multiple matches → highest priority. None match → default model.

### Plugins: built-in intelligence

This post’s table has **five** built-in plugins, configured per decision:

| Plugin | Purpose | Key features |
| --- | --- | --- |
| **semantic-cache** | Cache similar queries | Configurable similarity threshold, cost optimization |
| **jailbreak** | Detect prompt injection | Threshold-based detection, request blocking |
| **pii** | Protect sensitive information | Redact / hash / mask modes, GDPR / HIPAA compliance |
| **system_prompt** | Inject custom instructions | Replace or insert mode, role customization |
| **header_mutation** | Modify HTTP headers | Add / update / delete headers, metadata propagation |

Plugins run in configured order. Each can modify the request, block execution, or add metadata for downstream.

Iris later adds `hallucination` and others; this page’s inventory is the five above.

#### Plugin chain execution flow

![signal 6](../../../../assets/vllm/blog/serving/semantic-router-signal/08-signal-6.png)

## Scaling from 14 to unlimited

**Traditional (limited):**

```text
14 MMLU Categories → 14 Routing Rules → 14 Model Selections
```

Cannot create custom categories, combine conditions, apply different policies per rule, or scale beyond domain classification.

**Signal-Decision (unlimited):**

```text
3 Signal Types × N Conditions × AND/OR Logic → Unlimited Decisions
```

Unlimited custom rules, flexible combination, unique plugin chains per decision, enterprise complexity.

### Scalability example: enterprise IT support

Traditional: 14 domain routes (`computer_science` → code-model, `engineering` → engineering-model, plus 12 more fixed).

Signal-Decision: hundreds of specialized routes, examples named:

- Urgent + Security + Computer Science → security-expert + reasoning + jailbreak
- Code Review + High Complexity → architecture-model + reasoning
- FAQ + General → cached-model + semantic-cache
- Medical + PII Detected → medical-expert + PII-protection + disclaimer
- Legal + Confidential → law-expert + PII-hash + audit-headers

Each decision can have unique model selection, reasoning configuration, and plugin chains.

## Kubernetes-native design

Two CRDs: **IntelligentPool** and **IntelligentRoute**.

### Complete example: enterprise IT support

#### IntelligentPool: define the model pool

![signal code 0](../../../../assets/vllm/blog/serving/semantic-router-signal/09-signal-code-0.png)

**Caption (YAML is in the figure, not dumped as HTML):** pool with base model `qwen3`, **4** specialized LoRA adapters, a fallback `qwen3` for non-specialized queries, reasoning-family configuration per model.

#### IntelligentRoute: define routing logic

![signal code 1](../../../../assets/vllm/blog/serving/semantic-router-signal/10-signal-code-1.png)

**Caption:** route spec in the screenshot. Surrounding prose lists:

**Multi-signal extraction**

- **3** keyword signals: urgency, security, code-review
- **2** embedding signals: technical-support, architecture-design
- **1** domain signal: computer-science

**Layered decision logic**

- Priority **100**: Urgent + Security + CS → security-expert + high reasoning + jailbreak + PII protection
- Priority **80**: Code Review + CS → code-reviewer + medium reasoning + cache + custom prompt
- Priority **60**: Architecture Design + CS → architecture-expert + high reasoning + cache
- Priority **40**: General Support → base model + aggressive cache

**Plugin orchestration**

- Security-critical queries: jailbreak + PII
- Code reviews: semantic cache + custom system prompts
- Architecture queries: longer cache TTL (**2h vs 1h**)
- General queries: aggressive caching (**0.90** threshold, **4h** TTL)

**Fallback**

- no match → `defaultModel` (`general-assistant`)
- multiple matches → highest priority

### Dynamic configuration flow

![signal 7](../../../../assets/vllm/blog/serving/semantic-router-signal/11-signal-7.png)

Claimed Kubernetes-native properties: zero-downtime config updates, GitOps, multi-cluster, namespace isolation and RBAC.

## Real-world applications

### Enterprise IT support

Challenge: urgency, technical domain, security sensitivity.

Solution: priority layers — 100 Urgent+Security+CS → security-expert + reasoning + jailbreak; 80 Technical Support+Debugging → code-expert + semantic-cache; 60 General → general-model + aggressive-cache.

Results claimed: appropriate model, cost via cache, security on sensitive issues.

### Healthcare platform

Challenge: HIPAA — PII protection and medical disclaimers.

Solution: Health Domain → medical-expert + PII-redaction + disclaimer-prompt + audit-headers.

Results claimed: automatic PII, consistent disclaimers, audit trail.

### Financial services

Challenge: layered security, PII, jailbreak, cost.

Solution: Economics Domain → finance-expert + jailbreak + PII-hash + disclaimer + cache + compliance-headers.

Results claimed: enterprise-grade security, regulatory compliance, cost efficiency.

### Educational platform

Challenge: subject + learning intent.

Solution: Math + Learning Intent → math-expert + reasoning + patient-tutor-prompt + cache; Science + Tutorial → science-expert + engaging-educator-prompt.

Results claimed: personalized teaching, reasoning on complex topics, cost optimization.

### Code assistant

Challenge: different complexity wants different models.

Solution: Architecture Design → reasoning-model + high-effort + complexity-header; Code Review → code-expert + medium-reasoning + cache; Simple Questions → code-expert + cache-only.

Results claimed: optimal selection, cost-effective reasoning, fast simple queries.

## Future roadmap

Foundation for later work. Two buckets:

### Routing core performance

- **Radix tree for keyword matching** — replace regex; target consistent performance at **10,000+** keyword rules.
- **HNSW for embedding search** — approximate nearest neighbor; “millions of candidate phrases” named as the scale target.
- **Parallel LoRA for decode-only models** — multiple LoRA adapters in Decode, one base serving several domains; reduce model-switching overhead for multi-tenant.

### Feature enhancements

- **Visual configuration console** — web UI, real-time validation and testing, without YAML editing.
- **Custom plugin framework** — SDK, community marketplace.
- **Advanced analytics** — real-time decision / signal / cost monitoring, ML-driven recommendations.
- **Model evaluation via multi-turn dialogue** — parallel conversations with candidate models, LLM-as-a-Judge on coherence, relevance, safety, domain expertise. Dynamic routing from actual performance rather than static rules.
- **Intent-aware internal/external model selection** — sensitive / proprietary → internal models; general queries may use external APIs (OpenAI, Anthropic, …). Cost, latency, and compliance balanced from query characteristics.

![signal 8](../../../../assets/vllm/blog/serving/semantic-router-signal/12-signal-8.png)

## Conclusion

Shift from fixed classification to flexible signal-based decisions:

- **Unlimited scalability** — 14 categories → unlimited custom rules
- **Multi-dimensional intelligence** — keyword, embedding, and domain at once
- **Flexible logic** — AND/OR and priority
- **Built-in security** — jailbreak, PII, compliance plugins
- **Cloud-native** — Kubernetes CRDs, dynamic config, zero-downtime updates

Framed for enterprise AI gateways, multi-tenant SaaS, industry-specific assistants.

## Getting started

The page’s close is an invite, not a CLI: try Signal-Decision routing, join the community, share feedback. Concrete install paths land in [semantic-router-iris.md](semantic-router-iris.md).
