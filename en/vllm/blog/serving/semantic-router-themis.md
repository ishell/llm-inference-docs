---
source: https://vllm.ai/blog/2026-06-05-v0.3-vllm-sr-themis-release
lang: en
fetched: 2026-09-05
---

# vLLM Semantic Router v0.3 Themis: From Signals to Stateful Production Routing

Chinese: [zh/vllm/blog/serving/semantic-router-themis.md](../../../../zh/vllm/blog/serving/semantic-router-themis.md)

2026-06-05. **vLLM Semantic Router Team**. Repo: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router). Launch: [semantic-router.md](semantic-router.md). Spine: [semantic-router-signal.md](semantic-router-signal.md). v0.1: [semantic-router-iris.md](semantic-router-iris.md). v0.2: [semantic-router-athena.md](semantic-router-athena.md). SAAR write-up: [semantic-router-session.md](semantic-router-session.md). Later MoM chapter: [semantic-router-mom.md](semantic-router-mom.md). Do not confuse with the in-engine [router.md](router.md). Release-page counts and leaderboard snapshot are theirs.

Siblings in this folder: [modular](semantic-router-modular.md), [amd](semantic-router-amd.md), [mom-amd](semantic-router-mom-amd.md), [vision](semantic-router-vision.md), [fusion](semantic-router-fusion.md), [micro-agent](semantic-router-micro-agent.md).

v0.3, codename **Themis**, is where the page says semantic routing becomes stateful, observable, and production-ready for real AI traffic. Iris made decisions composable. Athena rebuilt the model foundation and expanded into memory, safety, model selection, long-context signals, OpenClaw, and AMD ROCm. Themis is the next step: easier to operate, easier to inspect, harder to misuse.

Since v0.2.0: more than **350 commits** across router core, CLI, dashboard, DSL, Kubernetes, protocol compatibility, model selection, safety, replay, and release readiness. The largest value is not one feature. It is one stable contract:

> signals become projections, projections feed decisions, decisions choose algorithms, and algorithms select models.

That contract now shows up in the router, CLI, dashboard, DSL, Helm chart, and operator-oriented deployment surfaces.

Local figures (copyright remains with the original site; study copies):

![hero v2](../../../../assets/vllm/blog/serving/semantic-router-themis/01-hero-v2.png)

**Figure 1.** Themis turns signals, policy, operators, and model backends into one inspectable routing control plane.

## Why Themis?

Themis stands for order, rules, and judgment. Semantic routing is only useful in production if operators can answer:

- Which signals fired?
- Which decision matched?
- Which model-selection algorithm ran?
- Which model was selected?
- Which safety or replay plugin changed the path?
- Which config version produced this behavior?
- Can the same policy be deployed locally, through the dashboard, and in Kubernetes without becoming three different systems?

v0.3 keeps Athena’s ambition, with stronger boundaries around the runtime, API surface, and operational workflow.

![release value map](../../../../assets/vllm/blog/serving/semantic-router-themis/02-release-value-map.png)

**Figure 2.** Value is the connection between stable contracts, inspection, operations, serving, long context, and validation — not one isolated feature.

## What’s new in v0.3 Themis?

### 1. A canonical v0.3 configuration contract

New top-level shape:

```yaml
version: v0.3
listeners: []
providers: {}
routing: {}
global: {}
```

Before v0.3, overlapping layouts existed across local Docker, dashboard-generated config, Helm values, CRDs, examples, and older docs. Themis makes `config.yaml` the steady-state file and aligns those surfaces on the same top-level architecture.

This cleanup **removes `vllm-sr init`**. New flow:

- `vllm-sr serve` from an empty directory for dashboard-first setup
- author canonical `config.yaml` directly for YAML-first workflows
- migrate older files with `vllm-sr config migrate --config old-config.yaml`
- import supported provider inventories with `vllm-sr config import`

Breaking, on purpose, for a pre-1.0 router: fewer config dialects, clearer ownership, a more durable public contract.

Edges are stricter: warn on unknown YAML fields; canonical config loading covered by tests; Python CLI models aligned with modern Pydantic configuration; classifier assets gated more explicitly. Goal: typos and stale shapes should be caught before silent routing drift.

![config contract](../../../../assets/vllm/blog/serving/semantic-router-themis/03-config-contract.png)

**Figure 3.** Local YAML, CLI, dashboard, and Kubernetes converge on the same canonical v0.3 config shape.

### 2. Signal, projection, decision, algorithm, model

| Layer | What it owns |
| --- | --- |
| Signal | Extract evidence from the request, response, tools, language, domain, context, modality, identity, or safety classifiers |
| Projection | Normalize raw evidence into policy-ready concepts such as verification, urgency, feedback, or balance |
| Decision | Match named routing policies with priority and explainable conditions |
| Algorithm | Choose among candidate models inside a matched decision |
| Model | Serve the request through the selected backend alias or provider |

Richer signal families, projection traces, advanced selectors, and response-side plugins make implicit behavior unacceptable. The catalog describes not only the latest prompt, but safety posture, tool loops, user roles, multimodal intent, conversation shape, structured events, and replayable knowledge-base evidence:

| Signal family | What it captures | Typical use |
| --- | --- | --- |
| `authz` | Role and subject bindings from user or group context | Premium/admin routing, policy-gated models |
| `complexity` | Reasoning difficulty from learned or composed signals | Escalate hard synthesis and multi-step reasoning |
| `context` | Estimated context-window demand | Long-context routing, cost and latency decisions |
| `conversation` | Message and tool-loop shape | Multi-turn, active tool use, developer messages, heavy non-user context |
| `domain` | Learned or configured domain labels | Business, law, health, computer-science routing |
| `embedding` | Semantic similarity against candidate anchors, including text/image/audio query modality | Support intent, clinical intent, multimodal request matching |
| `event` | Structured event metadata, severity, action codes, and temporal urgency | Incident, payment, audit, or operational event routing |
| `fact_check` | Whether a request needs factual verification | Escalate legal, medical, or factual claims |
| `jailbreak` | Prompt-injection and jailbreak evidence, including history-aware scanning | Safety routing and response-side guardrails |
| `kb` | Knowledge-base group or label matches | Privacy policy, containment, frontier reasoning, local standard routes |
| `keyword` | Literal, fuzzy, BM25, or n-gram keyword evidence | Fast route guards, urgent keywords, sensitive terms |
| `language` | Detected language with configurable confidence | Locale-aware routing and multilingual model choice |
| `modality` | AR, diffusion, or mixed text/image execution needs | Choose text-only, image-generation, or multimodal paths |
| `pii` | Sensitive entity policy, including history-aware scanning | Redaction, deny/allow decisions, privacy routes |
| `preference` | User style or behavior preference examples | Terse answers, detailed answers, domain-specific style |
| `reask` | Repeated or rephrased user turns | Detect likely dissatisfaction in prior turns |
| `structure` | Regex, count, sequence, or density features | Many questions, numbered workflows, format-heavy prompts |
| `user_feedback` | User says an answer was wrong or needs clarification | Recover from dissatisfaction or route to stronger models |

Projection outputs are referenced with `type: projection`. They are **derived** routing surfaces, not another raw signal family: signals extract evidence; projections turn evidence into named policy bands such as `support_fast`, `support_balanced`, or `support_escalated`.

v0.3 composability called out: `conversation` can detect agentic request shape; `event` can route operational payloads; embedding rules can query non-text modalities; projections can turn noisy evidence into policy-ready bands.

Dashboard topology, DSL editor, compiler/decompiler, and runtime metrics were updated to understand these surfaces instead of silently dropping them.

DSL gained conflict detection, `SIGNAL_GROUP`, `TEST`, and `TIER` authoring constructs, a natural-language-to-DSL pipeline, `EMIT retention`, and dynamic tool retrieval. Themis policies are reviewable routing programs with tests, retained outputs, and safer generation paths — not only parsed YAML.

![routing contract](../../../../assets/vllm/blog/serving/semantic-router-themis/04-routing-contract.png)

**Figure 4.** Pipeline from request evidence to signal, projection, decision, algorithm, model, and replay.

### 3. Session-Aware Agentic Routing

First production-ready **Session-Aware Agentic Routing (SAAR)**. Dedicated note: [semantic-router-session.md](semantic-router-session.md).

Single-turn asks: which model should handle this prompt? Agentic routing also asks: is it safe to switch models inside this session right now?

SAAR adds router-owned session memory, hard locks around tool loops, provider-state portability checks, idle and decision-drift reset boundaries, switch economics, and replayable diagnostics. It keeps the normal Semantic Router pipeline and wraps the last mile of model selection.

Especially important for coding agents and long-horizon tool loops: a tool result should usually return to the model that asked; a provider-managed continuation id should not go to a different physical backend; a long warm session should not throw away prefix locality because the latest user message is short.

![session aware routing](../../../../assets/vllm/blog/serving/semantic-router-themis/05-session-aware-routing.png)

**Figure 5.** SAAR keeps multi-turn agent sessions stable with router-owned session memory, hard locks, portability checks, switch economics, and replay diagnostics.

Pieces:

- `conversation` signals identify multi-turn shape, active tool use, developer messages, and heavy non-user context.
- `session_aware` selection evaluates whether a switch is worth it after quality gap, switch margin, stay bias, prefix locality, and remaining-turn priors.
- Hard locks stop unsafe switches during active tool loops or provider-state continuations.
- Router-owned memory can retrieve and store route-local facts, preferences, and context without a separate session-state DSL.
- Replay records preserve why a session stayed, switched, or reset.

Memory is the durable complement: facts, preferences, and retrieved context under user or session scope, so the agent can keep continuity without pinning every request to the most expensive model forever.

Reference policy (ordinary YAML):

```yaml
routing:
  signals:
    conversation:
      - name: active_tool_use
        feature:
          type: count
          source:
            type: assistant_tool_cycle
        predicate:
          gte: 1

  decisions:
    - name: agentic_session_route
      rules:
        operator: AND
        conditions:
          - type: conversation
            name: active_tool_use
      algorithm:
        type: session_aware
        session_aware:
          base_method: hybrid
          tool_loop_hard_lock: true
          context_portability_hard_lock: true
          prefix_cache_weight: 0.20
          handoff_penalty_weight: 1.0
      plugins:
        - type: memory
          configuration:
            enabled: true
            retrieval_limit: 6
            auto_store: true
            hybrid_search: true
```

### 4. Projections turn evidence into policy

Without projections, a complex policy repeats low-level signal details across many decisions. With them, the router computes raw evidence once, derives a reusable output such as `support_fast` or `support_escalated`, and lets decisions route on that concept.

Three core patterns:

- `partitions` choose one winner from an exclusive family (e.g. competing support intents).
- `scores` combine declared signals or knowledge-base metrics into a continuous value.
- `mappings` turn those values into policy bands through calibrated thresholds.

v0.3 also adds `multi_emit` projection mappings: one projection step can emit multiple named routing concepts while keeping replay traceability.

![projection layer](../../../../assets/vllm/blog/serving/semantic-router-themis/06-projection-layer.png)

**Figure 6.** Noisy signal evidence → named outputs that decisions can reference.

Compact example:

```yaml
routing:
  signals:
    embeddings:
      - name: technical_support
        threshold: 0.75
        aggregation_method: max
        candidates:
          - installation guide
          - troubleshooting steps
      - name: account_management
        threshold: 0.72
        aggregation_method: any
        candidates:
          - password reset
          - billing information
    context:
      - name: long_context
        min_tokens: 32K
        max_tokens: 256K

  projections:
    partitions:
      - name: support_intents
        semantics: exclusive
        members:
          - technical_support
          - account_management
        default: technical_support
    scores:
      - name: request_difficulty
        method: weighted_sum
        inputs:
          - type: embedding
            name: technical_support
            weight: 0.18
            value_source: confidence
          - type: context
            name: long_context
            weight: 0.18
    mappings:
      - name: request_band
        source: request_difficulty
        method: threshold_bands
        outputs:
          - name: support_fast
            lte: 0.20
          - name: support_escalated
            gte: 0.45

  decisions:
    - name: escalated_support_route
      rules:
        operator: AND
        conditions:
          - type: projection
            name: support_escalated
```

Projection traces are stored with replay records, so the dashboard can explain which derived policy band caused the final route.

### 5. Protocol compatibility becomes a release surface

Beyond basic OpenAI Chat Completions:

- native Anthropic `/v1/messages` ingress through an internal request envelope
- Anthropic streaming with OpenAI SSE translation
- custom Anthropic upstream routing and tool-calling support
- outbound Anthropic response emission for non-streaming paths
- protocol detection from request path headers
- session-id mirroring and header pass-through controls
- response headers that explain when protocol translation is **lossy**
- Responses API tool-trace fidelity and OpenAI SDK-aligned message handling
- OpenAI reasoning-effort mutation fixes
- identity-encoded upstream responses to avoid transparent decompression surprises
- stronger Responses API state and persistence paths

Goal is not to make every provider look identical. Translation should be explicit, observable, and safe enough that a logical model such as `auto` can sit in front of multiple provider protocols without surprising operators.

### 6. The dashboard becomes an operator console

Not only a config editor. First-run setup, topology graph, replay-backed insights, logs, status pages, evaluation flows, auth behavior, and model inventory are tightened. Operators can import a profile, validate it, activate it, send test prompts, inspect signal paths, read router logs, and verify replay records without leaving the dashboard.

![operator console](../../../../assets/vllm/blog/serving/semantic-router-themis/07-operator-console.png)

**Figure 7.** Setup, topology, logs, playground, replay, and model health.

Notable:

- built-in routing modes and missing-model completion
- topology dry-run paths that show matched signals, projections, decisions, and models
- router replay and aggregate insights through the dashboard proxy
- natural-language DSL builder and evaluation-flow fixes
- file attachments in the playground
- auth **fail-closed** when the auth service cannot initialize
- policy version lifecycle with shadow, activate, and revert states
- safer logs and URL redaction for user-supplied fetch/open-web requests
- UTF-8-safe display for multilingual content
- slimmer production route shell and smaller backend runtime dependencies
- dashboard-aware model list and status surfaces

### 7. CLI and deployment are more predictable

`vllm-sr` is the supported operating interface:

```bash
vllm-sr serve
vllm-sr serve --algorithm latency_aware
vllm-sr serve --algorithm session_aware
vllm-sr serve --platform amd
vllm-sr serve --platform nvidia
vllm-sr chat
vllm-sr eval
vllm-sr model list
vllm-sr config migrate --config old-config.yaml
```

Local `vllm-sr serve` remains Docker-based on Linux, macOS, and WSL2. AMD ROCm remains the **release-validated** GPU path. `--platform nvidia` adds local NVIDIA Docker passthrough for users who already have the NVIDIA container runtime. **Native Windows Docker serving is rejected** with an explicit support message rather than failing later.

Inspection / smoke-test: `vllm-sr model list` (configured inventory), `vllm-sr chat` (one-shot completion), `vllm-sr eval` (evaluation endpoints). `VLLM_SR_DNS` lets local containers join custom DNS when enterprise or lab networks require it.

Kubernetes: Helm, release defaults, OpenShift deployment fixes, multiple `IntelligentRoute` reconcile behavior, CRD modality contracts, optional Gateway API `HTTPRoute` ingress, AgentGateway installation guidance. Release ops move away from vague `latest` toward explicit artifact contracts, upgrade/rollback docs, and release checks.

### 8. Safety, replay, memory, and retrieval are more trustworthy

Athena brought many of these in. Themis hardens them.

**Replay and observability**

- router replay PostgreSQL insert correctness so dashboard insights do not silently stay empty
- projection traces stored with replay records
- response-side jailbreak and replay path tightening

**Storage and retrieval**

- Qdrant vector search provider
- Valkey cache, vector store, and memory backend, including TLS and search-module prechecks
- Redis and Responses API storage defaults that better match local and Kubernetes deployments
- hybrid cache rebuild preallocation reduction
- streaming Redis semantic-cache correctness and bounded streaming chunk memory
- O(N) cache-LRU read paths replaced with a constant-time list-backed implementation
- BM25 and n-gram classification caching
- hybrid HNSW entry-point propagation fixes
- shared Milvus lifecycle across replay, cache, memory, and vector store paths

**Runtime and security hardening**

- history-aware PII and jailbreak scanning across prior user turns
- model switch gate fixes for previous-model population
- goroutine panic recovery in extproc background paths
- concurrency race fixes in selection randomness
- path traversal protection for config rollback versions
- dependency security updates across Python, Go, Rust, and frontend

### 9. Long-context routing gets cheaper

Three controls:

1. Context token estimation can learn an **online calibration ratio** from observed response usage when exact tokenization is unavailable. Fallback remains conservative.

2. Native mmBERT embedding bounds memory without silent clipping. The **#2007** native-binding fix processes attention in **query chunks** instead of materializing one dense attention tensor for the whole sequence.

![long context binding](../../../../assets/vllm/blog/serving/semantic-router-themis/08-long-context-binding.png)

**Figure 8.** Chunked mmBERT attention: keep the long-context signal, bound native memory.

3. Prompt compression is a named profile surface for **signal extraction only**:

| Profile | Intended use |
| --- | --- |
| `default` | Balanced compression for general routing |
| `coding` | Preserve code-like and implementation-heavy sentences |
| `medical` | Preserve clinically relevant detail |
| `security` | Preserve safety and policy evidence |
| `multi_turn` | Preserve conversational continuity |

The original user prompt still goes to the selected serving model unless a decision-owned plugin explicitly changes it. Routing optimization must not silently rewrite user intent.

### 10. Hardware backend paths broaden

Four paths: NVIDIA CUDA and AMD ROCm for served vLLM backends; Intel OpenVINO for router-owned classifier and embedding inference; CPU/local for development and smoke tests.

v0.3 adds an initial **OpenVINO binding**: native C++ and Go integration for ModernBERT sequence classification, token classification, and embedding inference, with benchmark entrypoints comparing OpenVINO and Candle. **Backend and binding milestone, not a blanket production-parity claim.**

![hardware backend paths](../../../../assets/vllm/blog/serving/semantic-router-themis/09-hardware-backend-paths.png)

**Figure 9.** One routing control plane across NVIDIA CUDA, AMD ROCm, Intel OpenVINO, and CPU/local.

AMD path from Athena remains in the v0.3 contract:

```bash
vllm-sr serve --platform amd
```

Maintained profile: `deploy/recipes/balance.yaml` — multiple served aliases through a ROCm vLLM backend, same signal → projection → decision → model-selection pipeline as CPU/local. AMD note: [semantic-router-mom-amd.md](semantic-router-mom-amd.md).

Release-readiness validation on an AMD ROCm stack:

- ROCm vLLM backend exposing the expected served aliases
- dashboard setup import, validate, and activate using the reference balance profile
- router health and Envoy OpenAI-compatible `/v1/models`
- topology dry-run for a coding/debug request
- direct Envoy chat completions for coding, math, and legal prompts
- dashboard proxy chat completions
- router replay list and aggregate insight APIs

![amd validation path](../../../../assets/vllm/blog/serving/semantic-router-themis/10-amd-validation-path.png)

**Figure 10.** Serve, dashboard import, router health, model listing, ROCm backend serving, and routed requests as one flow.

### 11. RouterArena SOTA refresh

In the RouterArena snapshot captured for this release update, **vLLM-SR returned to #1**. Public [RouterArena leaderboard](https://routeworks.github.io/?p=/leaderboard) snapshot: ranked first by weighted Arena Score **75.4**, ahead of Sqwish Router, AgentForge Router, Nadir Router, and other published baselines. Same snapshot: **76.0** accuracy, **$0.11** cost per 1K queries, **73.1** robustness.

![routerarena leaderboard vllm sr](../../../../assets/vllm/blog/serving/semantic-router-themis/11-routerarena-leaderboard-vllm-sr.png)

**Figure 11.** Leaderboard snapshot: vLLM-SR #1 by weighted Arena Score.

Not a substitute for release testing. Outside check while Themis improves policy, cost-aware selection, protocol compatibility, and operational traceability.

## What changed since v0.2?

| Area | Themis value |
| --- | --- |
| API and config | Canonical v0.3 contract across local, dashboard, Helm, and operator paths |
| Router core | Richer signals, projections, response state, replay, safety, and selection algorithms |
| Model selection | Session-aware, multi-factor, latency-aware, RL-driven, hybrid, and other algorithm surfaces |
| Protocols | Stronger OpenAI and Anthropic compatibility with explicit translation behavior |
| Dashboard | Setup, topology, status, logs, insights, replay, auth, and model inventory hardening |
| CLI | Clearer serve modes, model inspection, chat/eval commands, config migration, platform boundaries |
| Deployment | AMD ROCm path, OpenVINO binding, NVIDIA local passthrough ergonomics, Helm/OpenShift/Gateway API fixes, release artifact contracts |
| Storage and retrieval | Valkey, Qdrant, Redis, Milvus, replay, cache, memory, and vector-store lifecycle hardening |
| Reliability | Chunked mmBERT attention, UTF-8-safe display, secure logging, streaming cache correctness, replay correctness, concurrency fixes |

More capable, more constrained in the right places.

## Get started

macOS or Linux:

```bash
curl -fsSL https://vllm-semantic-router.com/install.sh | bash
```

Manual:

```bash
pip install vllm-sr==0.3.0
vllm-sr serve
```

If the current directory has no `config.yaml`, `vllm-sr serve` starts the dashboard in setup mode. YAML-first:

```bash
vllm-sr config migrate --config old-config.yaml
vllm-sr serve --config config.yaml
```

AMD ROCm:

```bash
vllm-sr serve --platform amd
```

Local NVIDIA Docker passthrough:

```bash
vllm-sr serve --platform nvidia
```

Kubernetes:

```bash
helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router
```

Resources:

- Docs: [vllm-semantic-router.com](https://vllm-semantic-router.com)
- GitHub: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
- Reference AMD profile: [deploy/recipes/balance.yaml](https://github.com/vllm-project/semantic-router/blob/main/deploy/recipes/balance.yaml)
- Models: [Hugging Face](https://huggingface.co/LLM-Semantic-Router)

## Looking ahead: v0.4 Hermes

Next codename **Hermes**. Themis makes the contract stable enough to operate. Hermes should make the router faster to improve, easier to evaluate, and safer to adapt under real workloads. Core goal: a **self-improving router**. Loop: auto research for router performance at GPU scale, tune DSL recipes with router evaluation, feed validated evidence back into the codebase and encoder-model fine-tuning. Highest-value work named:

- **Self-improving router**: GPU-scale performance research, DSL recipe tuning, codebase plus encoder fine-tuning. Generated changes still have to be reviewable, replayable, versioned, and rollback-safe.
- **SAAR as the agentic routing layer**: switch economics, tool-loop continuity, provider-state portability, replay diagnostics, router memory.
- **Evaluation as a release gate**: system-level and signal-level evaluation so every signal, projection, algorithm, plugin, and dashboard path can be replayed against representative traffic before release.
- **CLI-first design**: every operation closes the loop through `vllm-sr` — config authoring, migration, serving, inspection, evaluation, replay, policy lifecycle, dashboard import/export, release smoke tests.
- **Better router-owned models**: embedding, classifier, multimodal, and safety signal models.
- **More useful signals**: request, response, tool, modality, identity, freshness, latency, cost, and runtime-health — without turning the DSL into application code.
- **Operator debugging loop**: what-if routing, policy replay, evaluation-driven tuning, and trace comparison as first-class dashboard workflows.

![hermes roadmap](../../../../assets/vllm/blog/serving/semantic-router-themis/12-hermes-roadmap.png)

**Figure 12.** Hermes: GPU-scale performance research, DSL recipe tuning, router evaluation, codebase updates, encoder fine-tuning.

## Acknowledgments

v0.2.0 → v0.3.0: more than **350 commits** from **80+ contributor author identities**. Research collaborators named: MBZUAI, McGill University, Mila, Rice University. Broader thanks: vLLM, AMD, Intel, Meta, Red Hat, Microsoft, Google, IBM, NVIDIA, Hugging Face, NASA, Nutanix, DaoCloud, and open-source communities.

Closing line on the page: *Welcome to Themis: from signals to stateful production routing.*
