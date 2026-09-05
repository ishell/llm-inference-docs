---
source: https://vllm.ai/blog/2026-06-16-vllm-sr-fusion-api
lang: en
fetched: 2026-09-04
---

# Fusion: panel, judge, synthesis — as a routing decision

Chinese: [zh/vllm/blog/serving/semantic-router-fusion.md](../../../../zh/vllm/blog/serving/semantic-router-fusion.md)

2026-06-16. **vLLM Semantic Router Team**. Repo: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router). Launch: [semantic-router.md](semantic-router.md). Spine: [Iris](semantic-router-iris.md) / [signal-decision](semantic-router-signal.md). MoM: [mom](semantic-router-mom.md). Loopers: [micro-agent](semantic-router-micro-agent.md). AMD routing essay: [semantic-router-amd.md](semantic-router-amd.md). OpenRouter DRACO rows are **their** numbers, not a vLLM-SR eval. The page says this release is the serving primitive; a larger public quality eval is future work.

Fusion is policy, not a global slug. Signals first; only a Fusion decision pays for a panel. [OpenRouter's Fusion launch](https://openrouter.ai/blog/announcements/fusion-beats-frontier/) is cited as a market signal that model panels are a live serving pattern — this post is not about cloning a hosted endpoint.

Local figures (copyright remains with the original site; study copies):

![hero v2](../../../../assets/vllm/blog/serving/semantic-router-fusion/01-hero-v2.png)

**Figure 1.** Fusion API turns model diversity into a vLLM-SR routing primitive: panel, judge, synthesis, trace.

## The vLLM-SR thesis

For years the default serving question was: *which single model should serve this request?* Still useful; no longer enough. Production policies on the page need to:

- route simple requests to fast low-cost models
- escalate difficult requests to stronger specialists
- preserve session continuity when model switching would hurt context
- apply privacy, safety, and tenant policy before model execution
- fan out to several models when disagreement is valuable
- record the decision path so operators can debug and improve it

Core view: model quality is not only a checkpoint property. It is also a property of the serving system around that checkpoint.

[Mixture-of-Models on AMD GPUs](semantic-router-mom-amd.md) introduced that router-centered view: capture signals, select models, coordinate heterogeneous backends, expose the route. ReMoM extended it into multi-round collaboration (see [micro-agent](semantic-router-micro-agent.md)). Fusion adds a more direct panel–judge–synthesis pattern when multiple independent passes are worth the latency.

## What Fusion adds

Fusion is not the whole MoM story. It is one algorithm in the router's toolbox. In vLLM-SR it is **routing policy**, not a fixed global endpoint:

1. **Signals** describe the request: domain, complexity, context, safety, feedback, or other evidence.
2. **Decisions** choose a normal route or a Fusion route.
3. **Fusion-only entry** with `model: "vllm-sr/fusion"` narrows matching to Fusion-capable decisions — no silent fallback to a single-model route.
4. **Panel models** produce independent candidate answers.
5. **A judge model** extracts consensus, contradictions, partial coverage, unique insights, and blind spots.
6. **A synthesis call** returns one user-facing answer.
7. **The trace** records which models participated and what happened.

A hosted model slug hides most of this. vLLM-SR makes panel, judge, policy, and trace explicit so operators choose *where* Fusion belongs instead of paying for it on every request.

## Why the OpenRouter result is a useful signal

OpenRouter's launch is treated as a public proof point for the same systems idea. On [DRACO](https://ar5iv.labs.arxiv.org/html/2602.11685) (hard open-ended deep-research tasks), OpenRouter reported fused panels beating individual models.

**These are OpenRouter's numbers, not a vLLM-SR benchmark.** External evidence that model composition deserves to be a first-class serving primitive:

| Configuration reported by OpenRouter | Score |
| --- | ---: |
| Fusion: Fable 5 + GPT-5.5, synthesized by Opus 4.8 | 69.0% |
| Fusion: Opus 4.8 + GPT-5.5 + Gemini 3.1 Pro, synthesized by Opus 4.8 | 68.3% |
| Fusion: Opus 4.8 + Opus 4.8, synthesized by Opus 4.8 | 65.5% |
| Solo Claude Fable 5 | 65.3% |
| Fusion: Gemini 3 Flash + Kimi K2.6 + DeepSeek V4 Pro, synthesized by Opus 4.8 | 64.7% |
| Solo DeepSeek V4 Pro | 60.3% |
| Solo Kimi K2.6 | 53.7% |
| Solo Gemini 3 Flash | 43.1% |

The interesting row for vLLM-SR is the **budget panel** (64.7%): independent diversity recovering quality a single cheaper model lacks. That is the tradeoff a router should control.

## How Fusion works in vLLM-SR

Principle: Fusion is a routing algorithm, not a global model setting.

Global runtime config only registers which model slugs trigger direct Fusion execution. Panel, judge, error policy, templates, and runtime knobs live on the **matched routing decision** — workload-specific. A research route may want three diverse providers. A code-review route may want two local specialists and one stronger synthesis model. A privacy-sensitive route may keep the whole panel on self-hosted vLLM backends.

![fusion entry modes](../../../../assets/vllm/blog/serving/semantic-router-fusion/02-fusion-entry-modes.png)

**Figure 2.** Fusion is signal-driven. Auto routing can choose any decision; direct Fusion routing chooses among Fusion decisions only; request plugins override execution, not global policy.

Three entry paths into the same algorithm:

| Entry path | How vLLM-SR handles it |
| --- | --- |
| `model: "vllm-sr/auto"` | Full signal and decision policy. Fusion executes only if the selected decision uses `algorithm.type: fusion`; otherwise the matched non-Fusion route runs. Legacy aliases `auto` and `MoM` remain supported. |
| `model: "vllm-sr/fusion"` | Same signal extraction, but decision matching limited to Fusion-capable decisions. If none match, a clear no-match error unless the request provides a panel override. |
| `plugins: [{ "id": "fusion", ... }]` | Overrides judge, panel, and selected runtime knobs for one request. If no Fusion decision matches and `analysis_models` is provided, vLLM-SR builds a request-scoped `fusion_direct` execution. |

Once the Fusion looper runs, execution is explicit:

1. **Resolve policy.** Merge decision-level Fusion config, decision model refs, and request-level plugin overrides.
2. **Protect the router.** Registered Fusion slugs **cannot** be used as judge or panel models — a Fusion request cannot recursively call Fusion.
3. **Run the panel.** Analysis models execute concurrently, bounded by `max_concurrent`.
4. **Handle failures by policy.** `on_error: skip` allows partial panels; `on_error: fail` makes provider failure visible immediately.
5. **Analyze disagreement.** The judge produces structured analysis over consensus, contradictions, partial coverage, unique insights, and blind spots.
6. **Synthesize or call a tool.** Final judge/synthesis call returns one assistant response, or an OpenAI-compatible `tool_calls` response when the client supplied tools.
7. **Return trace and accounting.** Fusion trace, intermediate panel outputs, failed-model records, aggregated token usage across panel, judge, and synthesis.

The caller gets an OpenAI-compatible response; the operator gets which decision fired, which models participated, how many iterations ran, what failed, and total token usage.

This release: policy-controlled panels, explicit stage contracts, provider interoperability, traceable execution. Quality eval (Fusion vs single-model vs frontier panels) is called out as future work.

## Fusion is a decision, not a default

Fusion is useful when independent perspectives help. It is expensive: panel calls + judge + synthesis, usually more latency. The production question is not only "can we fuse" but "**when is Fusion worth it?**"

`model: "vllm-sr/auto"` lets the router decide whether a request uses Fusion at all. Simple prompts stay on a fast single-model route. Hard research, ambiguous analysis, high-stakes synthesis, or tasks where disagreement is valuable can match a Fusion decision. The same signal-decision layer can encode domain, tenant, privacy, cost, session, or safety policy **before** paying the latency.

`model: "vllm-sr/fusion"` is Fusion-only routing. Still signals and decisions; matching narrowed so it does not silently fall back. Request-level Fusion plugins override the panel for one call.

![fusion decision not default](../../../../assets/vllm/blog/serving/semantic-router-fusion/03-fusion-decision-not-default.png)

**Figure 3.** Fusion is a decision, not a default. Policy decides when extra latency is worth it.

Control plane vs a single hosted Fusion slug:

| Production question | vLLM-SR control |
| --- | --- |
| Should this request use Fusion? | `vllm-sr/auto` with signals and decisions |
| Which Fusion policy should apply? | Fusion-capable decisions with priorities and rules |
| Which models should participate? | Per-decision judge and panel config |
| How should latency and failures be handled? | `max_concurrent`, `on_error`, and optional token policy |
| Where can models run? | Local vLLM backends, private endpoints, and public providers |
| How do operators debug the route? | Decision metadata, Fusion trace, failures, and aggregated usage |

## After the decision: traceable Fusion

A small multi-model workflow with explicit stage boundaries. Panel → independent candidates. Judge → structured analysis. Final stage → one assistant answer, or a tool call if the client provided tools.

If a panel model fails: `on_error: skip` continues with partial evidence and records the failed model; `on_error: fail` stops immediately. If structured judge output cannot be parsed: preserve the raw analysis and mark the parse failure instead of hiding it. The final response can include Fusion trace, intermediate panel outputs, failed-model records, and total token usage.

![fusion stage contracts](../../../../assets/vllm/blog/serving/semantic-router-fusion/04-fusion-stage-contracts.png)

**Figure 4.** Explicit stage contracts: panel output, judge analysis, synthesis, and trace accounting stay inspectable.

That is how Fusion becomes one implementation of a programmable Mixture-of-Models control plane — not just a feature.

## Try it with vLLM-SR

### Let the router decide

`vllm-sr/auto` chooses among all configured decisions:

```json
{
  "model": "vllm-sr/auto",
  "messages": [
    {
      "role": "user",
      "content": "What are the strongest arguments for and against carbon taxes?"
    }
  ]
}
```

Matched decision with `algorithm.type: fusion` → Fusion. Otherwise the normal selected-model path.

### Request Fusion explicitly

`vllm-sr/fusion`: still signal extraction; only Fusion-capable decisions are eligible:

```json
{
  "model": "vllm-sr/fusion",
  "messages": [
    {
      "role": "user",
      "content": "What are the strongest arguments for and against carbon taxes?"
    }
  ]
}
```

### Override the panel for one request

Request-scoped; does not move judge or panel defaults into global config:

```json
{
  "model": "vllm-sr/fusion",
  "messages": [{ "role": "user", "content": "..." }],
  "plugins": [{
    "id": "fusion",
    "model": "google/gemini-3-flash-preview",
    "analysis_models": [
      "google/gemini-3-flash-preview",
      "moonshotai/kimi-k2.6",
      "deepseek/deepseek-v4-pro"
    ]
  }]
}
```

### Use Fusion in agent loops

Keep the OpenAI-compatible tool loop. Fusion gives tool-call authority **only to the final judge**. Panel models and the structured judge-analysis call run **text-only**: they see conversation history, including prior tool results, but they do **not** receive `tools` or `tool_choice`.

```json
{
  "model": "vllm-sr/fusion",
  "messages": [
    {
      "role": "user",
      "content": "Find the latest benchmark result and explain whether it changes our launch plan."
    }
  ],
  "tools": [{
    "type": "function",
    "function": {
      "name": "web_search",
      "parameters": {
        "type": "object",
        "properties": {
          "query": { "type": "string" }
        },
        "required": ["query"]
      }
    }
  }],
  "tool_choice": "auto"
}
```

Panel produces independent text analysis; judge compares; only the final judge can answer directly or return standard OpenAI-compatible `tool_calls`. Non-streaming: regular Chat Completions JSON. Streaming: tool-call SSE chunks with `finish_reason: "tool_calls"`. Client-appended tool results are preserved in the next Fusion turn — multi-round agent loops continue to work.

### Configure entrypoints and decisions

Global config registers API entry aliases only:

```yaml
global:
  router:
    auto_model_names:
      - vllm-sr/auto
      - auto
      - MoM
```

Fusion slugs under the looper integration:

```yaml
global:
  integrations:
    looper:
      fusion:
        model_names:
          - vllm-sr/fusion
```

Per-decision config owns route semantics, judge, panel, and runtime knobs:

```yaml
routing:
  decisions:
    - name: deep-research-fusion
      description: Use model diversity for research prompts with high synthesis risk.
      rules:
        operator: AND
        conditions:
          - type: domain
            name: research
          - type: complexity
            name: needs_reasoning:hard
      algorithm:
        type: fusion
        fusion:
          model: google/gemini-3-flash-preview
          analysis_models:
            - google/gemini-3-flash-preview
            - moonshotai/kimi-k2.6
            - deepseek/deepseek-v4-pro
          max_concurrent: 3
          on_error: skip
```

Separation is deliberate. `global` is route-independent runtime state. Judge, panel, optional token budget, concurrency, and route semantics belong to the decision.

Optional OpenRouter-style alias for existing clients:

```yaml
global:
  integrations:
    looper:
      fusion:
        model_names:
          - vllm-sr/fusion
          - openrouter/fusion
```

**By default, vLLM-SR registers only `vllm-sr/fusion`.**

## What comes next

OpenRouter's DRACO result is a signal that model panels deserve serious evaluation. Next steps named:

- run larger public evals beyond smoke coverage
- compare Fusion, ReMoM, AutoMix, Router-R1, and single-model baselines
- study budget panels against frontier-model panels
- expose trace-level diagnostics for disagreement, missing coverage, and judge behavior
- let routing policy decide when the extra latency is justified

Direction: the best answer will not always come from the largest model. Increasingly from the best **model system**, and vLLM-SR is where that system should be programmable.
