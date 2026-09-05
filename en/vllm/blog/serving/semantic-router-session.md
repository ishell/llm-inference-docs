---
source: https://vllm.ai/blog/2026-06-02-session-aware-agentic-routing
lang: en
fetched: 2026-09-04
---

# SAAR: Session-Aware Agentic Routing

Chinese: [zh/vllm/blog/serving/semantic-router-session.md](../../../../zh/vllm/blog/serving/semantic-router-session.md)

2026-06-02. **Xunzhuo Liu, Bowei He, Huamin Chen, Haichen Zhang (AMD), Andy Luo (AMD), and the vLLM Semantic Router Team**. Repo: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router). Launch: [semantic-router.md](semantic-router.md). Spine: [semantic-router-signal.md](semantic-router-signal.md). v0.1: [iris](semantic-router-iris.md). v0.2: [athena](semantic-router-athena.md). Ships as production-ready in [Themis](semantic-router-themis.md). Vision signals: [vision](semantic-router-vision.md). Later MoM chapter: [mom](semantic-router-mom.md). Loopers: [micro-agent](semantic-router-micro-agent.md). Live AMD pool: [mom-amd](semantic-router-mom-amd.md). Do not confuse with the in-engine [router.md](router.md). Policy-matrix and ROCm numbers are **their** demos.

Siblings: [modular](semantic-router-modular.md), [amd](semantic-router-amd.md), [fusion](semantic-router-fusion.md), [halugate](halugate.md).

Long-horizon agents create a routing problem single-turn prompt routers were not designed for. The router still needs the best model for this request; it also needs to know **when switching would break the session**.

**Session-Aware Agentic Routing (SAAR)** keeps semantic routing, then adds router-owned session memory, hard locks around tool loops and non-portable provider state, safe reset boundaries, prefix-cache-aware switch pricing, and replayable traces.

Across **21,600** deterministic turns: switches **−79.29%**, **3,836** unsafe switches eliminated, estimated physical-model cost **−78.71%**. Across **2,896** live AMD ROCm requests: **0** observed continuity violations.

Local figures (copyright remains with the original site; study copies):

![hero v2](../../../../assets/vllm/blog/serving/semantic-router-session/01-hero-v2.png)

**Figure 1.** Long-horizon agents need routing that understands the session trajectory, not only the latest prompt.

## From prompt routing to session routing

Semantic Router started from: not every request should take the same path. Iris made signals composable. Athena made the router more strategic — model selection, memory, replay, long-context signals, multimodal primitives, AMD ROCm.

Agents change the unit of routing again. A coding or research agent is a **session**: plan, tool call, tool output, edit, test, recover, pause, resume, then short follow-ups like "continue" / "fix it" / "run that again". Those turns only mean something because of the trajectory.

The router is no longer answering only *which model should handle this request?* For agent traffic it also has to answer: **is it safe to switch models inside this session right now?**

## Why single-turn routing breaks down

Locally correct, session-wrong. Typical tool loop:

| Turn | What the client sends | What a prompt router sees | What a session router must remember |
| --- | --- | --- | --- |
| 1 | "Refactor this module and run the tests." | A coding task | Session started on a physical model |
| 2 | The model emits a tool call | A model response | The next tool result belongs to the same model |
| 3 | The client sends the tool result | A terse observation | The model that asked for the tool should receive the result |
| 4 | "fix the failing case" | A short follow-up | Depends on prior code, tests, routing state |
| 5 | Idle, then resume | A new short message | The router can reconsider whether the old model is still worth holding |

Failure modes if the latest message is all you have:

- A tool result goes to a model that did not make the tool call.
- A non-portable continuation id is sent to the wrong physical backend.
- A long warm session loses prefix locality because the current message is short.
- A logical model such as `auto` becomes hard to debug: which physical model actually served the turn?

Agents **should** still switch: cheap → strong when the task hardens, back when a safe boundary appears. The router needs session context to know which moments are safe.

## The SAAR design

Signals still extract, decisions still match, algorithms still rank candidates inside a matched decision. SAAR wraps a **session-control layer** around that result.

![policy flow](../../../../assets/vllm/blog/serving/semantic-router-session/02-policy-flow.png)

**Figure 2.** Router memory, hard locks, reset boundaries, switch economics, replay — then a physical model.

| Piece | What it stores or decides | Why it matters |
| --- | --- | --- |
| Router memory | Last physical model, matched decision, phase, switch count, idle time, cache evidence, replay metadata | Session context without becoming application memory |
| Hard locks | Block switching during active tool loops or non-portable provider-managed state | Correctness before cost or quality |
| Reset boundaries | Reselect after idle timeout or decision drift | Stops session-aware routing from becoming sticky |
| Switch economics | Handoff cost, switch history, remaining-turn priors, prefix-cache checkout | Switching is asymmetric across tiers and session length |
| Replay traces | Why stay / switch / refuse | Makes `auto` inspectable |

This is a **model-selection policy**, not an endpoint load balancer. Semantic Router can choose a model or cluster through the gateway contract. Endpoint membership, health, and LB inside a cluster stay infrastructure — [router.md](router.md) / Envoy.

## Sometimes the router must not switch

![switch boundaries](../../../../assets/vllm/blog/serving/semantic-router-session/03-switch-boundaries.png)

**Figure 3.** Tool loops and provider-managed continuation are hard constraints; idle and decision-drift permit reselection.

Two **hard locks**:

- **Tool-loop continuity.** If a physical model asked for a tool, the result returns to that same physical model. The observation is not a fresh prompt.
- **Provider-managed state.** Non-portable continuation (a response id that belongs to one backend) holds the previous physical model.

If a switch is unsafe, the router should not "buy" its way out with a cheaper model.

Opposite boundary: idle timeout and decision drift reopen selection. Continuity decays after a pause; a matched decision that moves from code editing to synthesis should not stick forever.

| Situation | SAAR behavior | Reason |
| --- | --- | --- |
| Tool call waiting for a result | Hold previous physical model | Local reasoning loop |
| Non-portable provider state | Hold previous physical model | State may be invalid elsewhere |
| Idle past configured boundary | Allow reselection | Continuity pressure decayed |
| Matched routing decision changes | Allow reselection | Task shape changed |
| Long warm session on an expensive model | Raise switch threshold | Prefix locality is valuable |
| Cheap short retry on a small model | Lower switch threshold | Checkout cost is small |

## Router memory is not user memory

SAAR memory is not conversation memory, retrieval memory, or a user profile. It does not summarize facts for the model. Per session it tracks:

- last physical model behind the logical name
- last matched routing decision
- phase: normal / tool-loop / provider-state / idle-reset / drift-reset
- recent switch count
- latest context length and cache evidence
- a replay id joining the response to the decision trace

Application memory stays in the application. Retrieval stays in the retrieval stack. SAAR memory exists only to make routing across turns coherent.

## Prefix cache makes switching asymmetric

![cache checkout discipline](../../../../assets/vllm/blog/serving/semantic-router-session/04-cache-checkout-discipline.png)

**Figure 4.** The same switch costs different amounts by model tier, session length, and physical prefix reuse.

A short retry on a cheap model and a 40-turn warm frontier session are not the same. The latter has a valuable prefix; switching may make the next physical model pay a large input cost even if the visible user message is short.

SAAR prices a **cached-input checkout delta**: gap between normal prompt input price and cached-input price for the physical model under consideration. Longer / more expensive sessions get stricter about discarding prefix locality.

If the user calls `auto`, the router may map that logical name to different physical models over time. A cache hit reported by one backend is **physical evidence for that backend**, not automatically transferable. SAAR keeps backend-reported cached tokens separate from router-estimated reuse and does **not** rewrite upstream usage fields.

## How a request moves through SAAR

Clients hit the OpenAI-compatible gateway, usually with a logical name such as `auto`, plus a stable session id (`x-session-id`).

1. Read request, session id, tool-call context, provider-state markers, candidate set.
2. Normal Semantic Router signal and decision pipeline.
3. Base model-selection (e.g. hybrid scoring).
4. Load previous session routing state from router memory.
5. Hard locks for tool loops and provider-managed state.
6. Idle timeout and decision-drift boundaries.
7. Adjust switch scores with prefix-cache checkout and switch history.
8. Select physical model; emit diagnostics.
9. Update router memory; write a replay trace.

```yaml
routing:
  decisions:
    - name: agentic_routing
      modelRefs:
        - model: qwen3-8b
        - model: qwen3-32b
      algorithm:
        type: session_aware
        session_aware:
          base_method: hybrid
          idle_timeout_seconds: 300
          tool_loop_hard_lock: true
          context_portability_hard_lock: true
          decision_drift_reset: true
          prefix_cache_weight: 0.20
          switch_history_weight: 0.04
```

Policy knobs, not one-size constants. Short customer-service sessions may idle more permissively; long coding agents may lock tool loops and prefix cache harder.

## Observability is part of the feature

![observability trace](../../../../assets/vllm/blog/serving/semantic-router-session/05-observability-trace.png)

**Figure 5.** Physical choices behind a logical model become traces and response headers.

Diagnostics: selected model, selected decision, replay id, session phase, selected confidence, context-token count. A useful trace answers:

- What would the base selector have chosen?
- Hold because of a tool-loop lock?
- Provider-managed state made switching unsafe?
- Idle or drift boundary crossed?
- How did prefix-cache evidence change adjusted scores?
- Final: stay, switch, or locked stay?

Without replay, `auto` is hard to debug. With replay, operators can audit continuity vs a safe switch.

## How they evaluate it

Three layers, one question: more agent-friendly without hiding correctness problems?

1. **Deterministic policy matrix** — isolate control logic from serving noise; stress tool loops, provider state, idle, drift, model tiers, switch history.
2. **Live OpenAI-compatible serving** on AMD ROCm — headers, session ids, diagnostics, failure handling through the real path.
3. **Deterministic agent-task traces** — simulated tool observations and exact final-answer scoring (no judge model).

The goal is not "fewer switches." Sticky sessions can do that. The goal is: remove unsafe switches, keep useful movement, respect expensive prefix locality, stay observable.

## Result 1: unit of control moves from turn to session

Balanced, tool-heavy, frontier-heavy, idle-heavy, provider-state-heavy, and drift-heavy sessions. Five seeds, 40 sessions per seed, 18 turns per session → **21,600** turns.

![synthetic headline](../../../../assets/vllm/blog/serving/semantic-router-session/06-synthetic-headline.png)

**Figure 6.** Headline policy result across 21,600 deterministic turns.

| Policy | Switches | Unsafe switches | Estimated cost reduction | Quality delta |
| --- | ---: | ---: | ---: | ---: |
| Single-turn | 9,709 | 3,836 | 0.00% | +0.0000 |
| Sticky session | 340 | 0 | 98.65% | −0.1433 |
| Initial SAAR | 1,810 | 200 | 70.92% | −0.0122 |
| Full SAAR | 2,011 | 0 | 78.71% | −0.0453 |

Single-turn churns and creates unsafe movement. Sticky nearly stops movement and gives up too much quality. Full SAAR sits in the middle for the right reason: unsafe movement gone, idle and drift still reopen the decision.

## Result 2: hard locks remove the correctness failures

![safety effect](../../../../assets/vllm/blog/serving/semantic-router-session/07-safety-effect.png)

**Figure 7.** Hard locks remove unsafe switching during tool loops and non-portable provider state.

Tool-loop switch violations: **3,404 → 0**. Provider-state switch violations: **432 → 0**. A tool result is not an ordinary prompt; a non-portable continuation id is not an ordinary text field.

## Result 3: not sticky sessions with a new name

![ablation effect](../../../../assets/vllm/blog/serving/semantic-router-session/08-ablation-effect.png)

**Figure 8.** Continuity with movement; not just sticky.

| Variant | Switch reduction | Unsafe switches | Cost reduction | Interpretation |
| --- | ---: | ---: | ---: | --- |
| No tool lock | 74.96% | 760 | 60.05% | Tool-loop violations return |
| No provider-state lock | 77.98% | 200 | 69.82% | Non-portable-state violations return |
| No drift reset | 83.14% | 0 | 81.31% | Over-sticks after task drift |
| No idle boundary | 83.98% | 0 | 80.14% | Over-sticks after pauses |
| No frontier cost | 73.96% | 0 | 54.75% | Leaves expensive warm sessions too easily |
| Full SAAR | 79.29% | 0 | 78.71% | Locks plus safe reselection |

Locks = correctness. Reset boundaries = liveness. Prefix-cache checkout = economic discipline.

## Result 4: invariants hold in live AMD ROCm serving

OpenAI-compatible traffic through the router and AMD ROCm backends; matched schedules for routed vs direct-backend runs.

![live rocm effect](../../../../assets/vllm/blog/serving/semantic-router-session/09-live-rocm-effect.png)

**Figure 9.** Live ROCm: continuity under long sessions and injected backend failures.

**2,896** live requests, **0** continuity violations.

| Workload | Requests | Success rate | p95 overhead | Continuity violations |
| --- | ---: | ---: | ---: | ---: |
| balanced-32x64 | 2,048 | 100.00% | 6.181 ms | 0 |
| stateful-16x48 | 768 | 100.00% | 26.805 ms | 0 |
| idle-16x5-75s | 80 | 100.00% | 283.463 ms | 0 |

The idle workload includes real wall-clock sleeps; that p95 is **not** hot-path routing overhead.

## Result 5: sessions recover after backend faults

| Fault phase | Requests | Injected 503s | Affected sessions | Recovery | Continuity violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| provider state | 360 | 48 | 8 | 100.00% | 0 |
| tool loop | 360 | 72 | 8 | 100.00% | 0 |
| topic drift | 432 | 48 | 8 | 100.00% | 0 |

One-shot disruption: **32/32** affected sessions recovered. Repeated-failure matrix: **24/24** recovered after **168** injected HTTP 503s. A transient 503 should not make the router forget an active tool loop, non-portable provider state, or replayable history.

## Result 6: task traces exercise the agent loop

Deterministic multi-turn traces with simulated tool observations and exact final-answer scoring (required labels present or not; no judge). AMD serving task run: **18/18** exact-scored instances complete; replay headers on **96/96** routed turns; no continuity violation. Smaller than a broad coding-agent benchmark; stronger than policy counters alone.

## What this changes for vLLM users

Useful when a logical model name hides a model portfolio. Users: call `auto`, send a stable session id. Operators: configure when continuity is required, when idle can reset, how much prefix locality matters, how decisions are traced.

Especially when: candidate models differ in cost / latency / capability; agents use tools across turns; clients depend on provider-managed continuation; long sessions build prefix-cache locality; operators need to inspect which physical model served each turn.

Clean boundary: Semantic Router owns policy-level model selection. Envoy, Kubernetes, and serving backends still own endpoint membership, health, and load balancing.

## The larger direction

Iris made decisions composable. Athena moved toward a strategic system brain. [Vision](semantic-router-vision.md) broadened the evidence surface from text to request-level signals. SAAR broadens the **time** horizon: not only this request, but where it sits in a long-running interaction.

The router does not become an agent. It becomes aware of the minimum session facts required to serve agents well. A router behind `auto` should know when a switch is allowed, when it is forbidden, and what the switch costs for a warm long-running session.

## Join them

Open work they list: session-aware policies on real agent traffic; multi-turn / tool-loop eval suites; AMD ROCm serving validation; observability / replay / production debugging; Envoy / Kubernetes / gateway integrations that keep policy separate from endpoint LB.

- GitHub: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
- Docs: [vllm-semantic-router.com](https://vllm-semantic-router.com)
- Slack: `#semantic-router` on [vLLM Slack](https://vllm-dev.slack.com/archives/C09CTGF8KCN)
