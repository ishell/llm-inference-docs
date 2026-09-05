---
source: https://vllm.ai/blog/2026-06-29-micro-agent-frontier-models
lang: en
fetched: 2026-09-04
---

# Micro-Agent: bounded collaboration behind one model name

Chinese: [zh/vllm/blog/serving/semantic-router-micro-agent.md](../../../../zh/vllm/blog/serving/semantic-router-micro-agent.md)

2026-06-29. **vLLM Semantic Router Team**. Repo: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router). Launch: [semantic-router.md](semantic-router.md). Spine: [Iris](semantic-router-iris.md) / [signal-decision](semantic-router-signal.md). Panel–judge: [fusion](semantic-router-fusion.md). MoM as a system: [mom](semantic-router-mom.md). Session continuity: [session](semantic-router-session.md). Operable contract: [themis](semantic-router-themis.md). Do not confuse with the in-engine [router.md](router.md). Scorecard rows are **their** closed/hybrid recipes — not “always run every closed model.”

Siblings: [athena](semantic-router-athena.md), [amd](semantic-router-amd.md), [mom-amd](semantic-router-mom-amd.md), [modular](semantic-router-modular.md), [vision](semantic-router-vision.md), [halugate](halugate.md).

Everyone watches the next frontier checkpoint. The more interesting layer may sit in front of it. Routers already cut cost (when a frontier model is deserved), make safety executable (stricter models / filters / review), and coordinate cloud vs edge. The next job on the page:

> A router can make the model better.

Not by changing weights. Not by asking every app to build a bespoke agent graph. By turning **one model API call** into a **bounded collaboration** inside the serving layer.

Local figures (copyright remains with the original site; study copies):

![router capability layer](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/01-router-capability-layer.png)

**Figure 1.** From model selection to capability construction.

[Sakana Fugu](https://sakana.ai/fugu/) made a commercial product of: a “model” can be a surface, a team behind it. Useful language: [Fugu technical report](https://arxiv.org/abs/2606.21228), [Conductor](https://arxiv.org/abs/2512.04388), [Trinity](https://arxiv.org/abs/2512.04695). vLLM-SR’s bet is different **where** the abstraction lives: collaboration should be an **open serving primitive**, not only one hosted endpoint or one app-side graph.

The user still calls one model:

```json
{
  "model": "vllm-sr/auto",
  "messages": [{"role": "user", "content": "..."}]
}
```

Behind that identity the router can select a recipe, fan out to workers, collect a quorum, verify disagreement, synthesize, repair the output contract, and return one ordinary OpenAI-compatible response. Complexity stays inside. Collaboration should **feel like a model**.

## The looper is the runtime

A request enters as an ordinary chat completion. Signals → projections (task-shape or risk bands) → decision → algorithm. That algorithm may be a single-model route **or** a looper route.

Today’s main patterns:

- **Confidence** — sequential escalation: cheaper first; escalate only when the score is too low
- **Ratings** — bounded fan-out under `max_concurrent`; rating-aware aggregation
- **ReMoM** — breadth samples, wait for a success quorum, synthesis round
- **Fusion** — panel → judge → finalizer ([fusion](semantic-router-fusion.md))
- **Workflows** — static roles or a dynamic planner; bounded worker steps; synthesize

![looper micro agents](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/02-looper-micro-agents.png)

**Figure 2.** Looper algorithms inside the router; the model API surface stays.

A looper is not the slogan “ask more models.” It is a small runtime with **budget, topology, trace, and failure policy**.

### Confidence: spend escalation only on hard cases

Start smaller/cheaper; stop if confident enough. Confidence can come from token-level log probability, logprob margin, a hybrid score, self-verification, or an AutoMix-style entailment verifier. Thresholds, failure behavior, and stopping conditions are explicit router policy.

![confidence loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/03-confidence-loop.png)

**Figure 3.** Escalation as a measured stopping policy.

### Ratings: parallel quality under a hard cap

Several candidates in parallel, only up to `max_concurrent`. Collect successes, rating-aware aggregation, failures per route policy. Fit for A/B-style eval, ensembles, routes that already have per-candidate quality signals.

![ratings loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/04-ratings-loop.png)

**Figure 4.** Multi-candidate execution stays bounded and rating-aware.

### ReMoM: breadth with a contract

High reasoning variance, but the answer format must survive. Fan out attempts, wait for a minimum-success quorum, synthesis model merges into the required output contract. If synthesis fails but earlier workers produced valid evidence, fall back to the best valid evidence instead of collapsing into an API error.

![remom loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/05-remom-loop.png)

**Figure 5.** Breadth, quorum, synthesis, fallback as serving-time controls.

### Fusion: disagreement as signal

Sometimes the useful object is not the average answer; it is the **structure of disagreement**. Independent panel answers become evidence. The judge sees agreement, contradiction, unique insight; the finalizer returns one answer with the trace collapsed behind the API. Hard multiple-choice, long-form expert judgment, exact-answer tasks where one confident response is brittle.

![fusion loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/06-fusion-loop.png)

**Figure 6.** Disagreement is evidence, not something to hide.

### Workflows: roles under a budget

Most agentic, therefore the strictest boundaries. Planner can only choose **allowed** worker models. Plan is validated. Steps bounded by max steps, max parallelism, timeouts, error policy. Final response still has to satisfy the output contract. SWE-style: planner, patcher, verifier, finalizer — without the application owning a bespoke agent stack.

![workflows loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/07-workflows-loop.png)

**Figure 7.** A bounded role system, not an unbounded autonomous agent.

### Auto recipes: one name, many loops

Public surface remains `vllm-sr/auto`. Internally, signals and projections choose the loop. Difficulty, risk, contract pressure, latency, and cost are routing facts, not comments in a prompt.

![auto recipe loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/08-auto-recipe-loop.png)

**Figure 8.** Signals pick the collaboration pattern; one model identity remains.

Difference: “agent as app logic” vs “micro-agent as serving runtime.” The router owns budget, policy, topology, trace, and failure mode.

## Recipes beat one universal loop

Eval lesson on the page: **the best loop is task-shaped.** GPQA-Diamond wants strict multiple-choice preservation. LiveCodeBench wants runnable code and hidden-test robustness. Humanity’s Last Exam wants disagreement resolution and exact-answer formatting. SWE-style wants planner / patcher / verifier / finalizer.

`vllm-sr/auto` should not mean “always the biggest loop.” It should mean: select the recipe that fits this task.

![benchmark shaped recipes](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/09-benchmark-shaped-recipes.png)

**Figure 9.** Benchmark-shaped collaboration via signals and projections.

Their recipes:

- GPQA-Diamond: hard science multiple-choice → ReMoM with strict `ANSWER: X` preservation
- LiveCodeBench: constraints, starter code, stdin, float tolerance, timeout risk, hidden-test risk → a code-shaped loop
- HLE: formal reasoning, disagreement risk, long context, exact-answer pressure → deeper ReMoM, smaller Fusion, or fallback

The prompt is only one part. The recipe also defines model pool, roles, reasoning effort, concurrency, quorum, timeout, synthesis model, fallback, output contract, observability labels.

## The scorecard is a proof, not the whole story

Closed-model recipe across three hard benches. **VSR Closed** = only closed-model backends. **VSR Hybrid** = mix open and closed; stronger closed models where judging / repair / synthesis / fallback is higher-risk.

![three eval scorecard](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/10-three-eval-scorecard.png)

**Figure 10.** VSR Closed and VSR Hybrid across LiveCodeBench, GPQA-Diamond, HLE.

| Benchmark | VSR scorecard row | Score | Reference rows (on the page) |
| --- | --- | ---: | --- |
| LiveCodeBench, Jan–Apr 2025 | VSR Closed | 92.6 | Fugu Ultra 92.0, Fugu 90.3, GPT-5.5 90.7, Opus 4.8 90.3 |
| GPQA-Diamond | VSR Closed | 96.0 | Fugu Ultra 95.5, Fugu 95.5, Gemini 3.1 Pro 94.3, GPT-5.5 93.6 |
| Humanity’s Last Exam | VSR Closed | 50.0 | Fugu Ultra 50.0, Fugu 48.5, Gemini 3.1 Pro 45.0 |
| Humanity’s Last Exam | VSR Hybrid | 47.1 | GLM-5.2 40.5, Qwen3.7 Max 41.4, GPT-5.5 41.4 |

Not a claim that every request should use every closed model. Claim: router-owned collaboration can create a **stronger model identity** than the individual calls beneath it, while preserving one API. Users see one name; operators control the recipe; the system can improve without changing the client; open and closed models participate under the same serving abstraction.

## What this means for model serving

Old stack: accept a model name, send to a backend. Next stack asks: what evidence about this request? which quality / cost / latency / safety band? is one model enough? if not, which collaboration? which answer contract must be preserved? what if a provider is slow or wrong? how to expose one clean response while keeping the full trace?

That is infrastructure, not application glue. Micro-agents belong in the router because the router already owns aliases, provider policy, credentials, cost metadata, signals, decisions, retries, timeouts, traces, OpenAI-compatible response semantics.

## Takeaway

“Frontier model” is starting to mean two things: a checkpoint, and a **system boundary**. The orchestration wave made the direction visible. vLLM-SR bets that the capability should be programmable, observable, and open at the serving layer. Next race: better models **and** better routers — when to save money, enforce safety, stay on the edge, go to the cloud, or turn one request into a small, disciplined team.

## Acknowledgements

Research: MBZUAI, McGill, Mila, Agentic Intelligence Lab; especially Prof. Xue Liu and Dr. Bowei He. Individual: Huamin Chen, Yincheng Ren. AMD GPU eval: Andy Luo, Haichen Zhang.
