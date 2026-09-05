---
source: https://vllm.ai/blog/2026-06-29-micro-agent-frontier-models
lang: en
fetched: 2026-09-04
---

# Micro-agent: bounded collaboration behind one model name

Chinese: [zh/vllm/blog/serving/semantic-router-micro-agent.md](../../../../zh/vllm/blog/serving/semantic-router-micro-agent.md)

2026-06-29. **vLLM Semantic Router Team**. Repo: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router). Launch: [semantic-router.md](semantic-router.md). Spine: [Iris](semantic-router-iris.md) / [signal-decision](semantic-router-signal.md). MoM chapter: [mom](semantic-router-mom.md). Live AMD pool: [mom-amd](semantic-router-mom-amd.md). Fusion primitive: [fusion](semantic-router-fusion.md). Contract release: [themis](semantic-router-themis.md). Session layer: [session](semantic-router-session.md). Do not confuse with the in-engine [router.md](router.md). The scorecard is **their** closed/hybrid recipes — not “always run every closed model.”

Siblings: [athena](semantic-router-athena.md), [amd](semantic-router-amd.md), [vision](semantic-router-vision.md), [modular](semantic-router-modular.md).

Everyone watches the next frontier checkpoint. The page’s bet is the layer **in front of it**. A router already cuts cost (when a frontier model is worth it), makes safety executable (sensitive domains → stricter models / filters / review), and coordinates cloud vs edge. The next job:

> A router can make the model better.

Not by changing weights. Not by asking every app to build a bespoke agent graph. By turning **one model API call** into bounded collaboration inside the serving layer.

Local figures (copyright remains with the original site; study copies):

![router capability layer](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/01-router-capability-layer.png)

**Figure 1.** The router moves from model selection to capability construction.

[Sakana Fugu](https://sakana.ai/fugu/) is cited as the commercial product that made “a model can be a surface, and behind it a team” loud. Research language they name: [Fugu technical report](https://arxiv.org/abs/2606.21228), [Conductor](https://arxiv.org/abs/2512.04388), [Trinity](https://arxiv.org/abs/2512.04695). vLLM-SR puts the abstraction elsewhere: collaboration should be an **open serving primitive**, not one hosted endpoint or one app-side graph.

The client still calls one model:

```json
{
  "model": "vllm-sr/auto",
  "messages": [{"role": "user", "content": "..."}]
}
```

Behind that identity the router can select a recipe, fan out, collect a quorum, verify disagreement, synthesize, repair the output contract, and return one OpenAI-compatible response. The point is not to expose complexity. The point is to make collaboration **feel like a model**.

## The looper is the runtime

A request enters as an ordinary chat completion. Signals → projections (task-shape / risk bands) → decision → algorithm. The algorithm may be a normal single-model route, or a **looper** route.

Main looper patterns on this page:

- **Confidence**: sequential escalation. Cheap candidate first; escalate only when the score is too low.
- **Ratings**: bounded fan-out. Hard `max_concurrent` cap; rating-aware weights.
- **ReMoM**: repeated mixture-of-model reasoning. Breadth samples, minimum-success quorum, synthesis round.
- **Fusion**: panel–judge–final. Independent answers become evidence. Dedicated post: [fusion](semantic-router-fusion.md).
- **Workflows**: micro-agent workflow runtime. Static roles or a dynamic planner; bounded worker steps; synthesis.

![looper micro agents](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/02-looper-micro-agents.png)

**Figure 2.** Looper algorithms run inside the router; the model API surface stays one name.

A looper is not the slogan “ask more models.” It is a small runtime with **budget, topology, trace, and failure policy**.

### Confidence: spend escalation only on hard cases

Start smaller/cheaper. Stop if confident enough. Confidence can come from token-level log probability, logprob margin, a hybrid score, self-verification, or an AutoMix-style entailment verifier.

Pass the threshold → return immediately. Too low → next candidate. Escalation is explicit router policy: thresholds, failure behavior, stopping conditions are visible and tunable.

![confidence loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/03-confidence-loop.png)

**Figure 3.** Confidence turns escalation into a measured stopping policy.

### Ratings: parallel quality under a hard cap

Launch several candidates in parallel, only up to configured `max_concurrent`. Collect successes, apply rating-aware aggregation, fail according to route policy. Fit they name: A/B-style evaluation, ensembles, routes that already have per-candidate quality signals.

![ratings loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/04-ratings-loop.png)

**Figure 4.** Ratings keeps multi-candidate execution bounded and rating-aware.

### ReMoM: breadth with a contract

High reasoning variance, but the answer format must survive collaboration. Fan out reasoning attempts, wait for a minimum-success quorum, then a synthesis model merges evidence into the required output contract.

If synthesis fails but earlier workers produced valid evidence: fall back to the best valid evidence; still a normal response, not an API error.

![remom loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/05-remom-loop.png)

**Figure 5.** ReMoM treats breadth, quorum, synthesis, and fallback as serving-time controls.

### Fusion: disagreement as signal

Sometimes the useful object is not the average answer; it is the **structure of disagreement**. Independent panel answers become evidence. The judge sees agreement, contradiction, unique insight; the finalizer returns one answer with the trace collapsed behind the API.

Useful when there are competing paths: hard multiple-choice, long-form expert judgment, exact-answer tasks where a single confident response is brittle. Policy details: [fusion](semantic-router-fusion.md).

![fusion loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/06-fusion-loop.png)

**Figure 6.** Fusion does not hide disagreement. It turns disagreement into evidence.

### Workflows: roles under a budget

Most agentic pattern; needs the strictest boundaries. The planner can only choose **allowed** worker models. The plan is validated. Steps bounded by max steps, max parallelism, timeouts, error policy. Final response still has to satisfy the output contract.

SWE-style example on the page: planner, patcher, verifier, finalizer — without the application owning a bespoke agent stack. Powerful, still governed by infrastructure.

![workflows loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/07-workflows-loop.png)

**Figure 7.** Workflows gives the router a bounded role system, not an unbounded autonomous agent.

### Auto recipes: one model name, many loops

Public surface remains `vllm-sr/auto`. Internally, signals and projections choose the loop. Difficulty, risk, contract pressure, latency, and cost are routing facts, not comments in a prompt. They can select Confidence, Ratings, ReMoM, Fusion, Workflows, or a fallback.

![auto recipe loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/08-auto-recipe-loop.png)

**Figure 8.** Auto recipes let signals choose the collaboration pattern while preserving one model identity.

Difference they insist on: “agent as app logic” vs “micro-agent as serving runtime.” The router owns budget, policy, topology, trace, and failure mode.

## Recipes beat one universal loop

Eval lesson is not that one algorithm always wins. Opposite:

> The best loop is task-shaped.

- GPQA-Diamond wants strict multiple-choice answer preservation
- LiveCodeBench wants runnable code and hidden-test robustness
- Humanity's Last Exam wants disagreement resolution and exact-answer formatting
- SWE-style tasks need planner / patcher / verifier / finalizer

So `vllm-sr/auto` must not mean “always run the biggest loop.” It means: select the recipe that fits this task.

![benchmark shaped recipes](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/09-benchmark-shaped-recipes.png)

**Figure 9.** Signals and projections choose a benchmark-shaped collaboration pattern.

Their recipes, explicit:

- GPQA-Diamond: hard science multiple-choice → ReMoM with strict `ANSWER: X` preservation
- LiveCodeBench: looks for constraints, starter code, standard input, float tolerance, timeout risk, hidden-test risk before a code-shaped loop
- HLE: formal reasoning, disagreement risk, long context, exact-answer pressure → deeper ReMoM, smaller Fusion, or fallback

A recipe is more than a prompt. It also defines model pool, roles, reasoning effort, concurrency, quorum, timeout, synthesis model, fallback policy, output contract, observability labels.

## The scorecard is a proof, not the whole story

Closed-model recipe across three hard benchmarks. **VSR Closed** = only closed-model backends. **VSR Hybrid** = mix open and closed; stronger closed models where the recipe needs higher-risk judging, repair, synthesis, or fallback.

![three eval scorecard](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/10-three-eval-scorecard.png)

**Figure 10.** VSR Closed and VSR Hybrid scorecard across LiveCodeBench, GPQA-Diamond, and Humanity's Last Exam.

| Benchmark | VSR scorecard row | Score | Reference rows |
| --- | --- | ---: | --- |
| LiveCodeBench, January-April 2025 | VSR Closed | 92.6 | Fugu Ultra 92.0, Fugu 90.3, GPT-5.5 90.7, Opus 4.8 90.3 |
| GPQA-Diamond | VSR Closed | 96.0 | Fugu Ultra 95.5, Fugu 95.5, Gemini 3.1 Pro 94.3, GPT-5.5 93.6 |
| Humanity's Last Exam | VSR Closed | 50.0 | Fugu Ultra 50.0, Fugu 48.5, Gemini 3.1 Pro 45.0 |
| Humanity's Last Exam | VSR Hybrid | 47.1 | GLM-5.2 40.5, Qwen3.7 Max 41.4, GPT-5.5 41.4 |

Read carefully. Not a claim that every request should use every closed model. The claim: router-owned collaboration can create a stronger **model identity** than the individual calls beneath it. Beat or match frontier single-model baselines while keeping one API surface.

Product shape on the page:

- Users see one model name
- Operators control the recipe
- The system can improve without changing the client integration
- Open and closed models participate under the same serving abstraction

## What this means for model serving

Old stack: accept a model name, send to a backend. Next stack asks:

- What evidence do we have about this request?
- What quality, cost, latency, and safety band?
- Is one model enough?
- If not, which collaboration pattern?
- Which answer contract must be preserved?
- What if one provider is slow or wrong?
- How to expose one clean response while keeping the full trace?

That is infrastructure, not application glue. Micro-agents belong in the router because the router already owns aliases, provider policy, credentials, cost metadata, signals, decisions, retries, timeouts, traces, and OpenAI-compatible response semantics.

## Takeaway

“Frontier model” starts meaning two things: a checkpoint, and a **system boundary**. The orchestration wave made the direction visible. vLLM-SR’s bet: programmable, observable, open at the serving layer.

The next race still involves better models. It also involves better routers: when to save money, when to enforce safety, when to stay on the edge, when to go to the cloud, and when to turn one request into a small, disciplined team.

## Acknowledgements

Research collaboration: [MBZUAI](https://mbzuai.ac.ae/), [McGill University](https://www.mcgill.ca/), [Mila](https://mila.quebec/), [Agentic Intelligence Lab](https://agentic-in.ai/), especially [Prof. Xue Liu](https://www.linkedin.com/in/xueliu) and [Dr. Bowei He](https://www.linkedin.com/in/bowei-he-8a9450199/).

Individual contributors: [Huamin Chen](https://www.linkedin.com/in/huaminchen/), [Yincheng Ren](https://www.linkedin.com/in/yincheng-ren/).

AMD GPU evaluation support: [Andy Luo](https://www.linkedin.com/in/andyluo77/), [Haichen Zhang](https://www.linkedin.com/in/haichen-zhang-9010b6382/).
