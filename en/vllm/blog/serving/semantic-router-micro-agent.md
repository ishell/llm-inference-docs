---
source: https://vllm.ai/blog/2026-06-29-micro-agent-frontier-models
lang: en
fetched: 2026-09-01
---

# Micro-agent: bounded collaboration behind one model name

Chinese: [zh/vllm/blog/serving/semantic-router-micro-agent.md](../../../../zh/vllm/blog/serving/semantic-router-micro-agent.md)  
Scorecard is their closed/hybrid recipes — not “always run every closed model”.

Clients still call `vllm-sr/auto`. The looper runs in the router: Confidence (cheap first, escalate on low score), Ratings (capped `max_concurrent` ensemble), ReMoM (breadth + quorum + synthesis), Fusion (panel–judge), Workflows (budgeted roles). Task-shaped: GPQA keeps `ANSWER: X`, LiveCodeBench watches hidden tests, HLE watches disagreement. Their table: VSR Closed LiveCodeBench 92.6, GPQA-Diamond 96.0, HLE 50.0 — comparators on the original page. Collaboration is a serving primitive, not another app-side agent graph.

Local figures (copyright remains with the original site; study copies):

![router capability layer](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/01-router-capability-layer.png)

![looper micro agents](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/02-looper-micro-agents.png)

![confidence loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/03-confidence-loop.png)

![ratings loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/04-ratings-loop.png)

![remom loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/05-remom-loop.png)

![fusion loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/06-fusion-loop.png)

![workflows loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/07-workflows-loop.png)

![auto recipe loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/08-auto-recipe-loop.png)

![benchmark shaped recipes](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/09-benchmark-shaped-recipes.png)

![three eval scorecard](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/10-three-eval-scorecard.png)
