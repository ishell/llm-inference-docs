---
source: https://vllm.ai/blog/2026-06-29-micro-agent-frontier-models
lang: en
fetched: 2026-09-01
---

# Micro-agent: bounded collaboration behind one model name

Chinese: `../../zh/vllm/blog/serving/semantic-router-micro-agent.md`  
Scorecard is their closed/hybrid recipes — not “always run every closed model”.

Clients still call `vllm-sr/auto`. The looper runs in the router: Confidence (cheap first, escalate on low score), Ratings (capped `max_concurrent` ensemble), ReMoM (breadth + quorum + synthesis), Fusion (panel–judge), Workflows (budgeted roles). Task-shaped: GPQA keeps `ANSWER: X`, LiveCodeBench watches hidden tests, HLE watches disagreement. Their table: VSR Closed LiveCodeBench 92.6, GPQA-Diamond 96.0, HLE 50.0 — comparators on the original page. Collaboration is a serving primitive, not another app-side agent graph.
