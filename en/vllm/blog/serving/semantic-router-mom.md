---
source: https://vllm.ai/blog/2026-07-21-vllm-sr-new-chapter-mom
lang: en
fetched: 2026-09-01
---

# Mixture-of-Models: from picking a model to building a system

Chinese: `../../zh/vllm/blog/serving/semantic-router-mom.md`  
Release-page community stats (5k stars / 150+ contributors / 300k HF downloads) are theirs, then.

MoM ≠ MoE: MoE gates experts per token inside one forward; MoM orchestrates different architectures, even different boxes, per request. Timeline: 14-class fast/slow → Iris signals → Athena control plane → Themis operable contract → Fusion / Micro-Agent pick a **collaboration recipe**. Next: one versioned contract to train, eval, export, deploy, invoke. Algorithms: [micro-agent](semantic-router-micro-agent.md).

Local figures (copyright remains with the original site; study copies):

![hero](../../../../assets/vllm/blog/serving/semantic-router-mom/01-hero.png)

![evolution](../../../../assets/vllm/blog/serving/semantic-router-mom/02-evolution.png)

![research arc](../../../../assets/vllm/blog/serving/semantic-router-mom/03-research-arc.png)

![fragmentation before mom](../../../../assets/vllm/blog/serving/semantic-router-mom/04-fragmentation-before-mom.png)

![fragmentation after mom](../../../../assets/vllm/blog/serving/semantic-router-mom/05-fragmentation-after-mom.png)

![execution topologies](../../../../assets/vllm/blog/serving/semantic-router-mom/06-execution-topologies.png)

![preference models](../../../../assets/vllm/blog/serving/semantic-router-mom/07-preference-models.png)

![four planes](../../../../assets/vllm/blog/serving/semantic-router-mom/08-four-planes.png)

![artifact resolution lifecycle](../../../../assets/vllm/blog/serving/semantic-router-mom/09-artifact-resolution-lifecycle.png)

![matched compute evaluation](../../../../assets/vllm/blog/serving/semantic-router-mom/10-matched-compute-evaluation.png)

![mom lifecycle](../../../../assets/vllm/blog/serving/semantic-router-mom/11-mom-lifecycle.png)

![portable realizations](../../../../assets/vllm/blog/serving/semantic-router-mom/12-portable-realizations.png)

![next stage roadmap](../../../../assets/vllm/blog/serving/semantic-router-mom/13-next-stage-roadmap.png)

![community](../../../../assets/vllm/blog/serving/semantic-router-mom/14-community.png)
