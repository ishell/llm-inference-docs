---
source: https://vllm.ai/blog/2025-01-21-stack-release
lang: en
fetched: 2026-09-04
---

# High Performance and Easy Deployment of vLLM in K8S with vLLM production-stack

Chinese: [zh/vllm/blog/serving/production-stack.md](../../../../zh/vllm/blog/serving/production-stack.md)

2025-01-21. Repo: [vllm-project/production-stack](https://github.com/vllm-project/production-stack). LMCache team + vLLM. Demo numbers, not your SLA. Official page has architecture / TTFT / ITL / Grafana figures; this tree has no local copies of those artworks.

## TL;DR

vLLM already had the largest open-source community around a **single-node** engine. This post asks what it takes to turn that engine into a **cluster serving system**.

They release **vLLM production-stack**, a full inference stack on top of vLLM, with two claimed advantages:

- **~10× better performance** on their write-up: **3–10× lower response delay** and **2–5× higher throughput**, from prefix-aware routing and KV-cache sharing.
- **Easy cluster deployment**: fault tolerance, autoscaling, observability, via Helm.

Open source from day one.

## Context

The line they quote: in the AI arms race it is no longer just who has the best model — it is who has the best **LLM serving system**. vLLM's hardware/model coverage and contributor density were already there; deployments were still mostly **single-node**. LMCache + vLLM built production-stack so an organization can run a GPU cluster with reliability, throughput, and latency they can point a dashboard at.

## Four functions on top of the engine

Reference implementation for a **cluster of GPU nodes**. Complements native vLLM; does not replace PagedAttention.

1. **KV cache sharing & storage** — reuse context KV (powered by [LMCache](https://github.com/LMCache/LMCache)).
2. **Prefix-aware routing** — send the query to the vLLM instance that already holds that context KV.
3. **Observability** — per-engine status and query-level **TTFT**, **TBT** (time between tokens), throughput.
4. **Autoscaling** — react to workload dynamics.

The original page has a comparison table vs neighbouring stacks (KServe and others). Local notes do not copy that table.

### Design path

Application → prefix-aware router checks whether the requested context is already in some instance's memory pool → forward to that node. Autoscaling / cluster manager watches load and starts new vLLM nodes. Observability gathers TTFT, TBT, throughput.

Later [router.md](router.md) extracts the “remember KV” idea into a Rust gateway; [mooncake.md](mooncake.md) adds a pool for when the local instance does not hold the prefix.

## Advantage 1: Helm on Kubernetes

Original command uses `sudo`; keep or drop it to match your cluster:

```bash
sudo helm repo add llmstack-repo https://lmcache.github.io/helm/ &&\
  sudo helm install llmstack llmstack-repo/vllm-stack
```

README + [tutorials](https://github.com/vllm-project/production-stack/tree/main/tutorials) cover standing up a cluster and customizing values. Reference implementation, not “Helm install = production.”

## Advantage 2: multi-round Q&A bench

Workload shape that should light up prefix cache. Comparators: **vLLM + KServe** and a commercial endpoint. They report TTFT and ITL (inter-token latency) wins on the official plots. Multi-round Q&A looking good does **not** imply the same for short-ask / no-shared-prefix traffic.

## Advantage 3: monitoring

Live cluster metrics they name: latency distributions, requests over time, **KV cache hit rate**.

## Close

They frame this as the step from best-in-class single-node engine to a full-scale serving system. Call to action on the page: clone the repo, try it, [interest form](https://forms.gle/mQfQDUXbKfp2St1z7). Contacts: [vLLM Slack](https://slack.vllm.ai/), [LMCache Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ).

One month later ByteDance ships [AIBrix](aibrix.md). Read both as 2025-Q1 **control-plane** layer, not as a bake-off you must pick in these notes.
