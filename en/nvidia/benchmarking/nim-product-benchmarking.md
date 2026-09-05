---
source: https://docs.nvidia.com/nim/large-language-models/latest/reference/benchmarking.html
lang: en
fetched: 2026-09-01
---

# Benchmarking — NVIDIA NIM product page

Short entry in the NIM for LLMs product docs (product version 2.0.11 when fetched). It does not teach commands. The real walkthrough is `nim-01` … `nim-05` in this folder; the ruler itself is `../tools/aiperf.md`.

Once generative apps roll out, cost is how many queries per second you can dismiss while users still wait and still read. **Do not talk cost before accuracy is acceptable.** This page does not cover accuracy.

## Two rulers; do not mix them

Many clients can hit an LLM: long-standing Locust / K6, and NVIDIA **AIPerf** (formerly GenAI-Perf), which knows tokens. They all emit “latency” and “throughput,” but definitions, measurement points, and divisions often disagree. Two numbers on one spreadsheet may be two different kinds of waiting.

| | Load testing | Performance benchmarking |
|---|---|---|
| Typical tools | K6, Locust | **AIPerf** |
| Asks | The system: capacity, autoscaling, network, resources | The model under a given load: throughput, latency, token-level metrics |
| Fails as | Lobby, autoscaler, connection pools | Config, quantization, batch, KV |

Load testing alone never tells you if the knife is dull. Performance benchmarking alone never tells you the lobby is too small when the first real peak arrives. Official guidance: **do both**. This page only points at the Benchmarking Guide — locally, start at `nim-01-overview.md`.
