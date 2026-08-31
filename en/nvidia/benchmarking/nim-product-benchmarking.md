---
source: https://docs.nvidia.com/nim/large-language-models/latest/reference/benchmarking.html
lang: en
fetched: 2026-08-31
---

# Benchmarking — NVIDIA NIM for LLMs (product page)

This is the short product-doc entry. Full walkthrough: NIM LLMs Benchmarking Guide (`nim-01` … `nim-04` in this folder).

- Cost depends on queries/sec while staying responsive. Measure cost only after accuracy is acceptable.
- Tools: Locust / K6 (general load testing) vs NVIDIA AIPerf (LLM-specific token metrics). Definitions differ across tools.
- **Load testing** (K6): capacity, autoscaling, network, resources.
- **Performance benchmarking** (AIPerf): throughput, latency, token-level metrics.

Do both. This page points to the Benchmarking Guide for the real procedure.
