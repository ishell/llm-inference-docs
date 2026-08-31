---
source: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/perf-analyzer-README.html
lang: en
fetched: 2026-08-31
---

# Triton Performance Analyzer

CLI to measure Triton model performance while you try optimizations. For **LLMs**, use AIPerf / GenAI-Perf instead; this tool is the classic (non-generative) path.

**Load modes:** concurrency | request-rate | custom interval  
**Measurement:** time windows until stable, or count windows (N requests).

Supports sequence / ensemble / decoupled models. Inputs auto-generated or supplied.

Quick start: Triton server container → simple model repo → `tritonserver` → SDK container → `perf_analyzer -m simple`.

Remote: `perf_analyzer -m <model> -u <host>:8000`.
