---
source: https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality
lang: en
fetched: 2026-08-31
---

# Keeping vLLM Production Quality

2026-07-16. Snapshot stats (86k stars, 5.6M pip/month, 1918 commits in June 2026, 13M CI minutes) will age. Three gates: CI (loud breaks), nightly perf/accuracy (silent slow/wrong), release (which commit ships).

## CI

Buildkite pipeline is **diff-dynamic** (docs: few jobs; kernels: 100+). Then 37 groups / 266 jobs. Shared staged Docker; `pip-compile` lockfiles after silent FlashInfer/transformers upgrades bit them. 58 runner queues; outbound Buildkite agents (no inbound/VPN). H200 MIG ×7; scale-to-zero hourly VMs; sccache S3; shared weight store. HUD: ci.vllm.ai. Nightly analyzer bot: ~1.5 revert PRs/day, ~70% right culprit.

Green is not enough: v0.20.0 passed CI; gpt-oss TP>1 broke on Blackwell; DeepSeek V4 throughput collapsed on GB200.

## Nightly

https://github.com/vllm-project/perf-eval — `vllm-bench` + `lm-eval` + BFCL. 17 model×hardware recipes (then). Dashboard compares images.

## Release

Two-week cadence since Nov 2025. Monday: cut from greenest main CI. RC through Wednesday with three gates. No qualifying candidate → no ship. Then 7 wheels + 11 Docker images, smoked before publish.
