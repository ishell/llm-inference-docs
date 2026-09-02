---
source: https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality
lang: en
fetched: 2026-08-31
---

# Keeping vLLM Production Quality

2026-07-16. Snapshot stats (86k stars, 5.6M pip/month, 1918 commits in June 2026, 13M CI minutes) will age. Three gates: CI (loud breaks), nightly perf/accuracy (silent slow/wrong), release (which commit ships).


Local figures (copyright remains with the original site; study copies):

![00 production quality hero airport](../../../../assets/vllm/blog/performance/production-quality/01-00-production-quality-hero-airport.png)

![01 ci pipeline and selected jobs](../../../../assets/vllm/blog/performance/production-quality/02-01-ci-pipeline-and-selected-jobs.png)

![02 ci test groups 266 jobs](../../../../assets/vllm/blog/performance/production-quality/03-02-ci-test-groups-266-jobs.png)

![03 container build stages](../../../../assets/vllm/blog/performance/production-quality/04-03-container-build-stages.png)

![04 pip compiled dependency graph](../../../../assets/vllm/blog/performance/production-quality/05-04-pip-compiled-dependency-graph.png)

![05 accelerator runner fleet](../../../../assets/vllm/blog/performance/production-quality/06-05-accelerator-runner-fleet.png)

![06 standalone buildkite agent flow](../../../../assets/vllm/blog/performance/production-quality/07-06-standalone-buildkite-agent-flow.png)

![07 kubernetes buildkite agent flow](../../../../assets/vllm/blog/performance/production-quality/08-07-kubernetes-buildkite-agent-flow.png)

![08 h200 mig slices](../../../../assets/vllm/blog/performance/production-quality/09-08-h200-mig-slices.png)

![12 ci analyzer bot](../../../../assets/vllm/blog/performance/production-quality/10-12-ci-analyzer-bot.png)

![14 nightly perf eval workload](../../../../assets/vllm/blog/performance/production-quality/11-14-nightly-perf-eval-workload.png)

![14 performance trends](../../../../assets/vllm/blog/performance/production-quality/12-14-performance-trends.svg)

![15 compare view](../../../../assets/vllm/blog/performance/production-quality/13-15-compare-view.png)

![16 accuracy sample debugging](../../../../assets/vllm/blog/performance/production-quality/14-16-accuracy-sample-debugging.svg)

![18 release candidate loop](../../../../assets/vllm/blog/performance/production-quality/15-18-release-candidate-loop.png)

![08 main branch health](../../../../assets/vllm/blog/performance/production-quality/16-08-main-branch-health.png)

## CI

Buildkite pipeline is **diff-dynamic** (docs: few jobs; kernels: 100+). Then 37 groups / 266 jobs. Shared staged Docker; `pip-compile` lockfiles after silent FlashInfer/transformers upgrades bit them. 58 runner queues; outbound Buildkite agents (no inbound/VPN). H200 MIG ×7; scale-to-zero hourly VMs; sccache S3; shared weight store. HUD: ci.vllm.ai. Nightly analyzer bot: ~1.5 revert PRs/day, ~70% right culprit.

Green is not enough: v0.20.0 passed CI; gpt-oss TP>1 broke on Blackwell; DeepSeek V4 throughput collapsed on GB200.

## Nightly

https://github.com/vllm-project/perf-eval — `vllm-bench` + `lm-eval` + BFCL. 17 model×hardware recipes (then). Dashboard compares images.

## Release

Two-week cadence since Nov 2025. Monday: cut from greenest main CI. RC through Wednesday with three gates. No qualifying candidate → no ship. Then 7 wheels + 11 Docker images, smoked before publish.
