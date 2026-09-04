---
source: https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality
lang: en
fetched: 2026-09-04
---

# Keeping vLLM Production Quality

Chinese: [zh/vllm/blog/performance/production-quality.md](../../../../zh/vllm/blog/performance/production-quality.md)

2026-07-16. Snapshot stats will age: **86K+** GitHub stars, **5.6M+** monthly pip installs, **2.5M+** monthly image pulls, **1000+** model architectures, **600+** accelerator types. June 2026: **1,918** commits into main (64/day, PyTorch/Kubernetes pace), CI **13 million** job-minutes, peak **1,400** concurrent runners.

The surface that makes vLLM worth using is the surface they have to defend on every commit: clean on H100, may fail to compile on AMD, lose throughput on B200, or nudge outputs on one backend. This post is process, not kernel internals.

Three gates from PR to a version:

1. **CI** — what breaks *loudly*, on every PR.
2. **Nightly perf + accuracy** — what breaks *silently* (slow or wrong), beyond what CI can afford.
3. **Release** — which commit ships, then wheels and images.

Local figures (copyright remains with the original site; study copies):

![00 production quality hero airport](../../../../assets/vllm/blog/performance/production-quality/01-00-production-quality-hero-airport.png)

## Layer 1: CI

### Diff-dynamic unit tests

Every PR starts with lightweight **GitHub Actions** (lint, format). When a committer is ready to merge, heavier unit tests run on **Buildkite**.

![01 ci pipeline and selected jobs](../../../../assets/vllm/blog/performance/production-quality/02-01-ci-pipeline-and-selected-jobs.png)

A bootstrap step reads job definitions, inspects the diff, and schedules only relevant groups. Docs-only: a handful of jobs. Touch important kernels: **100+** in parallel.

Full suite then: **37 test groups, 266 jobs** — kernels, speculative decoding, LoRA, and combinations. Groups range from a couple of jobs to a few dozen.

![02 ci test groups 266 jobs](../../../../assets/vllm/blog/performance/production-quality/03-02-ci-test-groups-266-jobs.png)

### Same environment every run

Two kinds of drift: runner setup, and dependencies moving under you. Shared container image for the first; pinned graph for the second.

**One image, every machine.** Majority of the 266 jobs pull the same image, built once per run. Dockerfile stages:

- `base` — CUDA toolchain
- `build` — compile wheels on top
- `runtime` — install those wheels + runtime deps

Then it forks: serving entrypoint → **release image**; test deps → **`test` image** CI pulls. Shared ancestry keeps what they test close to what they ship. A kernel test on B200 and an entrypoints test on L4 see the same bytes.

![03 container build stages](../../../../assets/vllm/blog/performance/production-quality/04-03-container-build-stages.png)

**Pinned versions.** Unpinned deps made the same test pass Monday and crash Wednesday. FlashInfer shipped; the build quietly picked it up. Same story with **nixl**, **transformers**, and transitive deps — cause buried a layer down.

They `pip-compile` top-level deps into lockfiles that pin **every** package, including transitives. Locks update periodically, full CI each time. After that, dependency-caused breakages stopped being a recurring headache.

![04 pip compiled dependency graph](../../../../assets/vllm/blog/performance/production-quality/05-04-pip-compiled-dependency-graph.png)

### Heterogeneous multi-provider fleet

Each job lands on a Buildkite **runner queue** (hardware profile). Example: `gpu_1` = VMs with L4; `b200` = a K8s cluster with B200s. A free runner claims the next job, runs it, reports back.

**58 runner queues** at the time of writing, hardware from many partners. They cannot own this fleet; coverage exists because partners donate it.

![05 accelerator runner fleet](../../../../assets/vllm/blog/performance/production-quality/06-05-accelerator-runner-fleet.png)

Partners differ: some hand over access, some manage their own iron, some have tight security. The glue is the **Buildkite agent**: it runs *inside* the provider’s environment and connects **outbound HTTPS** to receive work. Buildkite never initiates inbound connections — no open ports, no VPN, no access to the provider network.

The agent runs the command, streams logs, reports exit status. Persistent agents wait for more work; ephemeral agents exit after one job.

Two ways to run it:

- **Standalone machine** (their 8×A100, or an Arm server): install the agent, point it at a queue, loop forever.

![06 standalone buildkite agent flow](../../../../assets/vllm/blog/performance/production-quality/07-06-standalone-buildkite-agent-flow.png)

- **Kubernetes** via [Buildkite Agent Stack for Kubernetes](https://github.com/buildkite/agent-stack-k8s): controller turns each matching job into a K8s Job with a single Pod. Recommended: do not install agents on every node; add the stack to the cluster.

![07 kubernetes buildkite agent flow](../../../../assets/vllm/blog/performance/production-quality/08-07-kubernetes-buildkite-agent-flow.png)

Onboarding on their side: create a queue, hand over a token, provider starts the agent. They never need machine access. That is how they test on a donated fleet they describe as **worth millions of dollars a year**.

### Using the hardware

Demand is high; compute is finite.

**MIG-slice the big GPUs.** Most CI jobs use small models and do not need a whole card. An H200 becomes **seven 18 GB** partitions — 7 jobs per GPU. Figure in the post: **eight H200 → 56 MIG slices**. They claim slicing a big GPU is often cheaper than renting many small ones for the same work.

![08 h200 mig slices](../../../../assets/vllm/blog/performance/production-quality/09-08-h200-mig-slices.png)

**Autoscale from zero, one job per machine.** Hourly rented queues: jobs waiting → start machines; idle → **scale to zero**. Each machine takes one job in a container and shuts down. Bonus: every job is a fresh machine, no leftover state.

**Don’t rebuild what you can reuse.** The expensive loops are: (1) the standard Docker image (CUDA kernels + deps), (2) Hugging Face weight downloads.

- **Docker layers** — registry cache, including deps.
- **Warm-cache AMI** for builders — nightly job bakes latest layers so builders start close to main.
- **sccache** — C++/CUDA objects in **S3**. Every builder can **read**; only **main-branch** builders can **write**.
- **Model weights** — download once per cluster onto shared storage; jobs read locally instead of pulling gigabytes each time.

### Making CI health visible

Hundreds of runs/day × hundreds of jobs. A queue quietly backs up for hours; a test flakes 1/20; a job is 10 minutes slower than last month.

They copied the idea of PyTorch’s [hud.pytorch.org](https://hud.pytorch.org/) and built [ci.vllm.ai](https://ci.vllm.ai/). Every **15 minutes**, Buildkite data lands in **Databricks** and **ClickHouse**. Example question on the dashboard: is `main` healthy? (Caption in the post: for the past 3 days, no — and why did jobs take 10 hours?)

![08 main branch health](../../../../assets/vllm/blog/performance/production-quality/16-08-main-branch-health.png)

### Automating failure response

Every night a **CI-analyzer bot** runs the full suite, diffs against last night, reads new error logs, classifies, walks intervening commits for a culprit, posts Slack with an **auto-revert PR**. About **1.5 revert PRs/day**, right failure + culprit **~70%** of the time — on-call starts from a diagnosis, not a blank page. Shout-out in the post to the community and Red Hat’s on-call rotation.

![12 ci analyzer bot](../../../../assets/vllm/blog/performance/production-quality/10-12-ci-analyzer-bot.png)

### What a green check cannot tell you

Broad unit tests, huge fleet, consistent env, monitored. Merge risk drops. CI still skips a lot of e2e to stay fast and cheap, and it does not closely simulate daily user traffic. A change can pass every test and still make a model slower or wrong. That is layer 2.

## Layer 2: nightly performance and accuracy

May (then): they shipped **v0.20.0** and within days cut **v0.20.1** and **v0.20.2**. Two misses:

- `gpt-oss` on **Blackwell** with **tensor parallelism > 1** broke.
- `DeepSeek V4` throughput **collapsed on GB200**.

No benchmarking pipeline yet; nothing ran those models e2e on that hardware before ship. Perf regressions rarely crash — the server starts, requests succeed, users get fewer tokens/s or wait longer for the first token. Accuracy regressions return a valid JSON with a wrong answer. Layer 2 is the system that would have caught v0.20.0.

### Matrix every night

Pipeline: [https://github.com/vllm-project/perf-eval](https://github.com/vllm-project/perf-eval). Each config: how to start the server, args, model, accelerator, tasks.

Three tasks per workload:

- Performance — TTFT, TPOT, and other metrics via `vllm-bench`
- Math/reasoning accuracy — GSM8K, GPQA, AIME via `lm-eval`
- Function-calling — Berkeley Function-Calling Leaderboard (**BFCL**)

![14 nightly perf eval workload](../../../../assets/vllm/blog/performance/production-quality/11-14-nightly-perf-eval-workload.png)

Every night **and** every release candidate, selected models × hardware. Then: DeepSeek V4 Pro/Flash, gpt-oss, Kimi K2.5, MiniMax M2.5 and M3, Qwen3.5, GLM 5.1, Gemma 4, Nemotron 3 Super on **H200, B200, MI300X, MI355X** — **17 model×hardware recipes**. Growing. Planned: GB200/GB300, P/D disaggregation, more models.

### Is it always fast?

Results land in the same CI HUD. [Performance dashboard](https://ci.vllm.ai/perf) — example in the post: gpt-oss 120B on H200, TP=8, split by concurrency.

![14 performance trends](../../../../assets/vllm/blog/performance/production-quality/12-14-performance-trends.svg)

[Compare view](https://ci.vllm.ai/compare) puts two images head-to-head (RC vs last release).

![15 compare view](../../../../assets/vllm/blog/performance/production-quality/13-15-compare-view.png)

### Is it always correct?

[Evaluation dashboard](https://ci.vllm.ai/eval): aggregate scores + error bars, then drill into question, reference, raw response, extracted answer, correctness. Sample-level beats debugging from one number. Figure: an incorrect GSM8K sample.

![16 accuracy sample debugging](../../../../assets/vllm/blog/performance/production-quality/14-16-accuracy-sample-debugging.svg)

## Layer 3: two-week release

Since **November 2025**, a two-week cadence. Why they keep it:

- Changes reach users fast (release never far behind main).
- Predictable for downstream.
- Bisect **~500** commits, not a few thousand.
- Less deadline pressure — miss this train, next one in two weeks.
- Cherry-picks stay clean (a fix from days ago, not a merge-conflict novel).

Every other **Monday** starts release week.

![18 release candidate loop](../../../../assets/vllm/blog/performance/production-quality/15-18-release-candidate-loop.png)

**Start from the safest commit.** Release manager looks at recent full-CI runs on `main`, picks the **greenest**, cuts `releases/vX.Y.Z` there, announces the window.

**Heavy testing on every RC.** Monday through Wednesday: review cherry-pick requests, pick in batches, tag the next candidate. Every candidate walks the same three gates:

- Full CI suite
- Performance benchmark suite
- Model accuracy evaluation suite

Results are tied to a candidate. When a later RC moves CI health / perf / eval, the delta is tens of commits away.

Wednesday ends the cherry-pick window. After that, **only fixes for issues already on RCs**, then a new tag and the three gates again, until one candidate meets the bar.

**No compromise.** All three gates or it does not ship. No qualifying candidate at week’s end is allowed. “Done when it’s done” (GTA 6 joke in the post); they do not take a decade.

**Ship for every platform.** Qualifying commit → build artifacts → **smoke test the artifacts themselves** before publish. At writing:

**7 Python wheels**

- CUDA 12.9 x86_64 / arm64
- CUDA 13.0 x86_64 / arm64
- CPU x86_64 / arm64
- ROCm

**11 Docker images**

- CUDA 12.9, x86_64 / arm64, Ubuntu 22.04 / 24.04
- CUDA 13.0, x86_64 / arm64, Ubuntu 22.04 / 24.04
- ROCm
- CPU x86_64 / arm64

## What’s next (roadmap in the post)

- **Automatic test selection** — today’s hand-maintained mapping goes stale. Trying LLM-based selection, static/dynamic analysis, labeling source paths.
- **Faster time-to-signal** — CI **1–2 hours** average; want **under 30 minutes**.
- **Leaner unit tests** — many “unit” tests spin up a full vLLM server and fire real requests.
- **Better exit codes** — infra failures still reported as failed tests; breaks triage / alert / retry.
- **Flaky-test detection and quarantine** — infra, upstream packages, or unsafe tests; want automatic catch-and-quarantine.
- **Auto-detect bad machines** and pull them from the fleet before they fail a pile of jobs.
- **Better alerting** — they have basic queue congestion and regression alerts; want disk pressure, jobs failing *faster* than usual, broken dep installs.
- **Code-coverage reporting** — coverage is broad; they cannot yet prove every corner is exercised.

Slack `#sig-ci`. Full-time: Inferact was hiring (Ashby link in the original).

## Acknowledgements (as listed)

Not a solo effort. Orgs named (people listed alphabetically in the post): Amazon, AMD, Arm, EmbeddedLLM, Google, HuggingFace, Inferact, Intel, Meta, NVIDIA, Red Hat, Reflection AI; independents Cyrus Leung (DarkLight1337), Yuqi Wang (noooop), haosundent, Mohammad Angkad.

Compute sponsors: **AWS, Crusoe, LambdaLabs, Nebius, NVIDIA, Roblox, RunPod**. **Buildkite** for running CI free of charge. Mentors from Anyscale/Ray days: Lonnie Liu (later OpenAI), Cuong Nguyen (later NVIDIA).
