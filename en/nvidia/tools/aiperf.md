---
source: https://github.com/ai-dynamo/aiperf
lang: en
fetched: 2026-09-01
---

# NVIDIA AIPerf

Successor to GenAI-Perf. Client-side generative-AI benchmark: it sends load at an already-running inference server and reports TTFT, ITL, TPS, RPS, and goodput. The ruler lives on the client; the server must already be up.

NIM walkthrough: `../benchmarking/nim-04-aiperf.md`. Load flags: `aiperf-load-generator.md`. Formulas: `aiperf-metrics.md`. Five worked scenarios: `aiperf-comprehensive.md`. Full CLI is not copied here — see https://docs.nvidia.com/aiperf/reference/command-line-options

Repo: https://github.com/ai-dynamo/aiperf  
Docs: https://docs.nvidia.com/aiperf/

Works against any OpenAI-compatible server: NIM, vLLM, `trtllm-serve`, Triton, Ollama, SGLang, Dynamo. Change the engine; do not change the metric definitions.

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install aiperf
```

On Linux **aarch64**, the `crick` dependency is sdist-only — install a C toolchain first (`build-essential` on Debian/Ubuntu). x86_64 / macOS / Windows use wheels.

Optional extras: `aiperf[mlflow]`, `aiperf[otel]`, `aiperf[wandb]`, or `aiperf[mlflow,otel,wandb]`. They stream results; they do not change the stopwatch.

The NIM guide prefers the Triton SDK container plus `pip install aiperf`. For a local toy server, Ollama:

```bash
docker run -d --name ollama -p 11434:11434 \
  -v ollama-data:/root/.ollama ollama/ollama:latest
docker exec -it ollama ollama pull granite4:350m
```

## Minimal command

```bash
aiperf profile \
  --model "granite4:350m" \
  --streaming \
  --endpoint-type chat \
  --tokenizer ibm-granite/granite-4.0-micro \
  --url http://localhost:11434 \
  --request-count 10
```

Against vLLM / NIM, point `--url` at `localhost:8000`, `--model` at the served name, and `--tokenizer` at the **same** tokenizer the model uses. A wrong tokenizer lies about ISL/OSL and every ratio downstream.

`--streaming` is almost always required. TTFT, ITL, and TTST need SSE with at least one non-empty chunk. Without streaming you mostly measure end-to-end latency.

The CPU-only Ollama table in the official README is **not** an official score. It only shows the table shape: TTFT / TTST / TTFO, request latency, ITL, per-user TPS, ISL/OSL, system TPS, RPS, request count.

Default artifacts under `artifacts/<model>-<endpoint>-concurrencyN/`:

- `profile_export_aiperf.json` / `.csv` — aggregates
- `profile_export.jsonl` — one record per request (use this for P75 and custom percentiles)
- `logs/aiperf.log`

## UI modes

`dashboard` (live TUI), `simple` (progress bars), `none` (headless). Selected-text copy in the TUI is unreliable; press `c` to copy all logs.

## Three-plane architecture

From the Architecture page. Services talk over ZMQ:

| Plane | Components | Role |
|---|---|---|
| Control | SystemController, Timing Manager, Dataset Manager, Worker Manager | What, when, how many |
| Data | Workers ↔ inference server | HTTP request/response |
| Analytic | Record Processors, Records Manager, GPU / Server Metrics | Metrics and telemetry |

Lifecycle: load dataset → optional warmup (discarded) → issue credits to workers → hit the server → capture timing → compute metrics in parallel → export.

A **credit** is permission to send one request. The Timing Manager issues credits according to the timing mode (fixed timestamps, target QPS, or per-user turn gap). Workers wait without a credit. When the server slows down, credits back-pressure naturally instead of stacking a fake client-side delay.

Workers share no state. Multi-turn context lives on that worker. Datasets are memory-mapped so prompts are not copied over IPC.

Only **single-node multiprocess** is supported. Kubernetes appears in forward-looking comments; distributed K8s execution is not registered in this release.

OTel / MLflow telemetry is a sidecar off the analytic plane: a child process and a bounded queue. A full queue drops the oldest event rather than blocking the hot path.

## Supported APIs

OpenAI chat, completions, embeddings, audio, images. NIM embeddings / rankings. Custom frontends or Jinja2 payloads. Plugins cover endpoints, datasets, transports, metrics.

Public datasets include ShareGPT, Mooncake / Bailian / BurstGPT-style traces, plus vision / ASR / spec-decode benches. See the official tutorial index; this note does not recopy the catalog.

## GPU and server metrics

Optional collectors: DCGM Exporter, PyNVML (NVIDIA), amdsmi (AMD). Prometheus server metrics (vLLM `/metrics`, etc.) auto-discover, or pass `--server-metrics`. TRT-LLM’s default `/metrics` is sometimes iteration-stats JSON, not Prometheus — AIPerf probes `/prometheus/metrics` once and then disables that endpoint after a warning.

After a run: `aiperf plot`. `--dashboard` serves an interactive UI on `localhost:8050` by default.

## Known issues

- `--output-tokens-mean` is **not guaranteed** unless you pass `ignore_eos` / `min_tokens` via `--extra-inputs` to a server that honors them. The NIM guide does this.
- Extreme concurrency (docs: typically >15,000) can exhaust client ports. That is the client lobby collapsing, not the model slowing down.
- Invalid config can hang the process. Kill it and check flags.
- Warmup is separate from scored traffic.

Migrating from GenAI-Perf: commands are nearly isomorphic (`profile`, `--streaming`, concurrency / request-rate). New work uses AIPerf. Empty first chunks still do not count as TTFT; ITL still excludes TTFT. Formulas live in `aiperf-metrics.md`.
