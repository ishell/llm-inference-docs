---
source: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html
lang: en
fetched: 2026-09-01
---

# GenAI-Perf

Chinese study note: [zh/nvidia/tools/genai-perf.md](../../../zh/nvidia/tools/genai-perf.md)

Official banner: **being phased out.** New work uses AIPerf (`aiperf.md`). Commands are nearly isomorphic; the concepts still apply. Some numbers on older NIM Performance pages were taken with this ritual. Change the tool; do not change the ruler.

Client-side load against an already-running generative server: output token throughput, TTFT, TTST, ITL, request throughput. The server must already be up. Supports OpenAI chat/completions and Triton’s TensorRT-LLM backend; custom frontends or Jinja2 payloads for private APIs. Custom frontends are more flexible; Jinja2 only changes the envelope.

## Install

```bash
pip install genai-perf   # CUDA 12 must already be on the machine
```

Or the Triton SDK container (docs example `RELEASE=25.01`):

```bash
docker run -it --net=host --gpus=all \
  nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
genai-perf --help
```

It still calls Perf Analyzer underneath. LLMs: this page. Classic non-generative models: `perf-analyzer.md`.

## Minimal example: GPT-2 on Triton

Official quickstart: in the TRT-LLM container, `triton import -m gpt2 --backend tensorrtllm`, then `triton start`. In the SDK container:

```bash
genai-perf profile \
  -m gpt2 \
  --backend tensorrtllm \
  --streaming
```

Sample table (GPT-2 / Triton): TTFT 16.26 ms, ITL 1.85 ms, request latency 499 ms, OSL ~262, ISL ~550, ~521 tok/s out, ~1.99 RPS. The numbers only show which columns exist.

## YAML config

```bash
genai-perf create-template          # default genai_perf_config.yaml
genai-perf create-template -v       # with comments
genai-perf config -f genai_perf_config.yaml
genai-perf config -f genai_perf_config.yaml \
  --override-config --warmup-request-count 100 --concurrency 32
```

Endpoint block in the template: `model_selection_strategy` (`round_robin` / `random`), `backend`, `type` (default `kserve`), `streaming`, `url`, `grpc_method`. For Triton TensorRT-LLM, set `exclude_input_in_output` true in the model config or the engine echoes the prompt and OSL inflates.

`--override-config` changes a couple of knobs without editing YAML.

`genai-perf analyze` sweeps stimulus (concurrency / rate) in one command. Process-export-files merges distributed exports. Details stay on the official Analyze / Process Export Files pages.

## Plots

Off by default. `--generate-plots`: TTFT distribution, request latency, TTFT vs ISL, ITL vs token position, ISL vs OSL.

## Where inputs come from

Synthetic:

- `--num-dataset-entries` — pool size, then recycle
- `--synthetic-input-tokens-mean` / `--stddev`
- `--random-seed`
- `--request-count`, `--warmup-request-count`

File: `--input-file`, JSON objects (prompts or image paths).

Any dataset may also use `--num-prefix-prompts` + `--prefix-prompt-length` (prefix KV); `--output-tokens-mean` / `--stddev`; `--output-tokens-mean-deterministic` (docs claimed Triton only); repeatable `--extra-inputs name:value`.

LLMs have no client batch: one request is one inference. Embeddings / rankings get `--batch-size-text N`.

## Mooncake payload

`--input-file payload:<file>` is a fixed schedule. JSONL: required `timestamp` (ms); optional `input_length`, `output_length`, `text_input`, `session_id`, `hash_ids`, `priority`. `hash_ids` map to 512-token synthetic blocks — same hash, same input block — for KV and spec-decode tests.

Dynamo’s `sin_synth.py` can emit sinusoidal arrivals (duration, rate min/max/period, two ISL/OSL pairs). For production traffic, build your own footprint collector; the README’s logger is hypothetical.

`--session-delay-ratio` scales multi-turn delays without editing the payload.

## Auth

```
-H "Authorization: Bearer ${API_KEY}" -H "Accept: text/event-stream"
```

## Metrics (the ones that still match AIPerf)

| Metric | Meaning | Aggregations |
|---|---|---|
| TTFT | send → first response | avg/min/max/p99/p90/p75 |
| TTST | first chunk → second | same |
| ITL | inter-response gap / tokens in the later response | same |
| Output Token Throughput Per User | (output excluding first token) / generation-phase duration | same |
| Request Latency | send → last chunk | same |
| OSL / ISL | output / input tokens for that request | same |
| Output Token Throughput | total output tokens / run duration | one value per run |
| Request Throughput | completed requests / run duration | one value per run |

Empty first chunks are not TTFT. ITL **excludes** TTFT. System throughput uses wall-clock for the whole run; per-user is 1/ITL. NIM chapter 2 says this in prose.

GPU telemetry (power, util, memory, temp, clocks, ECC, NVLink, PCIe, …) comes from DCGM Exporter `/metrics` on the same machine. `--verbose` prints it. Custom metric CSV: official GPU Telemetry tutorial.

## Flag groups (CLI is not copied flag-by-flag)

- **Endpoint**: `-m/--model` (multiple names mainly for LoRA), `--model-selection-strategy`, `--backend {tensorrtllm,vllm}`, `--endpoint`, `--endpoint-type` (default `kserve`), `--streaming`, `-u/--url`, `--grpc-method`, `--server-metrics-urls`
- **Load**: `--concurrency`, `--request-rate`, `--fixed-schedule`, `--measurement-interval` / `-p`, `--stability-percentage` / `-s` (default 999 ≈ “do not wait for steady state”)
- **Artifacts**: `--artifact-dir` (default `artifacts`), `--profile-export-file` (Perf Analyzer json; GenAI-Perf also writes `*_genai_perf.json/csv`), `--generate-plots`, checkpoint
- **Sessions**: `--num-sessions`, `--session-concurrency`, turns / delay mean/stddev
- **Tokenizer**: `--tokenizer`, defaults to the model name

Stability: last three windows, max/min within the percentage for both infer/s and latency. Default 999% means almost no steady-state wait — lab numbers can look optimistic.

Audio flags (duration, wav/mp3, sample rate, channels) are for audio models.

Full CLI remains on the official page. New work should attach the same concepts to `aiperf.md`.
