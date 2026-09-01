---
source: https://docs.vllm.ai/en/stable/benchmarking/cli/
lang: en
fetched: 2026-09-01
---

# Benchmark CLI — vLLM

Chinese: `../../zh/vllm/benchmarking/cli.md`  
Official: https://docs.vllm.ai/en/stable/benchmarking/cli/

The page’s own framing: this CLI is mainly **feature / regression evaluation**. For production vLLM servers they recommend **GuideLLM** (live progress, auto reports; more flexible datasets, request shapes, and traffic). `vllm bench serve` is still the in-tree client. NVIDIA’s stack uses AIPerf against the same OpenAI-compatible mouth. Names look alike; **formulas may differ — do not rank numbers across tools.**

Grid search of `max-num-seqs` × `max-num-batched-tokens`: `auto-tune.md`.

## Online: serve, then hit

```bash
vllm serve NousResearch/Hermes-3-Llama-3.1-8B

vllm bench serve \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path <path>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10
```

On success the client prints successful requests, duration, input/output tokens, request throughput, output / total token throughput, and Mean / Median / P99 **TTFT**, **TPOT** (excluding the first token), **ITL**. Those latencies are measured at the **benchmark client** — same as AIPerf: the ruler is outside the door.

`--plot-timeline` / `--plot-dataset-stats` write an HTML timeline and ISL/OSL histograms. `--timeline-itl-thresholds` defaults to 25ms, 50ms. `--save-result` keeps JSON.

## TTFT / TPOT / ITL (vLLM’s division)

Names are not standardized. Compare measurement points and formulas, not names.

```
TPOT = (e2e_latency − TTFT) / (output_tokens − 1)
```

- **TTFT**: send → first streamed output.
- **ITL**: gap between consecutive streamed outputs; stats pool those gaps across successful requests.
- **TPOT**: per request (drop the first token), then aggregate across requests.

With ordinary decoding, one streamed output is usually one token, so ITL ≈ TPOT.

With **speculative decoding**, one streamed output may hold several accepted draft tokens. ITL only records gaps *between* outputs; it does not invent zero-width gaps inside a chunk. TPOT amortizes decode time over every output token. Official example: two 40 ms ITL samples, three tokens in the second chunk → mean ITL stays 40 ms; TPOT = `(180 − 100) / (5 − 1) = 20 ms/token`. Same run, two stopwatches can differ by 2×. Do not scold spec-decode TPOT with ITL.

## How load is issued

Three knobs:

| Flag | Default | Meaning |
|---|---|---|
| `--request-rate` | `inf` | Target QPS; `inf` = fire immediately, max throughput |
| `--burstiness` | `1.0` | Gamma shape; only when rate is not `inf` |
| `--max-concurrency` | unlimited | In-flight cap, like a gateway |

`burstiness` is Gamma shape; CV ≈ `1/√burstiness`:

- `0.1`: very bursty (CV ≈ 3.16) — resilience
- `1.0`: Poisson (CV = 1) — people
- `5.0`: smoother (CV ≈ 0.45) — latency portraits

Official seats:

| Use | burstiness | rate | max-concurrency |
|---|---|---|---|
| Max throughput (most common in prod benches) | n/a | `inf` | limited |
| Realistic | 1.0 | moderate 5–20 | unlimited |
| Stress | 0.1–0.5 | high 20–100 | unlimited |
| Latency profiling | 2.0–5.0 | low 1–10 | unlimited |
| Capacity | 1.0 | variable | limited |
| SLA | 1.0 | target QPS | SLA cap |

`--request-rate inf --max-concurrency N`: users fire as fast as they can; the door only holds N. That is “limiter in front, engine eats what it can.” Startup logs report the theoretical ceiling:

```
GPU KV cache size: 15,728,640 tokens
Maximum concurrency for 8,192 tokens per request: 1920
```

`max_concurrency ≈ kv_cache_size / max_model_len`. Capacity planning: put `--max-concurrency` at 80–90% of that. Hugging the theoretical max measures the OOM cliff, not a sustainable lobby.

`--probe-request-rate` sends single-token probes **around** `--max-concurrency` and reports them separately — interference on unrelated guests. Request rate can also ramp over the run.

## Datasets (table kept; per-dataset recipes not recopied)

ShareGPT, ShareGPT4V/Video, BurstGPT, Random / RandomMultiModal / RandomForReranking, Prefix Repetition, HuggingFace sets (VisionArena, MMVU, InstructCoder, AIMO, MTBench, HumanEval, GSM8K, Blazedit, ASR, …), Spec Bench, SPEED-Bench, custom jsonl (text / audio / image).

HuggingFace: `--dataset-name hf`; a local dir still needs `--hf-name` for the Hub id.

Custom text jsonl: one `prompt` per line. Audio: `prompt` + `audio` path; Whisper uses `--backend openai-audio` + `/v1/audio/transcriptions`; Qwen2-Audio uses chat + `--enable-multimodal-chat`. Images: `openai-chat` + `/v1/chat/completions`; `--custom-ensure-client-side-data` base64-encodes local files.

Full wget URLs and per-dataset commands stay on the official page (VisionArena, SPEED-Bench, BFCL, long-document QA, prefix caching, hashing microbenchmarks, …).

## Offline: `vllm bench throughput`

No HTTP. Engine batching, not the lobby.

```bash
vllm bench throughput \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset-name sonnet \
  --dataset-path vllm/benchmarks/sonnet.txt \
  --num-prompts 10
```

Multimodal offline needs `--backend vllm-chat` or image tokens are under-counted. Sonnet is marked deprecated and still used in the example.

The rest of the official page (structured output, embedding, reranker, multimodal processor) is feature regression, not the production-SLA path. Production numbers belong on AIPerf or GuideLLM; explain client TTFT with `/metrics` (`../metrics/production-metrics.md`).
