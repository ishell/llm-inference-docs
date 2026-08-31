---
source: https://docs.vllm.ai/en/stable/cli/serve/
lang: en
fetched: 2026-08-31
---

# `vllm serve` — flags that matter for inference

Full generated CLI is huge (every EngineArg). This note keeps the knobs that show up in NVIDIA/vLLM tuning docs. Official page: https://docs.vllm.ai/en/stable/cli/serve/

JSON nested flags: `--json-arg '{"k": {"x": 1}}'` ≡ `--json-arg.k.x 1`. Lists: `--json-arg.k+ item`.

YAML: `--config` → see `configuration/serve_args` in the docs.

## Parallelism

| Flag | Role |
|---|---|
| `-tp` / `--tensor-parallel-size` | Tensor parallel groups (default 1) |
| `-dp` / `--data-parallel-size` | Data parallel |
| `-ep` / `--enable-expert-parallel` | MoE expert parallel instead of TP on experts |
| `-dcp` / `--decode-context-parallel-size` | Shard decode KV (does not grow world size by itself) |
| `-pcp` / `--prefill-context-parallel-size` | Split prefill sequence compute (does grow world size) |
| `--api-server-count` / `-asc` | Frontend processes (defaults to DP size) |

## KV / memory

| Flag | Role |
|---|---|
| `--gpu-memory-utilization` | Fraction of GPU for this instance. **Default 0.92**. Per-instance; two engines on one GPU should split (e.g. 0.5+0.5). |
| `--kv-cache-memory-bytes` | Absolute KV bytes; **overrides** utilization when set |
| `--kv-cache-dtype` | `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, `nvfp4`, … |
| `--block-size` | Tokens per KV block |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | APC |
| `--prefix-caching-hash-algo` | `sha256` (default, safer) vs `xxhash` (faster, collision risk) |
| `--max-model-len` | Context; `auto`/`-1` fits whatever GPU allows |

## Scheduler (the TRT-LLM `max_batch` / `max_num_tokens` analogues)

| Flag | Role |
|---|---|
| `--max-num-batched-tokens` | Tokens per engine step. **Primary throughput knob** on V1; try >8192. |
| `--max-num-seqs` | Sequences per step |
| `--enable-chunked-prefill` | V1: on by default when possible |
| `--scheduling-policy` | `fcfs` (default) or `priority` |
| `--async-scheduling` | Hide CPU gaps; better latency *and* throughput |
| `--watermark` | Keep a fraction of KV blocks free to cut preemption (default 0 = off) |

## Compile / “go faster”

| Flag | Role |
|---|---|
| `--optimization-level` | `-O0` fastest start … `-O3` best perf. **Default 2.** |
| `--performance-mode` | `balanced` (default), `interactivity` (small-batch latency), `throughput` (fat graphs / aggressive batching) |
| `-cc` / `--compilation-config` | torch.compile + cudagraph (`-cc.mode=3`) |
| `--speculative-config` / `--spec-method` | ngram, EAGLE, MTP, … |

Tune in the order given in `../optimization/optimization.md` (CPU cores → `-O*` → `max_num_batched_tokens` → parallelism). Do not dump every flag from the 80k generated page into a sweep.
