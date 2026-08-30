---
source: https://docs.vllm.ai/en/stable/usage/metrics/
lang: en
fetched: 2026-08-30
---

# Production Metrics — vLLM

Source: https://docs.vllm.ai/en/stable/usage/metrics/

Prometheus metrics on the OpenAI-compatible server: `GET /metrics`.

```bash
vllm serve unsloth/Llama-3.2-1B-Instruct
curl http://0.0.0.0:8000/metrics
```

Most relevant for inference SLOs:

| Metric | Type | Meaning |
|---|---|---|
| `vllm:time_to_first_token_seconds` | Histogram | TTFT |
| `vllm:inter_token_latency_seconds` | Histogram | ITL |
| `vllm:e2e_request_latency_seconds` | Histogram | end-to-end latency |
| `vllm:request_prefill_time_seconds` | Histogram | time in PREFILL |
| `vllm:request_decode_time_seconds` | Histogram | time in DECODE |
| `vllm:request_queue_time_seconds` | Histogram | time WAITING |
| `vllm:kv_cache_usage_perc` | Gauge | KV usage (1 = 100%) |
| `vllm:num_requests_running` | Gauge | in-flight executing |
| `vllm:num_requests_waiting` | Gauge | queued |
| `vllm:prefix_cache_hits` / `queries` | Counter | prefix cache tokens |
| `vllm:num_preemptions` | Counter | preemption count |
| `vllm:generation_tokens` / `prompt_tokens` | Counter | decode / prefill tokens processed |

Also: speculative-decoding counters, NIXL KV transfer histograms, MFU (`--enable-mfu-metrics`). See the official page for the full table and deprecation policy.
