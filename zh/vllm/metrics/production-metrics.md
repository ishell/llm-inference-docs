---
source: https://docs.vllm.ai/en/stable/usage/metrics/
lang: zh
fetched: 2026-08-30
---

# 生产指标 — vLLM（中文摘译）

英文带完整表格：`en/vllm/docs/metrics.md`（下面一并写入英文）  
原文：https://docs.vllm.ai/en/stable/usage/metrics/

OpenAI 兼容 API server 在 `/metrics` 暴露 Prometheus 指标。

```bash
vllm serve unsloth/Llama-3.2-1B-Instruct
curl http://0.0.0.0:8000/metrics
```

和压测最相关的：

| 指标 | 类型 | 含义 |
|---|---|---|
| `vllm:time_to_first_token_seconds` | Histogram | TTFT |
| `vllm:inter_token_latency_seconds` | Histogram | ITL |
| `vllm:e2e_request_latency_seconds` | Histogram | 端到端延迟 |
| `vllm:request_prefill_time_seconds` | Histogram | 请求处于 PREFILL 的时间 |
| `vllm:request_decode_time_seconds` | Histogram | 请求处于 DECODE 的时间 |
| `vllm:request_queue_time_seconds` | Histogram | WAITING 排队时间 |
| `vllm:kv_cache_usage_perc` | Gauge | KV 使用率，1 = 100% |
| `vllm:num_requests_running` | Gauge | 正在执行的请求 |
| `vllm:num_requests_waiting` | Gauge | 等待中的请求 |
| `vllm:prefix_cache_hits` / `queries` | Counter | 前缀缓存命中/查询的 token 数 |
| `vllm:num_preemptions` | Counter | 累计抢占次数 |
| `vllm:generation_tokens` / `prompt_tokens` | Counter | 已处理的生成 / prefill token |

还有 spec decode、NIXL KV 传输、MFU（需 `--enable-mfu-metrics`）等。过期指标的隐藏/删除策略见英文原文 Deprecation Policy。
