---
source: https://docs.vllm.ai/en/stable/usage/metrics/
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# 生产指标 — vLLM

英文对照：[en/vllm/metrics/production-metrics.md](../../../en/vllm/metrics/production-metrics.md)  
这些名字是**怎么算出来的**：`design-metrics.md`。客户端秒表（AIPerf / `vllm bench serve`）在门外；`/metrics` 在屋里。两套时钟可以对不齐，要对公式，不要对「都叫 TTFT」三个字母。

OpenAI 兼容 API server 在 `/metrics` 暴露 Prometheus。

```bash
vllm serve unsloth/Llama-3.2-1B-Instruct
curl http://0.0.0.0:8000/metrics
```

前缀一律 `vllm:`，带 `model_name` label。Histogram 的 bucket 还会改——仪表盘写死旧 bucket 时，过几个版本会看起来像指标坏了。

## 和 SLO 最亲的

| 指标 | 类型 | 含义 |
|---|---|---|
| `vllm:time_to_first_token_seconds` | Histogram | TTFT。设计页：从前端 `arrival_time`（开始 tokenize）起算，输入处理算进去 |
| `vllm:inter_token_latency_seconds` | Histogram | ITL |
| `vllm:e2e_request_latency_seconds` | Histogram | 端到端 |
| `vllm:request_prefill_time_seconds` | Histogram | 请求待在 PREFILL |
| `vllm:request_decode_time_seconds` | Histogram | 请求待在 DECODE |
| `vllm:request_queue_time_seconds` | Histogram | WAITING 排队 |
| `vllm:kv_cache_usage_perc` | Gauge | KV 占用，**1 = 100%** |
| `vllm:num_requests_running` | Gauge | 正在执行 |
| `vllm:num_requests_waiting` | Gauge | 在等 |
| `vllm:prefix_cache_hits` / `queries` | Counter | 前缀缓存命中 / 查询的 **token 数**（不是请求数） |
| `vllm:num_preemptions` | Counter | 累计抢占。它往上爬，e2e 和 ITL 通常一起发抖 |
| `vllm:generation_tokens` / `prompt_tokens` | Counter | 已处理的 decode / prefill token |

仓库里有 Grafana 示例仪表盘。它画的那一子集就是官方眼里的「重要」：e2e、TTFT、ITL、KV%、running/waiting、token 直方图、排队 / prefill / decode 时间。先把这一子集接上，再去收藏完整表。

完整表（投机解码计数、LoRA、并行、tokenizer 等）在原页，会随版本增减。过期指标有隐藏 / 删除策略——原页 **Deprecation Policy**。复制一整张表进仓库，过三个月就是谎言。

## 旁边几族

- **投机解码**：接受长度、draft 率一类 counter。`vllm bench serve` 的 ITL/TPOT 分叉，在这里能看到引擎侧的对应物。
- **NIXL KV 传输**：P/D 分离或跨实例 KV 时的直方图。Mooncake / connector 那几篇博客的运维面。
- **MFU**：`--enable-mfu-metrics` 才开。默认关，因为算它要付成本。
- **HTTP 层**：`prometheus_fastapi_instrumentator` 的 `http_requests_total` 等。那是门的次数，不是 token 的缝。

`--api-server-count > 1` 走 Prometheus multiprocess。进程级 `python_gc_*` / `process_*` 会消失。不是坏了，是记账换了房间。

## 日志里那五行

不必刮 Prometheus 也能看见天气。大约每 5 秒：running/waiting、GPU cache %、prompt/gen token/s、最近 1k block 的 prefix-cache hit rate。`vllm:cache_config_info` 把启动配置（block size、前缀缓存开关、`gpu_memory_utilization`…）当成 label 钉在那里，换配置等于换时间线。

可选 `--kv-cache-metrics-sample`：block 寿命、驱逐前空闲、reuse 间隔。要问「KV 是不是在白住」，开这个，不要只盯占用百分比。
