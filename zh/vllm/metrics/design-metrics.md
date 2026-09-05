---
source: https://docs.vllm.ai/en/stable/design/metrics/
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# Metrics 设计 — vLLM V1

英文对照：[en/vllm/metrics/design-metrics.md](../../../en/vllm/metrics/design-metrics.md)  
运维对照表：[production-metrics.md](production-metrics.md)。原文：https://docs.vllm.ai/en/stable/design/metrics/  
间隔示意图在文档站 `assets/design/metrics/intervals-*.png`，本库不收。

这一页回答：这些直方图是在哪间屋子里、用哪只钟算出来的。路线图曾经挂在 ["Even Better Observability"](https://github.com/vllm-project/vllm/issues/3616)。

## 目标

- 引擎级 + 请求级都要覆盖，好做生产监控。
- Prometheus 优先——他们预期生产用这个。
- 日志（INFO 打指标）给临时测试、调试、探索。

## 两层

1. **Server-level**（Gauge / Counter）：引擎此刻的天气——在跑、在等、KV 占用、token 累计。用来**解释**为什么某条请求的 SLO 破了。
2. **Request-level**（Histogram）：TTFT、ITL、e2e、prompt/decode 长度。通常这些才是 SLO 本身。

心理模型：server-level 解释 request-level。

### V1 在 `/metrics` 上点名的名字

前缀 `vllm:`。生产页有完整表；设计页举例：

- Gauge：`vllm:num_requests_running`、`vllm:kv_cache_usage_perc`（0–1）
- Counter：`vllm:prefix_cache_queries` / `hits`、`vllm:prompt_tokens_total`、`vllm:generation_tokens_total`、`vllm:request_success_total`（按 finish reason）
- Histogram：`vllm:request_prompt_tokens`、`vllm:request_generation_tokens`、`vllm:time_to_first_token_seconds`（TTFT）、`vllm:inter_token_latency_seconds`（ITL）、`vllm:e2e_request_latency_seconds`、`vllm:request_prefill_time_seconds`、`vllm:request_decode_time_seconds`

### Grafana 示例画了哪些

仓库例子：[examples/observability/prometheus_grafana](https://github.com/vllm-project/vllm/tree/main/examples/observability/prometheus_grafana)。[#2316](https://github.com/vllm-project/vllm/pull/2316) 解释为什么是这一子集：

`e2e_request_latency_seconds_bucket`、`prompt_tokens`、`generation_tokens`、`inter_token_latency_seconds`（他们在这里把 ITL 也叫 TPOT）、`time_to_first_token_seconds`、`num_requests_running`（还写了 `_swapped` / `_waiting`）、`kv_cache_usage_perc`、`request_prompt_tokens`、`request_generation_tokens`、`request_success`、`request_queue_time_seconds`、`request_prefill_time_seconds`、`request_decode_time_seconds`、`request_max_num_generation_tokens`。

V1 里 swapped 那条已经是遗物，见下面废弃。

### Prometheus 库与 HTTP 层

先 [aioprometheus](https://github.com/vllm-project/vllm/pull/1890)，很快换成 [prometheus_client](https://github.com/vllm-project/vllm/pull/2730)。迁移时 HTTP `MetricsMiddleware` 短暂消失，又用 `prometheus_fastapi_instrumentator` 请回来（[#15657](https://github.com/vllm-project/vllm/pull/15657)）：

```bash
curl http://0.0.0.0:8000/metrics | grep -P '^http_(?!.*(_bucket|_created|_sum)).*'
# http_requests_total{handler="/v1/completions",method="POST",status="2xx"} 201.0
# http_request_size_bytes_count / http_response_size_bytes_count / http_request_duration_*
```

门开了几次，不是字与字之间的缝。把它当 TTFT 的替身，会得到一个很有礼貌的谎言。

### 多进程

历史上指标在 engine-core 进程采，再用 multiprocess 送到 API server（[#7279](https://github.com/vllm-project/vllm/pull/7279)）。现在在 API server 采；multiprocess **只** 在 `--api-server-count > 1` 时用（[#17546](https://github.com/vllm-project/vllm/pull/17546)）。

`prometheus_client` 默认还有 `python_gc_*`、`python_info`、`process_virtual_memory_bytes`、`process_resident_memory_bytes`、`process_start_time_seconds`、`process_cpu_seconds_total`、`process_open_fds` / `process_max_fds`。多进程模式不暴露它们；`--api-server-count > 1` 时这些会从 `/metrics` 消失——不是采集坏了。它们也不聚合组成 vLLM 实例的所有进程，相关性本来就可疑。

## 计算放哪

V1 把 **EngineCore** 当成 GPU 内环，尽量瘦。`AsyncLLM` 是外环，最好和 GPU 重叠。记账放在前端：`AsyncLLM.output_handler_loop` 吃 `EngineCoreOutputs`。实现 PR 簇挂在 [#10582](https://github.com/vllm-project/vllm/issues/10582)（[#11962](https://github.com/vllm-project/vllm/pull/11962) 等一串）。遗产 PR：#1890、#2316、#2730、#4464、#7279。

时间间隔必须用**同一进程的 `time.monotonic()`**，不要 `time.time()`（NTP 会拨钟）。跨进程的 monotonic 不能相减——两只钟各说各的「从开机起过了多久」。这就是为什么 TTFT 不在 GPU 进程里减 `arrival_time`。

Scheduler 会把 scheduled / waiting 一类统计塞进 `EngineCoreOutputs`。

## 间隔（engine-core 事件）

这些时间戳记在 **engine-core 进程**（前端看不见 `QUEUED` / `SCHEDULED` 的时刻）：

| Event | 含义 |
|---|---|
| `QUEUED` | 被 engine core 接到，进调度队列 |
| `SCHEDULED` | 第一次被调度执行 |
| `PREEMPTED` | 退回 waiting；将来重调度并 **重开 prefill** |
| `NEW_TOKENS` | 这份 `EngineCoreOutput` 里的 token（整个 `EngineCoreOutputs` 共用一个时间戳） |

推出来的间隔：

- Queue：`QUEUED` → 最近一次 `SCHEDULED`
- Prefill：最近一次 `SCHEDULED` → 随后第一次 `NEW_TOKENS`
- Decode：那次第一次 `NEW_TOKENS` → 最后一次 `NEW_TOKENS`
- Inference：最近一次 `SCHEDULED` → 最后一次 `NEW_TOKENS`
- ITL：相邻两次 `NEW_TOKENS`

TTFT **不是** 上面那条 prefill 间隔：前端从 `arrival_time`（开始 tokenize）起算，输入处理算进第一段等待。E2E 是前端 `arrival_time` → 前端收到最后一个 token。

文档站三张图：普通情况；decode 期抢占（ITL / decode / inference 被拉长）；prefill 期抢占（TTFT / prefill 被拉长）。Decode 期抢占时已生成的 token 会复用。

前端按 engine-core 的每一拍收集：本拍新 token、本拍完成的 prefill 的 prompt token、本拍新调度请求的 queue 间隔、完成 prefill 的 prefill 间隔 / TTFT、本拍所有人的 ITL（他们在这里也把 ITL 写成 TPOT）。完成的请求再记 inference / decode，以及 e2e。

## KV 驻留

`--kv-cache-metrics-sample` 把开销压得很小。抽到的块记：lifetime（分配 → 驱逐）、idle-before-evict（最后一次 touch → 驱逐）、reuse gaps。Prometheus：`vllm:kv_block_lifetime_seconds`、`vllm:kv_block_idle_before_evict_seconds`、`vllm:kv_block_reuse_gap_seconds`。engine core 只在 `SchedulerStats` 里送原始驱逐事件；前端做成 Prometheus 观察，日志开着时也走 `LLM.get_metrics()`。lifetime 和 idle 画在一起，容易看见 stranded cache，或长 decode 把 prompt 钉死。

## 怎么对外

- `LoggingStatLogger`：大约每 5 秒一条 INFO——running/waiting、GPU cache %、过去 5 秒的 prompt/gen tok/s、最近 **1k** 次 block 查询的 prefix-cache hit rate。
- `PrometheusStatLogger`：`/metrics`，给 Prometheus 去刮（页上举例每秒）。Counter 到重启前只增；Gauge 可上可下；Histogram 按桶计数。每条序列带 `model_name`。**bucket 还会改**——「对所有用户都好用的桶」没有一次选对。页上 TTFT 例子：`le="0.02"` 已经有 140 里的 13，`le="0.1"` 才到 140。不要把桶抄进仪表盘当法律。`request_success_total` 按 `finished_reason`：`stop` / `length` / `abort`。
- `vllm:cache_config_info`：Info 指标的思路（Gauge 钉死为 1），启动配置当 label（`block_size`、`cache_dtype`、`cpu_offload_gb`、`enable_prefix_caching`、`gpu_memory_utilization`…）。`prometheus_client` 在 multiprocess 下从未支持 Info，所以用 Gauge + `multiprocess_mode="mostrecent"`。
- LoRA：`vllm:lora_requests_info` 这个 Gauge 的 **值是墙上时钟**，每拍更新。label：`running_lora_adapters` / `waiting_lora_adapters` 是逗号分隔的 `adapter=count` 字符串，外加 `max_lora`。页上自己说把计数塞进 CSV「quite misguided」，本该用 label 区分 adapter。`multiprocess_mode="livemostrecent"`。[#9477](https://github.com/vllm-project/vllm/pull/9477)；至少有一个下游（Gateway API Inference Extension）。删之前要打招呼。
- Prefix cache：每次查询记问了多少 token、命中多少。日志给最近 **1k** 次的命中率。Prometheus 该留 **counter**，让 PromQL 自己选窗口：`rate(cache_query_hit[5m]) / rate(cache_query_total[5m])`——不要做成命中率 Gauge。讨论在 [#10582](https://github.com/vllm-project/vllm/issues/10582)。

## 废弃

删指标不能轻。`vllm:avg_prompt_throughput_toks_per_s` 先 [废弃](https://github.com/vllm-project/vllm/pull/2764)、再 [删除](https://github.com/vllm-project/vllm/pull/12383)，然后有 [用户发现没了](https://github.com/vllm-project/vllm/issues/13218)。页上的规矩：

1. 谨慎，用户冲击难预测。
2. `/metrics` 的 HELP 里写醒目的废弃说明。
3. 用户文档和发版说明列出。
4. 先藏到 CLI 逃生口后面一阵（他们点 Kubernetes 的 [show-hidden-metrics](https://kubernetes.io/docs/concepts/cluster-administration/system-metrics/#show-hidden-metrics)）。生产页口径：`X.Y` 废弃 → `X.Y+1` 隐藏，可用 `--show-hidden-metrics-for-version=X.Y` 再看见 → `X.Y+2` 删除。项目级政策另有 contributing 页。

点名的遗物：

- **从未实现：** `vllm:tokens_total`（[#4464](https://github.com/vllm-project/vllm/pull/4464)），直接删。
- **重复的 queue time：** `vllm:time_in_queue_requests`（[#9659](https://github.com/vllm-project/vllm/pull/9659)，`now - arrival_time`）和后来的 `vllm:request_queue_time_seconds`（Grafana 用后者）。该废前者。
- **Prefix cache hit rate Gauge：** 已改成 queries / hits counter。
- **KV swapped 遗物：** `vllm:num_requests_swapped`、`vllm:cpu_cache_usage_perc`。V1 不再用 GPU↔CPU swap；`--swap-space` 已撤。历史：beam search 的 SequenceGroup 共享 prompt KV、copy-on-write 分叉；后来 prefix caching 更好（V1 接近零开销、默认开），抢占走 recompute。SequenceGroup 在 V1 拆掉；`n>1` 并行采样还要替身。Beam search 已移出核心。

## 点名的后续（不是承诺）

**并行采样**（[#10980](https://github.com/vllm-project/vllm/pull/10980)）：`vllm:request_params_n`（每条完成请求的 `n`）；`vllm:request_max_num_generation_tokens`（sequence group 里最长输出；没有并行采样时等价于 `vllm:request_generation_tokens`）。

**投机解码：** `vllm:spec_decode_draft_acceptance_rate`、`vllm:spec_decode_efficiency`（Gauge）、`vllm:spec_decode_num_accepted_tokens` / `num_draft_tokens` / `num_emitted_tokens`（Counter）。接受率更该拆成 accepted / draft 两个 counter，像 prefix cache。当时点名 [#12193](https://github.com/vllm-project/vllm/pull/12193) 把 ngram 送进 V1。

**扩缩容 / 负载均衡：** Kubernetes Serving WG 的标准化文档、Inference Perf、[#5041](https://github.com/vllm-project/vllm/issues/5041) / [#12726](https://github.com/vllm-project/vllm/pull/12726)。要能看见饱和点：再提高请求率吞吐不再涨、延迟开始堆。Rob 的评论：估「让平均请求长度 > QPS」的最大并发——那才是把服务器喂饱。

**命名：** 冒号和 Prometheus「colon 留给 recording rule」对着干；多数名字带单位、不是全部；`_total` 后缀会在 OpenMetrics / Prometheus 文本格式之间被剥掉再加回去。

**再加指标时：** 容易加、难删；默认开才有用，但默认开就有性能税；维护成本随条数涨。优先 engine-core 事件 + 前端间隔，不要往内环塞活。别处的灵感：TGI、K8s autoscaling、[OTel Gen AI semantic conventions](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai)。

## Tracing

Metrics 是时间上的聚合；tracing 跟单条请求穿过组件。两件事都叫可观测性，设计页把它们拆开。

OpenTelemetry：[先加](https://github.com/vllm-project/vllm/pull/4687)、[再请回来](https://github.com/vllm-project/vllm/pull/20372)。旗标：`--oltp-traces-endpoint`、`--collect-detailed-traces`。用户文档在 `examples/observability/opentelemetry`。

详细 tracing 打开时才有这两条 Histogram（[#7089](https://github.com/vllm-project/vllm/pull/7089)）：

- `vllm:model_forward_time_milliseconds` — 这条请求在 batch 里时，花在 model forward 上的时间
- `vllm:model_execute_time_milliseconds` — execute：forward + worker 间 block/sync + CPU–GPU sync + sampling

`--collect-detailed-traces=all/model/worker`。文档自己写：可能贵、可能阻塞。span 上长这样：

```text
gen_ai.latency.time_in_scheduler
gen_ai.latency.time_in_model_forward
gen_ai.latency.time_in_model_execute
```

已经有 `inference_time` / `decode_time`。更高分辨率值不值得那笔开销，是未决问题。不要把 span 时长当成 TTFT。
