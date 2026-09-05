---
source: https://docs.vllm.ai/en/stable/design/metrics/
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Metrics 设计 — vLLM V1

运维对照表：`production-metrics.md`。这一页回答的是：这些直方图是在哪间屋子里、用哪只钟算出来的。

## 两层

- **Server-level**（gauge / counter）：引擎此刻的天气——在跑、在等、KV 占用、token 累计。用来**解释**为什么某条请求的 SLO 破了。
- **Request-level**（histogram）：TTFT、ITL、e2e、prompt/decode 长度。通常这些才是 SLO 本身。

前缀 `vllm:`。Grafana 示例仪表盘画的那一子集，就是「先接这些」。

## 计算放哪

V1 把 **EngineCore** 当成 GPU 内环，尽量瘦。记账放在前端：`AsyncLLM.output_handler_loop` 吃 `EngineCoreOutputs`。

时间间隔必须用**同一进程的 `time.monotonic()`**。跨进程的 monotonic 不能相减——两只钟各说各的「从开机起过了多久」，差值没有意义。这就是为什么 TTFT 不在 GPU 进程里减 `arrival_time`。

引擎事件：`QUEUED` → `SCHEDULED` → `NEW_TOKENS`，外加 `PREEMPTED`。前端由此拼出 queue / prefill / decode / inference / ITL。

**TTFT** 从前端 `arrival_time` 起算，那是开始 tokenize 的时刻，所以输入处理算进第一段等待。客户端 AIPerf 的 TTFT 从「HTTP 发出」起算。屋里多了 tokenize，门外多了网线。对不上时，先问哪一段不在另一段里。

Decode 期被抢占：ITL、decode、inference 被拉长。Prefill 期被抢占：TTFT、prefill 被拉长。`vllm:num_preemptions` 往上爬，先看是哪一段在被请出房间。

`--api-server-count > 1` 走 Prometheus multiprocess。进程级 `python_gc_*` / `process_*` 会从 `/metrics` 消失——不是采集坏了，是多进程模式不按单进程那套暴露。记账曾经在 engine-core 进程（[#7279](https://github.com/vllm-project/vllm/pull/7279)）；现在在 API server，multiprocess **只** 在 `--api-server-count > 1` 时用（[#17546](https://github.com/vllm-project/vllm/pull/17546)）。

库的路径：`aioprometheus` → `prometheus_client`。迁移时 HTTP 指标短暂消失，又用 `prometheus_fastapi_instrumentator` 请回来（[#15657](https://github.com/vllm-project/vllm/pull/15657)）：`http_requests_total`、请求/响应体积、时长——门开了几次，不是字与字之间的缝。

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

文档站图：普通情况；decode 期抢占（ITL / decode / inference 被拉长）；prefill 期抢占（TTFT / prefill 被拉长）。

前端按 engine-core 的每一拍收集：本拍新 token、本拍完成的 prefill 的 prompt token、本拍新调度请求的 queue 间隔、完成 prefill 的 prefill 间隔 / TTFT、本拍所有人的 ITL。

## KV 驻留

`--kv-cache-metrics-sample` 把开销压得很小。抽到的块记：lifetime（分配 → 驱逐）、idle-before-evict（最后一次 touch → 驱逐）、reuse gaps。Prometheus：`vllm:kv_block_lifetime_seconds`、`vllm:kv_block_idle_before_evict_seconds`、`vllm:kv_block_reuse_gap_seconds`。engine core 只在 `SchedulerStats` 里送原始驱逐事件；前端做成 Prometheus 观察，日志开着时也走 `LLM.get_metrics()`。

## 怎么对外

- `LoggingStatLogger`：大约每 5 秒一条 INFO——running/waiting、GPU cache %、过去 5 秒的 prompt/gen tok/s、最近 **1k** 次 block 查询的 prefix-cache hit rate。
- `PrometheusStatLogger`：`/metrics`。Counter 到重启前只增；Gauge 可上可下；Histogram 按桶计数。每条序列带 `model_name`。**bucket 还会改。** 页上 TTFT 例子：`le="0.02"` 已经有 140 里的 13——不要把桶抄进仪表盘当法律。
- `vllm:cache_config_info`：Info 指标的思路（Gauge 钉死为 1），启动配置当 label（`block_size`、`cache_dtype`、`cpu_offload_gb`、`enable_prefix_caching`、`gpu_memory_utilization`…）。`prometheus_client` 在 multiprocess 下从未支持 Info，所以用 Gauge + `multiprocess_mode="mostrecent"`。
- LoRA：`vllm:lora_requests_info` 这个 Gauge 的 **值是墙上时钟**，每拍更新。label：`running_lora_adapters` / `waiting_lora_adapters` 是逗号分隔的 `adapter=count` 字符串，外加 `max_lora`。页上自己说把计数塞进 CSV「quite misguided」。`multiprocess_mode="livemostrecent"`。[#9477](https://github.com/vllm-project/vllm/pull/9477)；至少有一个下游（Gateway API Inference Extension）。删之前要打招呼。
- Prefix cache：每次查询记问了多少 token、命中多少。日志给最近 **1k** 次的命中率。Prometheus 该留 **counter**，让 PromQL 自己选窗口：`rate(cache_query_hit[5m]) / rate(cache_query_total[5m])`——不要做成命中率 Gauge。

## 废弃

删指标不能轻（`vllm:avg_prompt_throughput_toks_per_s` 废弃、删除，然后有用户发现没了）。页上的规矩：谨慎；`/metrics` 的 HELP 里写废弃；用户文档和发版说明；先藏到 CLI 逃生口后面一阵。项目级 deprecation policy 另有一页。生产页口径：`X.Y+1` 隐藏，可用 `--show-hidden-metrics-for-version=X.Y` 再看见，`X.Y+2` 删除。

点名的遗物：从未实现的 `vllm:tokens_total`（直接删）；重复的 queue-time 序列；prefix-cache 命中率 Gauge vs counter；还在长的 KV-offload 指标。

## 点名的后续（不是承诺）

并行采样指标；跟功能页对齐的投机解码计数；扩缩容 / 负载均衡信号；命名一致性（`_total` 后缀、`vllm:` 前缀）；再加指标时优先 engine-core 事件 + 前端间隔，不要往内环塞活。

## Tracing

OpenTelemetry 是跟 Prometheus 直方图分开的路。那一节区分 model-forward 和 execute time——不要把 span 时长当成 TTFT。

HTTP 层仍是门开了几次。把它当 TTFT 的替身，会得到一个很有礼貌的谎言。

- 大约每 5 秒打日志：running/waiting、GPU cache %、prompt/gen token/s、最近 1k block 的 prefix-cache hit rate。这是不用 Grafana 也能看见的心跳。
- Prometheus `/metrics`：Counter / Gauge / Histogram，带 `model_name`。**bucket 还会改。**
- `vllm:cache_config_info`：启动配置当 label（block size、前缀缓存开关、`gpu_memory_utilization`…）。换配置等于换序列，不要把两条曲线叠在同一 label 上吵。
- 可选 KV 驻留直方图（`--kv-cache-metrics-sample`）：block 寿命、驱逐前空闲、reuse 间隔。占用百分比说「房间满了」；驻留直方图说「人是住着还是路过」。

HTTP 层来自 `prometheus_fastapi_instrumentator`（`http_requests_total`、时长…）。那是门开了几次，不是字与字之间的缝。把它当 TTFT 的替身，会得到一个很有礼貌的谎言。
