---
source: https://docs.vllm.ai/en/stable/design/metrics/
lang: zh
voice: literary-study
fetched: 2026-09-01
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

`--api-server-count > 1` 走 Prometheus multiprocess。进程级 `python_gc_*` / `process_*` 会从 `/metrics` 消失——不是采集坏了，是多进程模式不按单进程那套暴露。

## 怎么对外

- 大约每 5 秒打日志：running/waiting、GPU cache %、prompt/gen token/s、最近 1k block 的 prefix-cache hit rate。这是不用 Grafana 也能看见的心跳。
- Prometheus `/metrics`：Counter / Gauge / Histogram，带 `model_name`。**bucket 还会改。**
- `vllm:cache_config_info`：启动配置当 label（block size、前缀缓存开关、`gpu_memory_utilization`…）。换配置等于换序列，不要把两条曲线叠在同一 label 上吵。
- 可选 KV 驻留直方图（`--kv-cache-metrics-sample`）：block 寿命、驱逐前空闲、reuse 间隔。占用百分比说「房间满了」；驻留直方图说「人是住着还是路过」。

HTTP 层来自 `prometheus_fastapi_instrumentator`（`http_requests_total`、时长…）。那是门开了几次，不是字与字之间的缝。把它当 TTFT 的替身，会得到一个很有礼貌的谎言。
