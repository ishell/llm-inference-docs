---
source: https://docs.vllm.ai/en/stable/design/metrics/
lang: zh
fetched: 2026-08-31
---

# Metrics 设计（vLLM V1）

`/metrics` 是怎么算出来的。运维对照表见 `production-metrics.md`。

## 两层

- **Server-level**（gauge/counter）：引擎状态——在跑/在等、KV 占用、token 累计。用来**解释**请求 SLO。
- **Request-level**（histogram）：TTFT、ITL、e2e、prompt/decode 长度——通常就是 SLO。

前缀 `vllm:`。仓库里有 Grafana 示例；它画的那一子集就是「重要指标」（e2e、TTFT、ITL、KV%、running/waiting、token 直方图、排队/prefill/decode 时间）。

## 计算放哪

V1：**EngineCore** 是 GPU 内环，尽量瘦。记账在前端（`AsyncLLM.output_handler_loop`），吃 `EngineCoreOutputs`。时间间隔必须用**同一进程的 `time.monotonic()`**（跨进程 monotonic 不能减）。

引擎事件：`QUEUED` → `SCHEDULED` → `NEW_TOKENS`（还有 `PREEMPTED`）。前端由此算 queue / prefill / decode / inference / ITL。**TTFT** 从前端 `arrival_time`（开始 tokenize）算，把输入处理算进去。

Decode 期 preemption 拉长 ITL/decode/inference；prefill 期 preemption 拉长 TTFT/prefill。

`--api-server-count > 1` 走 prometheus multiprocess，进程级 python_gc_* / process_* 会消失。

## 怎么对外

- 每 5 秒打日志：running/waiting、GPU cache %、prompt/gen token/s、最近 1k block 的 prefix-cache hit rate。
- Prometheus `/metrics`：Counter / Gauge / Histogram，带 `model_name`。bucket 还会改。
- `vllm:cache_config_info`：启动配置当 label。
- 可选 KV 驻留直方图（`--kv-cache-metrics-sample`）：block 寿命、驱逐前空闲、reuse 间隔。

HTTP 层来自 `prometheus_fastapi_instrumentator`。
