---
source: https://docs.nvidia.com/aiperf/getting-started/ai-perf-comprehensive-llm-benchmarking
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# AIPerf：五类真实打法

官方综合指南（标为 AIPerf v0.5.0，演示日期 2025-11-13，页上更新 2026-02-02）。演示集群已经拆掉。下面的数字是**官方案例**，用来看曲线长什么样，不是你机器上的成绩。入口见 `aiperf.md`。

他们当时的靶：Qwen3-0.6B，vLLM v0.11.0，8 路数据并行（8×H200，一卡一副本）。小模型是为了让秒表有东西可写，不是为了崇拜 0.6B。

本地跟一条副本即可：

```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000
export ENDPOINT_URL=localhost:8000
```

## 1. 固定 ISL/OSL，再扫出 Pareto

```bash
aiperf profile \
  --model qwen3-0.6b \
  --url $ENDPOINT_URL \
  --endpoint-type chat --streaming \
  --concurrency 100 --request-count 1000 \
  --isl 1000 --osl 500 \
  --tokenizer Qwen/Qwen3-0.6B
```

这是 100 个虚拟用户、每人大约 1000→500 token。官方演示表（8 副本合计）：TTFT avg 347 ms，request latency 2.1 s，ITL 3.57 ms，系统输出约 22.5K tok/s，RPS 45.7，1000/1000 成功。

同一负载扫并发 10 / 50 / 100 / 200 / 500（每档 `--artifact-dir` 分开）：

| Concurrency | 合计 TPS | TPS/GPU | TPS/User | TTFT avg |
|---|---|---|---|---|
| 10 | 3,045 | 1,522 | 365 | ~250 ms |
| 50 | 12,890 | 6,445 | 326 | ~270 ms |
| 100 | 22,521 | 11,261 | 285 | ~347 ms |
| 200 | 35,999 | 18,000 | 239 | ~420 ms |
| 500 | 29,836 | 14,918 | 129 | ~1,129 ms |

TPS/GPU = 合计 TPS / 8。c=200 是这张演示表上 GPU 效率的峰；c=500 两边一起塌——排队吃掉了吞吐，也吃掉了体验。不能同时把「每卡吐多少」和「每个人觉得有多快」拧到最大。选座位：要体验就 10–50；要平衡就 100–200；只看卡账就停在峰，不要越过塌陷。

## 2. 默认分位不够时，读 jsonl

控制台默认 P50 / P90 / P99。老板要 P75，不要重跑，读 `profile_export.jsonl`：

```python
import json
import numpy as np

ttft = []
with open("./artifacts/profile_export.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        ttft.append(rec["metrics"]["time_to_first_token"]["value"])
print(f"P75 TTFT: {np.percentile(ttft, 75):.2f} ms")
```

官方那 1000 条上的 P75 TTFT 是 422.87 ms。每行还有 ISL/OSL、ITL、`benchmark_phase`（`warmup` 或 `profiling`）。Pydantic 读法见官方 Working with Profile Export Files。

## 3. Mooncake trace：合成均匀负载看不见的世界

Mooncake 公开了 arXiv QA 的生产 trace：到达时刻、ISL/OSL、以及每 512 token 一块的 **hash_ids**。同一份文档在多轮里复用同一串 hash，于是可以谈 KV 命中，而不把用户原文泄漏出来。

官方概括这批 23608 条、60 分钟：中位约 6402 token，P99 约 6 万，最大超过 12 万；大约每分钟 393 条，并不均匀得像实验室。

```bash
curl -o mooncake_trace.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl

# 按原时间戳：测系统能不能跟上当时的门厅
aiperf profile ... --input-file mooncake_trace.jsonl \
  --custom-dataset-type mooncake_trace --fixed-schedule --streaming

# 去掉 --fixed-schedule：尽快发，测容量
```

他们把前 5 分钟（1765 条）加速 5× 回放约一分钟：ISL 从 890 到 32236，成功率 96%——75 条撞上 Qwen3-0.6B 的 32K 窗。合成 1000→500 不会告诉你这件事。Trace 暴露屋顶；均匀 ISL 只暴露实验室地板。

## 4. Goodput：吞吐里有多少还符合 SLA

同一段加速 trace，加上：

```bash
--goodput "time_to_first_token:370 request_latency:648"
```

官方演示：RPS 26.67，goodput 7.43——大约 28% 的请求**同时**满足两条 SLO。平均 TTFT 已经高于 370 ms，中位 latency 高于 648 ms。按吞吐买 38 台机器的人，若改用 goodput，账会变成大约 135 台。忽略 goodput 就是按一个用户正在受苦的数字扩容。

阈值按产品档位改：严（250/500）、演示用的中间档、松（600/2500）。公式见 `aiperf-metrics.md`。

## 5. 时间切片：平均值会把冷启动藏起来

```bash
--slice-duration 10
```

多两份文件：`profile_export_aiperf_timeslices.csv` / `.json`。官方 10 秒窗：第 0 片 TTFT 545 ms，第 1 片掉到 381 ms，之后稳在 344–388 ms。整场平均约 386 ms，把第一片的 +41% 冷启动税洗掉了。SLA 不要拿含冷启动的平均去签。

切片太短（<5s）会抖，太长（>60s）会看不清。常用 10–30s。要抓泄漏：`--benchmark-duration 3600 --slice-duration 300`，看后期 TTFT 是否单边上行。

## 附录（同一页上的周边）

- **集群内打**：客户端和副本放同一 K8s，用 ClusterIP。高并发时先塌的常常是客户端端口，不是 GPU。
- **取消**：`--request-cancellation-rate 20 --request-cancellation-delay 0.5`，测连接池和善后。
- **服务端 Prometheus**：默认可从 `--url` 发现；或 `--server-metrics`。
- **出图**：`aiperf plot`；`--dashboard` 默认 8050。
- **合成加速 / 拉长前缀**：`--synthesis-speedup-ratio`、`--synthesis-prefix-len-multiplier` 等，用来受控地折磨 KV。
- **User-centric**：`--user-centric-rate` + `--num-users` + `--shared-system-prompt-length`，见 `aiperf-load-generator.md`。

官方自己的结论：用例 1 给基线容量；生产能不能上，还要 trace、goodput、时间切片。三件套缺一，你会爱上实验室里那条不会在门厅活下来的曲线。
