---
source: https://docs.nvidia.com/aiperf/benchmark-modes/load-generator-options-reference
lang: zh
fetched: 2026-08-31
---

# AIPerf 负载发生器参数

多种调度同时出现时的优先级：

1. `--fixed-schedule` / mooncake_trace → 按时间戳回放
2. `--user-centric-rate` → 每用户回合间隔（必须 `--num-users`）
3. `--request-rate` → 目标 QPS（`constant` / `poisson` / `gamma`）
4. 只开 `--concurrency` → 打满 / 饱和（N 以内尽快发）

| 模式 | 用途 |
|---|---|
| 只用 `--concurrency` | 最大吞吐 / 饱和 |
| `--request-rate` | 控 QPS 的压测 |
| `--fixed-schedule` | trace 回放 |
| `--user-centric-rate` + `--num-users` | 多轮 KV 压测（`turn_gap = num_users / user_centric_rate`） |

带速率时，`--concurrency` 是**上限**，不是负载驱动。不设 concurrency = 在途会话不封顶。

结束条件：`--request-count`、`--num-sessions` 或 `--benchmark-duration`。warmup 独立配置。`--prefill-concurrency` 必须开 `--streaming`。

```bash
aiperf profile --url localhost:8000 --model llama --concurrency 10 --request-count 100
aiperf profile --url localhost:8000 --model llama --request-rate 10 --arrival-pattern poisson --request-count 100
```

仓库：https://github.com/ai-dynamo/aiperf
