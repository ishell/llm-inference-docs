---
source: https://docs.nvidia.com/aiperf/benchmark-modes/load-generator-options-reference
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# AIPerf 负载发生器

官方页是一张兼容性矩阵：哪些旗标能一起用，哪些会当场报错。调度是这场基准的性格。选错模式，你测到的不是模型，是你自己发明的到达过程。

入口与安装见 `aiperf.md`。公式见 `aiperf-metrics.md`。

## 四种发请求的方式

多种调度同时出现时，优先级从上到下，下面的会被忽略或直接报错：

1. `--fixed-schedule` / `mooncake_trace` → 按数据集里的时间戳回放
2. `--user-centric-rate` → 每用户回合间隔（必须 `--num-users`）
3. `--request-rate` → 目标 QPS（`constant` / `poisson` / `gamma`）
4. 只开 `--concurrency` → 打满 / 饱和（N 以内尽快发）

| 你想问什么 | 用什么 |
|---|---|
| 这条 trace 当时长什么样？ | `--fixed-schedule` + mooncake_trace |
| KV 还在不在、多轮间隔稳不稳？ | `--user-centric-rate` + `--num-users` |
| 固定 QPS 下延迟如何？ | `--request-rate`（可选 `--arrival-pattern`） |
| GPU 能被喂到多饱？ | 只开 `--concurrency`，不要带 rate |

带速率时，`--concurrency` 是**天花板**：票按速率发，在途满了就等。不设 concurrency = 在途会话不封顶。官方提醒：user-centric 模式下，concurrency 至少要 ≥ `--num-users`，否则有的「用户」永远排不上队。

## 到达过程

只跟 `--request-rate` 走：

- `constant`：间隔 = `1/rate`
- `poisson`：指数间隔（`--request-rate` 的默认）
- `gamma`：用 `--arrival-smoothness` 调平滑；其它 pattern 上开这个旗标会报错
- `concurrency_burst`：没设 rate 时自动变成「尽快发」

`--request-rate-ramp-duration` 只属于 request-rate，不能跟 fixed-schedule 或 user-centric 一起。`--concurrency-ramp-duration` 三种调度都能用。

## 怎么停

至少要有一个结束条件：

- `--request-count` 与 `--num-sessions` **互斥**
- `--benchmark-duration` 才能配 `--benchmark-grace-period`（默认 30s；user-centric + duration 时 grace 默认无穷）

合成数据集、不设 request-count 时，官方自动取 `max(10, concurrency * 2)`。

## Prefill 上限

`--prefill-concurrency` 必须 `--streaming`，且必须 ≤ `--concurrency`。长上下文时，同时做 prefill 的人太多，显存先塌。这是给 decode 留座位，不是再发明一种 QPS。

## Warmup

热身子流程**内部永远走 rate-based 调度**，和主基准的模式无关。停法：`--warmup-request-count` / `--warmup-duration` / `--num-warmup-sessions`（前两个 count 类互斥）。未指定的 warmup 并发、速率、到达过程，回落到主基准对应旗标。`--warmup-grace-period` 默认无穷，但必须先启用了 warmup。

## 其它会咬人的组合

| 旗标 | 注意 |
|---|---|
| `--num-users` | 只能配 `--user-centric-rate` |
| `--session-turns-mean` | user-centric 要求 ≥ 2（单轮请用 request-rate） |
| `--dataset-sampling-strategy` | 与 `--fixed-schedule` 不相容 |
| `--fixed-schedule-auto-offset` 与 `--fixed-schedule-start-offset` | 互斥；都要求 `--fixed-schedule` |
| `--request-cancellation-delay` | 必须先有 `--request-cancellation-rate`（0–100 的百分比） |
| `--url` 可重复 | `--url-strategy` 默认 `round_robin` |

`--user-centric-rate` 的关键公式：

```
turn_gap = num_users / user_centric_rate
```

15 个用户、rate=1.0 → 每人每 15 秒一轮。这是在测 KV 的寿命，不是在测峰值 TPS。完整 KV 场景还要配 `--shared-system-prompt-length` 等，见官方 User-Centric Timing Tutorial。

## 例子

饱和：

```bash
aiperf profile --url localhost:8000 --model llama \
  --concurrency 10 --request-count 100
```

10 QPS、泊松到达：

```bash
aiperf profile --url localhost:8000 --model llama \
  --request-rate 10 --arrival-pattern poisson --request-count 100
```

恒定到达 + 并发天花板 + 按时间停：

```bash
aiperf profile --url localhost:8000 --model llama \
  --request-rate 20 --arrival-pattern constant \
  --concurrency 5 --benchmark-duration 60
```

prefill 限流（长上下文）：

```bash
aiperf profile --url localhost:8000 --model llama \
  --concurrency 20 --prefill-concurrency 5 --streaming \
  --benchmark-duration 60
```

Mooncake trace 回放（可加时间窗，单位毫秒）：

```bash
aiperf profile --url localhost:8000 --model llama \
  --input-file trace.jsonl --custom-dataset-type mooncake_trace \
  --fixed-schedule \
  --fixed-schedule-start-offset 60000 \
  --fixed-schedule-end-offset 120000
```

多轮 KV：

```bash
aiperf profile --url localhost:8000 --model llama \
  --user-centric-rate 1.0 --num-users 15 \
  --session-turns-mean 20 --streaming --benchmark-duration 300
```

## 常见报错（官方原文方向）

- user-centric 不能和 `--request-rate` / `--arrival-pattern` 一起
- user-centric 必须 `--num-users`，且 `--session-turns-mean >= 2`
- grace period 只能配 duration
- warmup grace 只能在启用了 warmup 之后
- prefill-concurrency 必须 streaming
- arrival-smoothness 只能配 gamma
- request-count 与 num-sessions 不能同时设
- `--num-users` 不能单独出现
- `--request-rate-ramp-duration` 不能配 user-centric 或 fixed-schedule

完整类型 / 默认值表在原页「Full Options Reference」。这里不把每一行再抄一遍；上表已经覆盖会改测量含义的那些。
