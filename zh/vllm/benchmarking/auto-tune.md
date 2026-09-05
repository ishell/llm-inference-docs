---
source: https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# vLLM auto_tune.sh

仓库脚本：在 `max-num-seqs` × `max-num-batched-tokens` 网格上找吞吐最高、还能活在约束里的那一格。可选约束：P99 e2e 延迟、前缀缓存命中率。它调的是 `optimization.md` 里排在并行之前的那只 batch 旋钮，不是 TP/DP。

路径里**不要出现 `vllm` 这个词**。脚本会 `pkill -f vllm`，连调参自己一起杀掉。用 `tmux` / `screen`，这事会跑很久。

## 必填环境变量

在脚本头改，或运行时覆盖。`BASE` 必须是 vLLM 仓库**父目录**的绝对路径。

```bash
MODEL=meta-llama/Llama-3.3-70B-Instruct \
SYSTEM=GPU TP=8 DOWNLOAD_DIR='' \
INPUT_LEN=128 OUTPUT_LEN=2048 MAX_MODEL_LEN=2300 \
MIN_CACHE_HIT_PCT=0 MAX_LATENCY_ALLOWED_MS=500 \
NUM_SEQS_LIST="128 256" NUM_BATCHED_TOKENS_LIST="1024 2048 4096" \
bash auto_tune.sh
```

| 变量 | 含义 |
|---|---|
| `BASE` | vLLM 仓库父目录的绝对路径 |
| `MODEL` | Hugging Face id |
| `SYSTEM` | `TPU` 或 `GPU`（别的硬件可能存不了 profiler） |
| `TP` | tensor parallel |
| `DOWNLOAD_DIR` | 权重目录；空 = 默认下载路径 |
| `INPUT_LEN` / `OUTPUT_LEN` / `MAX_MODEL_LEN` | 合成请求尺寸与模型窗 |
| `MIN_CACHE_HIT_PCT` | 前缀缓存命中率下限，0–100；`0` = 关掉这条约束 |
| `MAX_LATENCY_ALLOWED_MS` | P99 e2e 上限。极大的数 ≈ 不管延迟 |
| `NUM_SEQS_LIST` | 要扫的 `max-num-seqs` |
| `NUM_BATCHED_TOKENS_LIST` | 要扫的 `max-num-batched-tokens` |

默认两份 list 按中等 ISL/OSL 估的。极短上下文（例如 20/20）往往要更大的 `max-num-seqs`。先装好对应环境；TPU 还要自己的 conda / torch_xla。自定义模型把 config 放到服务找得到的地方。

## 三种目标

1. **只追吞吐**：`MAX_LATENCY_ALLOWED_MS` 写成天文数字（官方示例 `100000000000`），`MIN_CACHE_HIT_PCT=0`。
2. **吞吐 + P99**：例如 `MAX_LATENCY_ALLOWED_MS=500`。
3. **再加前缀缓存**：`MIN_CACHE_HIT_PCT=60`。命中率是约束，不是奖励分。

## 它怎么走

1. 从 `gpu-memory-utilization=0.98` 往下找第一个不 OOM 的值。后面所有格子共用这只杯子。
2. 双重循环扫 seqs × batched-tokens。
3. 每一格：起服务 → `vllm bench serve --request-rate inf`。P99 已在上限内，就把这场吞吐当作这一格的上限。超了，就**降 request-rate** 直到延迟合格——找的是「还能守住 SLA 的最高可持续吞吐」，不是实验室里那条 inf 曲线。
4. 记下合法吞吐最高的一格，只给赢家留 profiler（GPU 常是 JSON trace，TPU 是 `.xplane.pb`）。

## 产物

`$BASE/auto-benchmark/<YYYY_MM_DD_HH_MM>/`：

- `vllm_log_...txt` / `bm_log_...txt` — 每一格的服务日志和 bench 日志
- `result.txt` — 每格一行，最后是 best_*
- `profile/` — 只属于赢家

```
max_num_seqs: 128, max_num_batched_tokens: 2048, request_rate: 10.0, e2el: 450.5, throughput: 9.8, goodput: 9.8
max_num_seqs: 128, max_num_batched_tokens: 4096 does not meet latency requirement 500
best_max_num_seqs: 256, best_num_batched_tokens: 2048, best_throughput: 12.5, profile saved in: ...
```

找不到合法格时：`best_max_num_seqs: 0`（服务没起来，或延迟上限太狠）。

## 批量：`batch_auto_tune.sh`

JSON 数组，每项一次 `auto_tune.sh`。需要 `jq`。可选第二参数把产物传到 GCS（要已登录的 `gcloud`）。

```bash
bash batch_auto_tune.sh runs_config.json [gs://bucket/path]
```

键名对应上面的变量（小写，脚本会转成大写环境变量）。跑完会**原地改**这份 JSON：补 `run_id`、`status`（`SUCCESS` / `FAILURE` / `WARNING_NO_RESULT_FILE`）、`results`、可选 `gcs_results`。

auto_tune 找到的是这一台机器、这一组 ISL/OSL、这一条 SLA 上的格子。换卡、换上下文，网格作废。把它的赢家当作 `serve.md` 里那两只旗标的起点，而不是永远的真理。
