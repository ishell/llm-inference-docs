---
source: https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# 系列第 3 篇：用 TensorRT-LLM 调性能

英文对照：`en/nvidia/performance-tuning/blog-03-tensorrt-llm.md`

第 1 篇给尺子，第 2 篇隔着 HTTP 打 NIM。本篇把 HTTP 请出房间：用 `trtllm-bench` 直接问引擎，再用 `trtllm-serve` 把调好的旋钮送上线。TensorRT-LLM 是开源推理引擎，旋钮很多；不调，它也会跑，只是未必按你在乎的那种快来跑。

## 先让 GPU 回到默认天气

```bash
sudo nvidia-smi -rgc
sudo nvidia-smi -rmc
nvidia-smi -q -d POWER          # 看功耗上限
nvidia-smi -i <gpu_id> -pl <wattage>   # 需要时再设
```

测之前把时钟和功耗策略拉回默认，免得昨天晚上的「省电」悄悄改了今天的成绩。

## 准备数据集

可用 `prepare_dataset` 造合成数据，或按文档写 jsonl，每行一次请求。例子：

```json
{"task_id": 1, "prompt": "Generate infinitely: This is the song that never ends, it goes on and on", "output_tokens": 128}
```

下文的示例数字来自 ISL/OSL = 128/128 的合成集。那首歌永不结束——测试里我们倒希望它在 128 个 token 处准时闭嘴。

## 跑 trtllm-bench

`trtllm-bench` 是 Python 工具，不经过完整 serving 栈。它会按自己认为「还不错」的默认把引擎拉起来。PyTorch flow：

```bash
trtllm-bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset dataset.jsonl \
  --tp 1 \
  --backend pytorch \
  --report_json results.json \
  --streaming \
  --concurrency $CONCURRENCY
```

checkpoint 不在缓存里时会从 Hugging Face 拉。结果写入 `results.json`，终端打印长表。下面是**样例数字，不是性能承诺**：

**PERFORMANCE OVERVIEW** 里要看的：

| 字段 | 样例 | 含义 |
|---|---|---|
| Request Throughput (req/sec) | 86.54 | 每秒完成的请求 |
| Total Output Throughput (tokens/sec) | 11077 | 输出 token/秒（含 context 侧的「output」口径，见下） |
| Total Token Throughput | 22154 | ISL+OSL 合计 token/秒 |
| Average TTFT (ms) | 162.67 | 等到第一个字 |
| Average TPOT (ms) | 7.33 | 之后每个字 |
| Per User Output Speed (tps/user) | 137.15 | 用户视角的吐字速度 |
| Per GPU Output Throughput | 11077 | 每卡输出吞吐 |

文中约定：Overview 里的 `Output` 包含 context 相关输出口径；`Total Token` = ISL+OSL；`Per user` / TTFT / TPOT 把每个请求当成一个「用户」，再做成分布。P50/P90/P95/P99 会分别给 TTFT、TPOT、GTPS、request latency。

**WORLD + RUNTIME** 里两个上限，是后面要抄到 serve 上的：

- **Max Runtime Tokens（`max_num_tokens`）**：一轮 iteration 引擎能处理的 token 上限 = 所有 context 请求的输入 token 之和 + 每个 generation 请求各 1 token。样例 7680。
- **Max Runtime Batch Size（`max_batch_size`）**：一轮最多多少个请求。样例 3840。

两者会互相卡住。例如：一个 128 token 的 context + 四个 generation（共 132 token），`max_num_tokens=512` 且 `max_batch_size=5`——你会先撞上 batch size，哪怕 token 预算还没用完。厨房的炉子和盘子，哪样先满哪样说了算。

调度策略样例是 `GUARANTEED_NO_EVICT`，KV memory 90%。

## 你到底在优化谁

先问三个问题，答案会指向不同的山峰：

- 要每个用户吐字快？（`Per User Output Speed`）
- 要离线把文本榨干？（系统吞吐）
- 要第一个字尽快回来？（TTFT）

本篇选择优化**用户体验**：context 结束之后，字回到人那里的速度。目标大约 **50 tok/s/user**（约 20 ms/token）。`--concurrency` 限制在途请求数，用来画出「每 GPU 吞吐 vs 每用户速度」。

原文 Figure 1（128/128，Llama-3.1 8B）：

- 并发升高，每 GPU 吞吐变好（系统更忙），每用户速度变差（每个人更挤）。
- **FP16** 大约 **256 并发** 仍有 ~72 tok/s/user，再往上会跌破 50 的预算。
- **FP8** 优化 checkpoint 能在同一预算里撑到 **512 并发** ~66 tok/s/user。

量化在这里不是「差不多快一点」，是同一间屋子里能多站一倍的人。

你可以：

- 把 `max_batch_size` 钉死在 512。超过的请求排队，TTFT 会涨——门厅变长，不是刀变钝。
- 换数据集、换精度、再扫一遍。多卡用 `--tp` / `--pp` / `--ep`。更细的旋钮走 `--extra_llm_api_options`。

## 用 trtllm-serve 上线

`trtllm-serve` 会起 OpenAI 兼容端点，但**不会自动套用** bench 刚扫出来的那套。要把上限亲手递过去：

```bash
trtllm-serve serve nvidia/Llama-3.1-8B-Instruct-FP8 \
  --backend pytorch \
  --max_num_tokens 7680 \
  --max_batch_size 3840 \
  --tp_size 1 \
  --extra_llm_api_options llm_api_options.yml
```

`llm_api_options.yml` 对齐 CUDA graph：

```yaml
cuda_graph_config:
    max_batch_size: 3840
    padding_enabled: true
```

起来后：

```
INFO: Application startup complete.
INFO: Uvicorn running on http://localhost:8000
```

再用 GenAI-Perf / AIPerf，或移植版 `benchmark_serving.py` 验收。官方说未来会让 `trtllm-bench` 自己拉起调优过的 server。在那之前，bench 是实验室，serve 是门面，数字要人搬过去。

更细的旋钮：同目录 [调优手册](trtllm-tuning-guide.md)。DeepSeek-R1 有单独的 tuning 文。Nsight Systems 可分析一次 forward 到底把时间花在哪。
