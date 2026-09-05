---
source: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# GenAI-Perf

英文对照：[en/nvidia/tools/genai-perf.md](../../../en/nvidia/tools/genai-perf.md)

官方横幅：**正在停更。** 新项目用 AIPerf（`aiperf.md`）。命令几乎同构，概念仍适用。NIM Performance 页面上的老数字，有一批就是用这套仪式测出来的。换工具，不要换尺子。

客户端打已经起来的生成式服务：output token throughput、TTFT、TTST、ITL、request throughput。服务必须先活着。支持 OpenAI chat/completions、Triton TensorRT-LLM backend；也可用自定义前端或 Jinja2 payload 打自家 API。自定义前端更自由，Jinja2 只改信封。

## 安装

```bash
pip install genai-perf   # 需要本机已有 CUDA 12
```

或 Triton SDK 容器（官方示例 `RELEASE=25.01`）：

```bash
docker run -it --net=host --gpus=all \
  nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
genai-perf --help
```

它底下仍会叫 Perf Analyzer。LLM 看这一页；非生成式看 `perf-analyzer.md`。

## 最小例子：Triton 上的 GPT-2

官方 quickstart：TRT-LLM 容器里 `triton import -m gpt2 --backend tensorrtllm`，再 `triton start`。SDK 容器里：

```bash
genai-perf profile \
  -m gpt2 \
  --backend tensorrtllm \
  --streaming
```

表示例（GPT-2 / Triton）：TTFT 16.26 ms，ITL 1.85 ms，request latency 499 ms，OSL ~262，ISL ~550，输出约 521 tok/s，RPS ~1.99。数字只说明表格列什么。

## YAML 配置

```bash
genai-perf create-template          # 默认 genai_perf_config.yaml
genai-perf create-template -v       # 带注释
genai-perf config -f genai_perf_config.yaml
genai-perf config -f genai_perf_config.yaml \
  --override-config --warmup-request-count 100 --concurrency 32
```

模板里的 endpoint 段：`model_selection_strategy`（`round_robin` / `random`）、`backend`、`type`（默认 `kserve`）、`streaming`、`url`、`grpc_method`。打 Triton TensorRT-LLM 时，模型 config 要把 `exclude_input_in_output` 设 true，否则输出会把输入 echo 回来，OSL 会被灌胖。

`--override-config` 用来改两三只旋钮而不改 YAML。

`genai-perf analyze` 可扫一组刺激（并发 / 速率），一次命令多场景。`process export files` 用来把分布式多次导出合成一张表。细节回原页 Analyze / Process Export Files。

## 图

默认不开。`--generate-plots`：TTFT 分布、request latency、TTFT vs ISL、ITL vs token 位置、ISL vs OSL。

## 输入从哪来

合成：

- `--num-dataset-entries`：样本池大小，用完循环
- `--synthetic-input-tokens-mean` / `--stddev`
- `--random-seed`
- `--request-count`、`--warmup-request-count`

文件：`--input-file`，JSON 对象（prompt 或图像路径）。

任意数据集还可：`--num-prefix-prompts` + `--prefix-prompt-length`（测前缀 KV）；`--output-tokens-mean` / `--stddev`；`--output-tokens-mean-deterministic`（当时只宣称 Triton 上更准）；`--extra-inputs name:value` 可重复（`stream:true`、`max_tokens:5`）。

LLM 没有客户端 batch：一条请求就是一次推理。embeddings / rankings 才有 `--batch-size-text N`。

## Mooncake payload

`--input-file payload:<file>` 走固定时间表。JSONL 字段：必有 `timestamp`（毫秒）；可选 `input_length`、`output_length`、`text_input`、`session_id`、`hash_ids`、`priority`。`hash_ids` 映射到 512 token 一块的合成块——同一 hash 同一段输入，用来压 KV 和投机解码。

合成正弦到达可用 Dynamo 的 `sin_synth.py`（时长、rate 上下界与周期、两档 ISL/OSL）。生产流量建议自己在服务前加一只足迹收集器，官方示例只是假想 `Logger("mooncake_traffic.jsonl")`。

多轮延迟可用 `--session-delay-ratio` 整体缩放，不必改 payload。

## 鉴权

```
-H "Authorization: Bearer ${API_KEY}" -H "Accept: text/event-stream"
```

## 指标（与 AIPerf 对齐的那几只）

| 指标 | 含义 | 聚合 |
|---|---|---|
| TTFT | 发请求 → 第一包响应 | avg/min/max/p99/p90/p75 |
| TTST | 第一包 → 第二包 | 同上 |
| ITL | 中间响应间隔 / 后一包生成的 token 数 | 同上 |
| Output Token Throughput Per User | （不含首 token 的输出）/ 生成阶段时长 | 同上 |
| Request Latency | 发请求 → 最后一包 | 同上 |
| OSL / ISL | 该请求的输出 / 输入 token | 同上 |
| Output Token Throughput | 整场输出 token / 基准时长 | 整场一个值 |
| Request Throughput | 完成条数 / 基准时长 | 整场一个值 |

空内容的首包不算 TTFT。ITL **不含** TTFT。系统吞吐的分母是整场墙钟，和 per-user（1/ITL）不是同一只表。NIM 第 2 章把这件事写成人话。

GPU 遥测（功耗、利用率、显存、温度、时钟、ECC、NVLink、PCIe…）来自同机 DCGM Exporter 的 `/metrics`。`--verbose` 才打到控制台。自定义指标 CSV 见官方 GPU Telemetry tutorial。

## 旗标分组（不逐条搬 CLI）

- **Endpoint**：`-m/--model`（多 LoRA 才需要多个名字）、`--model-selection-strategy`、`--backend {tensorrtllm,vllm}`、`--endpoint`、`--endpoint-type`（默认 `kserve`）、`--streaming`、`-u/--url`、`--grpc-method`、`--server-metrics-urls`
- **负载**：`--concurrency`、`--request-rate`、`--fixed-schedule`、`--measurement-interval` / `-p`、`--stability-percentage` / `-s`（默认 999，几乎等于「别等稳态」）
- **产物**：`--artifact-dir`（默认 `artifacts`）、`--profile-export-file`（Perf Analyzer 的 json；GenAI-Perf 另写 `*_genai_perf.json/csv`）、`--generate-plots`、checkpoint
- **会话**：`--num-sessions`、`--session-concurrency`、turns / delay 的 mean/stddev
- **Tokenizer**：`--tokenizer`，默认等于模型名

稳态判定：最近 3 个测量窗的 max/min，吞吐和延迟都落在 stability 百分比内。默认 999% 等于几乎不稳态就收工——实验室数字会偏乐观。

音频相关旗标（时长、wav/mp3、采样率、声道）给音频模型；LLM 主线用不到。

完整 CLI 仍在原页。新工作请把同一组概念接到 `aiperf.md`。
