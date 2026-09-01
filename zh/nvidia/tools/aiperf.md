---
source: https://github.com/ai-dynamo/aiperf
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# NVIDIA AIPerf

GenAI-Perf 的后继。客户端生成式基准：对着已经起来的推理服务发请求，把 TTFT、ITL、TPS、RPS、goodput 打成一张表。尺子在客户端；服务必须先活着。

NIM 实操走 `../benchmarking/nim-04-aiperf.md`。调度旗标见 `aiperf-load-generator.md`。公式见 `aiperf-metrics.md`。五类真实场景见 `aiperf-comprehensive.md`。整页 CLI 不搬，官方在 https://docs.nvidia.com/aiperf/reference/command-line-options 。

仓库：https://github.com/ai-dynamo/aiperf  
文档站：https://docs.nvidia.com/aiperf/

OpenAI 兼容的服务都能打：NIM、vLLM、`trtllm-serve`、Triton、Ollama、SGLang、Dynamo。换引擎，不要换尺子的定义。

## 安装

```bash
python3 -m venv venv
source venv/bin/activate
pip install aiperf
```

Linux **aarch64** 上依赖 `crick` 只有 sdist，装之前要有 C 编译器（Debian/Ubuntu：`build-essential`）。x86_64 / macOS / Windows 走预编译 wheel。

可选 extras：`aiperf[mlflow]`、`aiperf[otel]`、`aiperf[wandb]`，或一次装齐 `aiperf[mlflow,otel,wandb]`。它们把成绩流到追踪系统；不改秒表本身。

NIM 手册喜欢用 Triton SDK 容器再 `pip install aiperf`。本地玩具可以用 Ollama：

```bash
docker run -d --name ollama -p 11434:11434 \
  -v ollama-data:/root/.ollama ollama/ollama:latest
docker exec -it ollama ollama pull granite4:350m
```

## 最小命令

```bash
aiperf profile \
  --model "granite4:350m" \
  --streaming \
  --endpoint-type chat \
  --tokenizer ibm-granite/granite-4.0-micro \
  --url http://localhost:11434 \
  --request-count 10
```

对着 vLLM / NIM 时，把 `--url` 换成 `localhost:8000`，`--model` 换成服务认的名字，`--tokenizer` 换成同一套分词器。**Tokenizer 必须对。** 错的分词器会把 ISL/OSL 量歪，后面所有除法都跟着撒谎。

`--streaming` 几乎总要开。TTFT、ITL、TTST 都要求流式、且至少有一包非空内容。关流式，你测到的主要是整段 e2e。

官方 README 里那张 CPU-only Ollama 表**不是**官方成绩。数字只说明表格长什么样：TTFT / TTST / TTFO、Request Latency、ITL、单用户 TPS、ISL/OSL、系统 TPS、RPS、Request Count。

产物默认写到 `artifacts/<model>-<endpoint>-concurrencyN/`：

- `profile_export_aiperf.json` / `.csv` — 汇总
- `profile_export.jsonl` — 每请求一行，要算 P75 就读这个
- `logs/aiperf.log`

## 三种 UI

`dashboard`（实时 TUI）、`simple`（进度条）、`none`（无头）。复制 TUI 选中文本可能不可靠，官方让按 `c` 拷全部日志。

## 它怎么拆成三层

文档站 Architecture 把 AIPerf 切成三架飞机，中间用 ZMQ 说话：

| 平面 | 谁 | 干什么 |
|---|---|---|
| Control | SystemController、Timing Manager、Dataset Manager、Worker Manager | 决定发什么、何时发、发多少 |
| Data | Workers ↔ 推理服务 | 真正的 HTTP 往返 |
| Analytic | Record Processors、Records Manager、GPU / Server Metrics | 算指标、收遥测 |

请求生命周期：装数据集 → 可选 warmup（成绩丢掉）→ 发 credit 给 worker → 打服务 → 收时序 → 并行算指标 → 汇总导出。

**Credit** 是一张「准许发一条」的票。Timing Manager 按调度模式发票：固定时间戳、目标 QPS、或每用户回合间隔。Worker 没票就等。这比「能发多快发多快」更接近真实到达；服务变慢时，票会自然堵住，而不是客户端自己再叠一层假延迟。

Worker 之间不共享状态。多轮对话的上下文只活在那个 worker 里。数据集走内存映射文件，避免把 prompt 在进程间复印。

当前只支持**单机多进程**。文档里出现的 Kubernetes 字样是前景；这一版没有注册分布式 K8s 执行，不要把它当已交付。

Telemetry（OTel / MLflow）是 Analytic 旁边的边车：独立子进程、有界队列。队列满了丢最老的事件，不堵热路径。测到的是推理，不是你的 collector 有多慢。

## 支持的 API

OpenAI：chat、completions、embeddings、audio、images。NIM embeddings / rankings。自定义前端或 Jinja2 payload 也能打。插件系统覆盖 endpoint、dataset、transport、metric。

公开数据集：ShareGPT、Mooncake / Bailian / BurstGPT 一类 trace、以及视觉 / ASR / 投机解码评测集。需要哪一种，去官方 Tutorials；这里不把目录再抄一遍。

## GPU 与服务端指标

可选：DCGM Exporter、PyNVML（NVIDIA）、amdsmi（AMD）。服务端 Prometheus（vLLM `/metrics` 等）可自动发现，也可用 `--server-metrics` 点名。TRT-LLM 默认 `/metrics` 有时是 iteration-stats JSON 而不是 Prometheus——AIPerf 会再探 `/prometheus/metrics`，两边都不是就警告一次然后关掉，避免整场基准被遥测拖死。

跑完以后：`aiperf plot` 出图；`--dashboard` 开交互页（默认 `localhost:8050`）。

## 已知陷阱

- `--output-tokens-mean` **不能保证**真的吐那么长，除非你用 `--extra-inputs` 把 `ignore_eos` / `min_tokens` 传给支持它们的服务。NIM 手册就是这么干的。
- 并发极端高（官方说通常 >15000）可能把客户端端口打光。那是客户端的门厅塌了，不是模型变慢。
- 配置非法时，进程可能挂住。杀掉，查配置。
- 热身和正式成绩分开。演员还在对词，不要记进票房。

从 GenAI-Perf 迁过来：命令几乎同构（`profile`、`--streaming`、concurrency / request-rate）。新项目用 AIPerf。概念（空首包不算 TTFT、ITL 不含 TTFT）还在，公式以 `aiperf-metrics.md` 为准。
