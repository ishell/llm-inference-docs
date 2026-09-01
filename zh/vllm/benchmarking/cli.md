---
source: https://docs.vllm.ai/en/stable/benchmarking/cli/
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Benchmark CLI — vLLM

英文对照：`en/vllm/benchmarking/cli.md`  
原文：https://docs.vllm.ai/en/stable/benchmarking/cli/

这一页的定位官方写得很清楚：偏 **功能回归 / 特性评测**。生产 serving 他们更推荐 **GuideLLM**（进度条、自动报告；数据集、请求格式、流量形态更灵活）。`vllm bench serve` 仍是仓库里那把自带客户端。NVIDIA 系列用 AIPerf 打同一类 OpenAI 兼容口。指标名字接近，**公式仍可能不同，不要直接横比数字。**

网格搜 `max-num-seqs` × `max-num-batched-tokens` 见 `auto-tune.md`。

## 在线：先起服务再打

```bash
vllm serve NousResearch/Hermes-3-Llama-3.1-8B

vllm bench serve \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path <path>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10
```

成功时客户端打印：Successful requests、duration、input/output tokens、request throughput、output / total token throughput，以及 Mean / Median / P99 的 **TTFT**、**TPOT**（不含首 token）、**ITL**。这些延迟都在 **benchmark 客户端**测——和 AIPerf 一样，尺子在门外。

`--plot-timeline` / `--plot-dataset-stats` 出 HTML 时间线和 ISL/OSL 分布。`--timeline-itl-thresholds` 默认 25ms、50ms，可改成 `2,5` 这种更严的色带。`--save-result` 把 JSON 留下。

## TTFT / TPOT / ITL（vLLM 自己的除法）

名字没有行业标准。对表时对测量点和公式，不要对名字。

```
TPOT = (e2e_latency − TTFT) / (output_tokens − 1)
```

- **TTFT**：发出请求到第一段流式输出。
- **ITL**：相邻两段流式输出之间的缝；统计是把所有成功请求里的这些缝聚在一起。
- **TPOT**：每个请求先算一遍（去掉首 token），再跨请求聚合。

普通 decode 时一段输出通常一个 token，ITL 和 TPOT 会很像。

**投机解码**时一段输出可以挤进多个被接受的 draft token。ITL 只记「段与段」的缝，不会给同一段里的 token 补零缝。TPOT 把整段 decode 时间摊到每一个输出 token 上。官方例子：两段输出、ITL 各 40 ms，第二段里有三个 token → 平均 ITL 仍是 40 ms；TPOT = `(180 − 100) / (5 − 1) = 20 ms/token`。同一场基准，两只秒表可以差一倍。别用 ITL 去骂投机解码的 TPOT。

## 负载怎么发

三只旋钮：

| 旗标 | 默认 | 含义 |
|---|---|---|
| `--request-rate` | `inf` | 目标 QPS；`inf` = 能发就发，打最大吞吐 |
| `--burstiness` | `1.0` | Gamma 形状；只在 rate 不是 `inf` 时生效 |
| `--max-concurrency` | 不限制 | 在途上限，模拟网关 / 负载均衡 |

`burstiness` 是 Gamma 的 shape，CV ≈ `1/√burstiness`：

- `0.1`：很突发（CV ≈ 3.16）— 压弹性
- `1.0`：泊松（CV = 1）— 像人
- `5.0`：更均匀（CV ≈ 0.45）— 给延迟画像

官方按用途给的座位：

| 用途 | burstiness | rate | max-concurrency |
|---|---|---|---|
| 最大吞吐（生产最常用） | 无效 | `inf` | 有限 |
| 像真人 | 1.0 | 中等 5–20 | 不限 |
| 压弹性 | 0.1–0.5 | 高 20–100 | 不限 |
| 延迟画像 | 2.0–5.0 | 低 1–10 | 不限 |
| 容量规划 | 1.0 | 可变 | 有限 |
| SLA | 1.0 | 目标 QPS | SLA 上限 |

`--request-rate inf --max-concurrency N`：用户能发多快发多快，门口只许站 N 个人。这就是「前面有限流器、后面引擎吃到饱」。启动日志里的 KV 句会告诉你理论上限：

```
GPU KV cache size: 15,728,640 tokens
Maximum concurrency for 8,192 tokens per request: 1920
```

`max_concurrency ≈ kv_cache_size / max_model_len`。容量规划把 `--max-concurrency` 放到这个数的 80–90%。贴着理论上限测，测的是 OOM 边缘，不是可持续的门厅。

还有：`--probe-request-rate` 旁路主流量发单 token 探针，**不受** `--max-concurrency` 限制，单独报延迟——用来看主负载对「不相干的客人」干扰有多大。请求速率还可以随时间 ramp（官方页有一节）。

## 数据集（表在，例子不逐抄）

ShareGPT、ShareGPT4V/Video、BurstGPT、Random / RandomMultiModal / RandomForReranking、Prefix Repetition、一串 HuggingFace（VisionArena、MMVU、InstructCoder、AIMO、MTBench、HumanEval、GSM8K、Blazedit、ASR…）、Spec Bench、SPEED-Bench、自定义 jsonl（文本 / 音频 / 图像）。

HuggingFace：`--dataset-name hf`，本地目录再用 `--hf-name` 标 Hub ID。

自定义文本 jsonl 每行一个 `prompt`。音频要 `prompt` + `audio` 路径；Whisper 走 `--backend openai-audio` + `/v1/audio/transcriptions`，Qwen2-Audio 走 chat + `--enable-multimodal-chat`。图像走 `openai-chat` + `/v1/chat/completions`；`--custom-ensure-client-side-data` 让客户端把本地图编成 data URL。

完整 wget 路径和每个数据集的命令，仍在官方页。这里不把 VisionArena / SPEED-Bench / BFCL / 长文档 QA / prefix caching / 哈希微基准再各抄一遍。

## 离线吞吐：`vllm bench throughput`

没有 HTTP。测引擎批处理，不是门厅。

```bash
vllm bench throughput \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset-name sonnet \
  --dataset-path vllm/benchmarks/sonnet.txt \
  --num-prompts 10
```

多模态离线要用 `--backend vllm-chat`，否则 image token 计数会少。Sonnet 数据集官方已标 deprecated，例子里还在用。

同页后半还有 structured output、embedding、reranker、多模态 processor 等特性评测入口——那是回归测试，不是生产 SLA 的主路径。生产数字请回到 AIPerf 或 GuideLLM，并用 `/metrics`（`../metrics/production-metrics.md`）解释为什么客户端看到那样的 TTFT。
