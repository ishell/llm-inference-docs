---
source: https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# DGX Spark：128GB 统一内存上的小并发 NVFP4，不是机房卡

英文对照：[en/vllm/blog/serving/dgx-spark.md](../../../../en/vllm/blog/serving/dgx-spark.md)  
原文：https://vllm.ai/blog/2026-06-01-vllm-dgx-spark  
2026-06-01。署名 **Inferact**。桌上的 **GB10** / `sm_121`，不是机房那张卡。干活的例子：[Nemotron-3-Super-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)。模型日笔记：[nemotron-3-super.md](nemotron-3-super.md)。Docker Model Runner 后来把 Spark 写进路线图：[docker-model-runner.md](docker-model-runner.md)。数字是办公室里一台 Spark 的 demo，不是排行榜。

**TL;DR：**

- CPU、GPU、OS、容器、权重、KV 抢 **同一池 128 GB**。`--gpu-memory-utilization` 必须留余量；菜谱用 **0.85**。
- `--max-num-seqs 4`。再高，单 token 带宽税压过 continuous batch，TTFT 会尖。
- 适合 **约 100–130B NVFP4 MoE**、**约 10–15B** active。Dense、高并发也能跑，只是跟带宽和统一内存不对齐。
- 官方镜像 `vllm/vllm-openai:cu130-nightly` 是**轨道**不是 pin。部署要钉 digest。
- 五场景、warmup 后中位：Decode **22.7–23.7 tok/s**。Prefill 从 **140** 到长 prompt **约 1,900 tok/s**。首请求 Inductor/FlashInfer JIT 约 **25 s**。safetensor 加载 **10–15 分钟**。

原文封面照片（未收录）：*vLLM running Nemotron-3-Super on the DGX Spark for a demo at the Inferact office.*

本地图（原文版权仍归原站；学习对照用）：

![dgx spark vllm serving architecture](../../../../assets/vllm/blog/serving/dgx-spark/02-dgx-spark-vllm-serving-architecture.svg)

![gb10 unified memory sm121 map](../../../../assets/vllm/blog/serving/dgx-spark/03-gb10-unified-memory-sm121-map.svg)

![dgx spark model fit decode rate](../../../../assets/vllm/blog/serving/dgx-spark/04-dgx-spark-model-fit-decode-rate.svg)

![spark vllm config stability performance slider](../../../../assets/vllm/blog/serving/dgx-spark/05-spark-vllm-config-stability-performance-slider.svg)

![vllm spark game demo flow](../../../../assets/vllm/blog/serving/dgx-spark/07-vllm-spark-game-demo-flow.svg)

![dgx spark vllm benchmark sweep](../../../../assets/vllm/blog/serving/dgx-spark/08-dgx-spark-vllm-benchmark-sweep.svg)

## 技术摘要

Spark 上的 vLLM 是一台**本地 OpenAI 兼容 endpoint**：内存、batch、KV、Prometheus，用来伺候大块 NVFP4。Nemotron-3-Super 的菜谱走 [官方 OpenAI-compatible server 镜像](https://docs.vllm.ai/en/latest/deployment/docker/)，再加 Spark 专用 flag。

架构决定配置：`sm_121` 消费级 Blackwell、CPU+GPU 统一池、Spark 自己的带宽。continuous batching、paged KV、NVFP4 kernel、`/metrics` 才是这台盒子上该拧的旋钮。

`--gpu-memory-utilization` 切的是**统一池**的一份。`--max-num-seqs` 要低：Spark 适合小 batch，不是高并发。当时的构建默认应开 **CUDA graphs**，除非部署有理由关。为吞吐去拧更新的 FP4 kernel、async scheduling、MTP speculative decoding——那是**模型和发版**的事，不是万能 Spark 默认。

## DGX Spark 架构与内存模型

GB10 Grace Blackwell SoC。三件事喂给后文所有旋钮。

**统一内存把能在桌上养活的模型变大。** 比「GPU 显存是一块孤岛」能多塞一些推理。原文写：视架构和运行时，单台 Spark 上加载更大的 NVFP4、**到 200B 参数**是可行的。vLLM 这边的把手：`--gpu-memory-utilization`、`--max-model-len`、`--max-num-seqs`、paged KV。多 Spark：靠 ConnectX「低延迟、高带宽」做分布式——页上**没有** Gbps 数字。

**`sm_121` 要单独验。** 镜像、tag、flag 得是为 Spark 验过的。从更大 GPU 的配置改过来，那是一份**工程清单**（kernel 支不支持、内存怎么走），不是性能预期。

**NVFP4 MoE 才是强拟合。** NVFP4 减内存压、帮 Prefill / 能否装下；Decode 仍看 **active** 参数量和当时构建的 kernel 路径。约 10–15B active 的 NVFP4 MoE 是甜点。Dense、高并发跟这台机器的带宽和统一池不对齐。

**Figure 2**（本地 `03-…svg`）：CPU、GPU、OS、权重、KV 共用 128 GB。

## 跟 Spark 有关的 vLLM 能力

焦点：一台 Spark 上的本地小 batch；多机是 Spark 用网线连起来以后的事。相关的是 paged KV、动态调度、OpenAI 兼容 serving、指标、`sm_121` 镜像。

### 统一内存预算上的 paged KV

老式 lockstep batch 会等最长的那条请求。continuous batching 每一步 Decode 都能进、能出。配上 paged KV，Spark 才能在不太碎片的前提下养活若干 in-flight。

他们在 Spark 上伺候 **120B NVFP4 MoE** 时：单用户 KV 占用通常 **低于 5%**，小 batch demo **低于 30%**。

### 本地 endpoint 上的 OpenAI 兼容流式

客户端代码不用换，指到 `http://localhost:8000/v1`。机房卡也许 Decode 更快；`stream=true` 仍让桌上盒子像在对话。聊天、写代码、agent——感知延迟跟总生成时间一样要紧。

**Figure 1**（本地 `02-…svg`）：客户端打本地官方镜像的 `/v1` 和 `/metrics`。

### Prometheus 上看这台 Spark

一台 Spark 上的可观测性很朴素：Prefill 快不快，Decode 稳不稳，统一池还有没有余量。不必再挂一个服务。Demo 时同一台机器轮询 `/metrics` 就行。

他们点名的信号：`vllm:kv_cache_usage_perc`、prompt / generation token 计数、**TTFT** 和 inter-token-latency 直方图。健康的 agent 回合：第一轮 Prefill 花时间；后面几轮 KV 在长，但前缀若已缓存，Prefill **不该**再尖。吞吐和 ITL 贴近预期 Decode。在 KV 靠近 context 上限之前，应用把对话压短。

### 官方镜像

他们那次 Nemotron-3-Super：CUDA 13 nightly [`vllm/vllm-openai:cu130-nightly`](https://hub.docker.com/r/vllm/vllm-openai/tags?name=cu130-nightly)，带 Spark 的 parser、FP4、调度、内存设置。nightly 会动——把它当**轨道**。部署请钉 release / commit nightly / digest。

Spark 不需要另做一套 serving 接口。Spark 特有的工作在**菜谱、镜像、flag**，对准 GB10 `sm_121`。

## 运行时配置与环境变量

### 先查的菜谱和文档

从 [vLLM Recipes](https://recipes.vllm.ai/) 起，再对生成的 [`vllm serve` CLI](https://docs.vllm.ai/en/latest/cli/serve/) 和 [Docker 文档](https://docs.vllm.ai/en/latest/deployment/docker/)。[OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) 和 [production metrics](https://docs.vllm.ai/en/latest/usage/metrics/) 放在手边。NVIDIA Spark 指南仍是 Spark 专用菜谱、parser 插件、kernel 设置的权威。

### 选模型

Spark 上最大的杠杆，**先于**拧 flag。**Figure 3**（本地 `04-…svg`）是**方向性**的模型拟合，不是性能表：100–130B MoE NVFP4、约 10–15B active，才是本地交互的强拟合。Nemotron-3-Super-120B-A12B-NVFP4 是具体例子。别的 Spark 体量 NVFP4 MoE：原则相同，从**那一个**模型的菜谱起。

### 预先放下权重

不要让第一次 `vllm serve` 顺便下一整份模型。先落到宿主机 Hugging Face cache，再把同一份 cache 挂进长跑容器。「下一遍，处处挂。」

### `vllm serve` 上真正要紧的 flag

例子：`vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`，再加：

**`--gpu-memory-utilization`。** GPU 可见内存的比例。在 Spark 上那就是统一池：OS、kernel page cache、容器、KV 增长、别的进程。从菜谱起，看余量和并发再拧。

**`--max-model-len 131072`。** prompt + completion 上限。131K 是因为系统提示、tool schema、文件、历史很容易超过 **20K** token。可以往模型支持的上限抬，也可以为演示压低。它**不是**每条 in-flight 都按最坏情况预留 KV——vLLM 按**正在用的** context 调度。

**`--max-num-seqs 4`。** 同时 in-flight 的序列数。当时 Nemotron NVFP4 on Spark 的菜谱把这个压低。超过四路并发 decode，单 token 带宽税会压过 continuous batching 的好处，TTFT 尖。

**Automatic prefix caching。** [Prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/) 在 **vLLM V1 默认开**；例子里不传 `--enable-prefix-caching`。长共享系统提示有用。应用在命中为零时也必须正确。

**Tool / reasoning parser。** 跟**模型菜谱**，不是硬件默认。只有模型会吐支持的 reasoning 块才设 reasoning parser；客户端真要 tool 才加 `--enable-auto-tool-choice` 和 tool-call parser。当时构建：Nemotron-3 可用内置 `--reasoning-parser nemotron_v3`。更老的 Spark 菜谱可能还写外部插件 `super_v3`。

值得评估、不要直接抄进 runbook：

- [`--kv-cache-dtype fp8`](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) 能减 KV，但可能伤可预期性，在 Spark 上对某些负载还有**可察觉的性能税**。除非内存真不够、并且质量过关，否则别开。
- [`--speculative-config`](https://docs.vllm.ai/en/latest/features/speculative_decoding/)：这个模型走 MTP。要复测。
- `--tensor-parallel-size 2` 只在 **两台 Spark** 经 ConnectX-7 连起来时有意义。不是单机调优 flag。

### 何时覆盖 vLLM 默认

单 GPU Spark：先菜谱 + 默认。显式覆盖只在你为「这个模型、这个镜像、这块硬件」验过之后。

**Backend。** 量化 linear 和 MoE backend 保持 `auto`，除非验过的菜谱钉死某一个。正确的 FP4 路径随发版和架构变；较新的 **FlashInfer CUTLASS** 比旧 Spark 指南强得多。要钉，优先 `--linear-backend`、`--moe-backend`。这条路上更老的环境变量已经 **deprecated**。

**版本 workaround。** 某些 Spark 菜谱里的兼容环境变量是某个 tag 的事，不是 vLLM 的一般要求。例如：单 Spark、没用 tensor parallelism 的命令，不需要 FlashInfer allreduce backend 覆盖。

**Checkpoint 量化。** vLLM 从模型 config 读量化。预量化的 NVFP4 checkpoint 让 `--quantization` **空着**。只有你打算在加载时再量化，才去设。

### 把 JIT 预先暖上

冷启动看模型、kernel、镜像、请求路径。他们的 Nemotron-3-Super：`vllm serve` 起来后**第一条**请求会触发 Inductor 和 FlashInfer JIT，大约 **25 s**。别把这条路送给真人。启动时用**同一条**客户端路径 ping 一下（同样的 `chat_template_kwargs`，`max_tokens=3`）。暖过之后，那条短 prompt 在他们的 setup 里 **不到 0.5 s**。

权重加载是另一件事。若 **10–15 分钟** 的 safetensor 加载要紧，按你的模型、镜像、存储去评 [fastsafetensors](https://docs.vllm.ai/en/latest/models/extensions/fastsafetensor/) 或 [InstantTensor](https://docs.vllm.ai/en/latest/models/extensions/instanttensor/)。

### 可预期 vs 吞吐

这篇测量：`--kv-cache-dtype` 不设，speculative decoding **关**，CUDA graphs **开**。是这份模型 / 镜像 / 负载的菜谱选择，不是 Spark 宇宙默认。冲吞吐仍可试 FP8 KV、async scheduling、投机、显式 backend——要复测。

他们为公开 demo 优化：可预期的本地 serving、清楚的遥测、稳定的回答。

**Figure 4**（本地 `05-…svg`）：从老实 demo 滑到拧过的吞吐（FP4 backend、async scheduling、投机）。

## 例子负载：vllm-spark-game

[vllm-spark-game](https://github.com/zlxi02/vllm-spark-game)：对着本地 endpoint 玩现场 20 Questions；旁边的 stats 视图在同一台 Spark 上轮询 vLLM 和 GPU。把 OpenAI 兼容聊天、流式、Prefill、Decode、KV、活指标从头走到尾。命令在 [项目 README](https://github.com/zlxi02/vllm-spark-game/blob/master/README.md)。

展位照片（未收录）：*vllm-spark-game demo at the Inferact booth during MLSys, May 2026.*

### Docker 启动

```bash
docker run -d --name vllm --ipc=host --restart unless-stopped \
  --gpus all -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-nightly \
  nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name nemotron-3-super \
    --trust-remote-code \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 4 \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

`cu130-nightly` 是验过的轨道；runbook 请钉 digest。默认 safetensor 首次加载 **10–15 分钟**。就绪：`curl -sS http://localhost:8000/v1/models | jq -r '.data[0].id'` 应返回 `nemotron-3-super`。

### 部署形状

**Figure 5**（本地 `07-…svg`）：游戏打 `/v1`；`spark-stats` 在同一 endpoint 上轮询 `/metrics` 和 NVML。

### 单 Spark 评测

五个面向应用的场景，一台 Spark 上的 Nemotron-3-Super-120B-A12B-NVFP4。讲方法，不是交榜。Decode 停在 **22.7–23.7 tok/s**。每行是 **warmup 一次之后三跑的中位**。token 数来自 `stream_options.include_usage`，不是 chunk 个数。

| Scenario | Prompt tok | Gen tok | TTFT | Total latency | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| typical judge call (real 20Q, noisy 2-token gen) | 58 | 2 | 0.42 s | ~0.53 s | 140 | ~23 |
| medium prompt, short gen | 1,834 | 32 | 1.12 s | ~2.47 s | 1,636 | 23.7 |
| long prompt, short gen | 7,234 | 32 | 3.85 s | ~5.26 s | 1,877 | 22.7 |
| medium prompt, long gen | 1,834 | 108 | 1.12 s | ~5.74 s | 1,639 | 23.4 |
| long prompt, long gen | 7,234 | 124 | 3.84 s | ~9.26 s | 1,884 | 22.9 |

*页上标 Table 2。抓取的 markdown 里没有 Table 1。*

**Figure 6**（本地 `08-…svg`）：同一组扫描——TTFT、总延迟、Prefill、Decode 在 22.7–23.7 那条带子里。

### 怎么读这些数

**Prefill 随 prompt 近线性。** prompt 大约变四倍，TTFT 大约变三倍。Prefill 从 **140** 爬到将近 **1,900 tok/s**，长 prompt 把每次请求的固定开销摊薄。Prefill 是计算密集、整段可并行。

**Decode 窄带 22.7–23.7 tok/s。** judge 那条更在乎人能感到的延迟，它只生成两个 token。Decode 仍取决于 active 参数、FP4 路径、CUDA graphs、具体镜像。这是 Nemotron-3-Super 在**一台** Spark 上的菜谱结果——不是 Spark 或 vLLM 的天花板。

复现时把镜像 tag、context 长度、CUDA graph 开关、backend、调度写在旁边。

**现场 20 Questions。** 典型一轮约 **1,000-token** prompt（系统 + facts + 秘密 + 问题）。感知延迟被 TTFT 和短 Decode 爆发主导。输出 **5–15** token 时，Decode 大约 **0.2–0.7 s**，仍在那条 tok/s 带子里。玩的时候 KV 占用**很少超过 2%**。遥测：每轮开始 `prompt_tps` 跳一下，随后 `gen_tps` 在带子里，答案往外流。

## 运维要点

先选模型类：100–130B NVFP4 MoE 对上容量和 active 画像；dense 通常不对齐本地交互 Decode。官方镜像 + Spark 验过的菜谱，好过自己源码编，除非你要自定义 kernel。按共享池拧 `--gpu-memory-utilization`。JIT 先暖。`/metrics` 给出 KV 占用和 TTFT 直方图。

## 收束

Spark 是开发、demo、小 batch serving 的本地推理盒子。画像跟机房 GPU 服务器不同。统一内存、`sm_121`、模型专用 FP4、本地 Decode——负载怎么拧特别要紧。模型、镜像、flag 验过之后，应用仍然拿到 OpenAI 兼容 API、流式、continuous batching、paged KV、Prometheus。

*原文：[Inferact](https://inferact.ai) 写在办公室里一直开着的那台 Spark 上。*
