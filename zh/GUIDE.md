---
source: catalog of NVIDIA + vLLM official URLs
lang: zh
voice: book-zh
fetched: 2026-09-06
---

# 怎么读这份笔记

这里收集已经公开的 LLM **推理**资料，写成对照笔记，方便以后查阅。覆盖两块：NVIDIA 一侧（NIM、TensorRT-LLM、AIPerf / GenAI-Perf、Triton），以及 vLLM 一侧（文档页、`vllm serve` 性能相关旗标、带日期的博客）。网页版：[https://ishell.github.io/llm-inference-docs/](https://ishell.github.io/llm-inference-docs/)。完整对照表在 [总目录](../README.md)。中文体例见 [STYLE.md](STYLE.md)。

**不是官方译本，也不是教材。** 原页版权仍归原站。篇幅和版权都不允许把 HTML 原样搬进来。中文按《Crafting Interpreters》中译本那类书面语来写：完整句子、用「我们」讲解，但不改公式、不改 CLI、不改指标名。NVIDIA 一侧多为压缩对照；vLLM 带日期的博客按原文分节写完（数字、API、caveat 都在）。文件头的 `source:` 指向原文；`en/` 是英文摘录或抓取，和 `zh/` 成对。

读之前先记住三件事：**load testing（压测）**、**performance benchmarking（性能测试）**、**performance improvement（调优）**。这份库围着这三件事转，不讲训练，也不讲「大模型是什么」。

## 这份库打算做什么

- 把散落在 NVIDIA Developer Blog、docs.nvidia.com、vllm.ai 上的推理文档收到一个能搜、能对照的地方。
- 中英对照：同一篇笔记两份文件，路径对齐。中文页自己读；英文页核对原文口径。
- 指标先对齐，再谈数字。TTFT、ITL、TPS、RPS 在不同工具里除法不同，横比之前先确认定义是否相同。
- 最短路径几篇配了另画的学习图（说明用中文，Prefill / Decode 等词留英文）。其余附图仍是原站图，只作对照；logo、头像、封面库存图不收。

刻意不做的：`vllm serve` 每一个旗标、AIPerf 整页 CLI、官方网页全文。NVIDIA 笔记仍是压缩对照，往往短于英文原页。vLLM 带日期的博客按原文分节写完；短公告本身就短，笔记也不注水。

## 目录怎么走

```
assets/          正文图；中文学习图在对应笔记的 zh/ 子目录
en/ · zh/
  nvidia/
    benchmarking/        压测 vs 性能测试、NIM 手册、系列博客 1–2
    performance-tuning/  Mastering、系列第 3 篇、TensorRT-LLM 调优全套
    cost/                系列第 4 篇 TCO
    tools/               AIPerf、GenAI-Perf、Perf Analyzer、Triton
  vllm/
    getting-started/     入口、quickstart、serve 旗标摘录
    optimization/        调优顺序
    benchmarking/        bench CLI、auto-tune
    metrics/             /metrics
    features/            prefix cache、spec decode、V1
    blog/                CATALOG、MUST-READ、FLAG-MAP、逐篇笔记
```

侧栏按主题排。顶栏「怎么读」就是本页；「必读」是 vLLM 博客主线，不是全库目录。

## 先把尺子对齐

| 词 | 在问什么 | 常用工具 |
|---|---|---|
| load testing | 模拟真实流量，看容量、伸缩、网络、资源 | **K6 / Locust** |
| performance benchmarking | 在受控条件下测 TTFT、ITL/TPOT、TPS、RPS | **AIPerf**（旧名 GenAI-Perf，已停更） |
| performance improvement | 测过之后再调 batch、KV cache、量化、并行 | TensorRT-LLM Tuning Guide；vLLM 的 `optimization` |

两套都要做。只压系统，我们看不出模型本身够不够快；只测模型，真实高峰到来时系统可能先撑不住。只做其中一套，得到的数字很难在生产里站住。

几个容易混淆的词，笔记里一律留英文：

- **TTFT**：从提交到第一个非空 token。包含排队 + prefill + 网络。
- **ITL / TPOT**：相邻输出 token 之间的间隔。GenAI-Perf / AIPerf **不含** TTFT；LLMPerf 常常把 TTFT 算进去。
- **e2e_latency** = TTFT + generation_time。
- **TPS**：系统合计与单用户往往方向相反。并发升高时，系统吞吐通常上升，单用户延迟通常变差。
- **Prefill / Decode**：Prefill 一次处理整段 prompt 并建立 KV cache，偏 compute-bound；Decode 每步一个 token，偏 memory-bandwidth-bound。

## 几条常见读法

不必从第一篇排到最后一篇。按目的选路。

**只想先会测。** [基本概念](nvidia/benchmarking/blog-01-fundamental-concepts.md) → [NIM 指标](nvidia/benchmarking/nim-02-metrics.md) → [用 AIPerf 打一轮](nvidia/benchmarking/nim-04-aiperf.md) → 尺子本身 [AIPerf](nvidia/tools/aiperf.md) → [指标公式](nvidia/tools/aiperf-metrics.md) → [调度](nvidia/tools/aiperf-load-generator.md) → [五类打法](nvidia/tools/aiperf-comprehensive.md)。扫描 concurrency，画出 latency–throughput 曲线，再谈 SLA。

**走 NVIDIA 栈（NIM / TensorRT-LLM）。** 上面那条测完之后：[Mastering](nvidia/performance-tuning/mastering-llm-techniques.md) 当地图 → [Tuning Guide](nvidia/performance-tuning/trtllm-tuning-guide.md) 六章（基线 → 编译旗标 → max batch / max tokens → 模型切分 → FP8 → 运行时）→ 邻居页 KV / IFB / `trtllm-bench`。案例里的 Llama-3.3-70B、4×H100、2048/2048 是演示，质量每一步都要自己测。成本是 [系列第 4 篇](nvidia/cost/blog-04-tco.md)。

**走 vLLM。** [optimization](vllm/optimization/optimization.md) 给出调优顺序；[serve 旗标摘录](vllm/getting-started/serve.md) 只留和性能有关的项。机制从哪来，走 [必读博客](vllm/blog/MUST-READ.md)；旗标对应哪篇博客，走 [对照表](vllm/blog/FLAG-MAP.md)。Anatomy 把 serving 从里讲到外，适合在 optimization 之后读，不要当第一篇。

**只想弄懂 PagedAttention / KV。** [立项](vllm/blog/architecture/paged-attention.md) → Anatomy 里 cache 那几节 → Mastering 的 KV / 分页 → TensorRT-LLM 的 [IFB 与 paged KV](nvidia/performance-tuning/trtllm-paged-attention-ifb.md)。同一套机制，NVIDIA 和 vLLM 各讲一遍。

传统 Triton 模型（分类、检测、非生成）不要拿 infer/sec 去和 TTFT 比较：[Perf Analyzer](nvidia/tools/perf-analyzer.md) 打基线，[Triton 调优](nvidia/tools/triton-performance-tuning.md) 搜 `config.pbtxt`。[GenAI-Perf](nvidia/tools/genai-perf.md) 已停更，只为对照旧数字。

## 建议先读的四篇

这四篇构成最短的工程闭环：先对齐指标，再跑一轮测试，再知道 vLLM 调什么、系统长什么样。

1. [基本概念](nvidia/benchmarking/blog-01-fundamental-concepts.md) — TTFT / ITL / TPS，压测 vs 性能测试，concurrency 怎么扫。
2. [用 AIPerf 打一轮](nvidia/benchmarking/nim-04-aiperf.md) — 画出 latency–throughput 曲线。
3. [vLLM 调优顺序](vllm/optimization/optimization.md) — CPU 核、`-O*`、`max_num_batched_tokens`、并行、cache。
4. [Anatomy of vLLM](vllm/blog/architecture/anatomy.md) — 把 serving 系统从里讲到外。

NIM 手册可以插在 1 和 2 之间：[总览](nvidia/benchmarking/nim-01-overview.md) → [指标](nvidia/benchmarking/nim-02-metrics.md) → [参数](nvidia/benchmarking/nim-03-parameters.md) → 再打 AIPerf。LoRA 怎么测见 [nim-05](nvidia/benchmarking/nim-05-lora.md)。

## TensorRT-LLM 调优手册

调优落到 NVIDIA 文档站，就是 [Performance Tuning Guide](nvidia/performance-tuning/trtllm-tuning-guide.md)。六章按官方目录排：先打基线，再调编译旗标、max batch / max tokens、模型切分、FP8、运行时调度。邻居页是 KV cache、IFB 调度、`trtllm-bench`。

**IFB**（in-flight / continuous batching）让 Prefill 和 Decode 出现在同一步里。三个尺寸：`max_batch_size`、`max_num_tokens`、`max_seq_len`。长 prompt 要靠 chunked context，否则 token 预算用尽时请求只能等待。切分策略先看互联：节点内 NVLink 倾向 TP，跨节点带宽较差倾向 PP；一张卡装得下就不必切。

## NVIDIA 工具（tools/）

生成式 LLM 用 **AIPerf**。入口和常见陷阱在 [aiperf.md](nvidia/tools/aiperf.md)；公式在 [aiperf-metrics.md](nvidia/tools/aiperf-metrics.md)；concurrency / QPS / trace / 每用户回合在 [aiperf-load-generator.md](nvidia/tools/aiperf-load-generator.md)；Pareto / jsonl / Mooncake / goodput / 时间切片在 [aiperf-comprehensive.md](nvidia/tools/aiperf-comprehensive.md)。

官方建议 benchmark 用 **concurrency**（始终维持 N 个在途请求），不要一上来就按 QPS 把队列堆满。`ignore_eos=True` 只为了让测试的 OSL 可控；生产环境应尊重 EOS。

## vLLM 博客（必读线）

文档页给出调优顺序。博客说明这些选项是怎么长出来的。完整顺序在 [MUST-READ](vllm/blog/MUST-READ.md)。CATALOG 里 134 篇带日期的都有中英对照学习笔记（按原文分节写完），但 **不必按全表逐篇读**；day-0 和活动文不是这条线。

V1 / spec-decode 文中的「还不支持」是当时的缺口，不是今天的功能表。EPD 那篇是把 **视觉编码器**拆出去；文本 Prefill/Decode 分离在 Router 与大规模两篇。Mooncake 是跨实例 KV 池（agent 前缀）；Elastic EP 是运行时改 DP 宽度。

**架构：** [立项](vllm/blog/architecture/paged-attention.md) → [Anatomy](vllm/blog/architecture/anatomy.md) → [V1](vllm/blog/architecture/v1-alpha.md) → [MRV2](vllm/blog/architecture/mrv2.md)

**性能：** [v0.6 CPU](vllm/blog/performance/v0.6-throughput.md) → [投机解码](vllm/blog/performance/spec-decode.md) → [FP8 KV](vllm/blog/performance/fp8-kvcache.md) → [生产级 CI](vllm/blog/performance/production-quality.md)

**Serving：** [分布式切分](vllm/blog/serving/distributed-inference.md) → [production-stack](vllm/blog/serving/production-stack.md) / [AIBrix](vllm/blog/serving/aibrix.md) → [Router](vllm/blog/serving/router.md) → [Encoder 分离](vllm/blog/serving/epd.md) → [Wide-EP](vllm/blog/serving/large-scale.md) → [Mooncake](vllm/blog/serving/mooncake.md) → [Elastic EP](vllm/blog/serving/elastic-ep.md)

**第二波（机制）：** [torch.compile](vllm/blog/architecture/torch-compile.md) → [Sleep](vllm/blog/architecture/sleep-mode.md) → [structured decoding](vllm/blog/performance/struct-decode.md) → [DCP](vllm/blog/performance/dcp.md) → [KV offload](vllm/blog/serving/kv-offload.md) → [分层 KV](vllm/blog/serving/tiered-kv-offload.md) → [MORI-IO](vllm/blog/serving/moriio.md) → [Hybrid SSM](vllm/blog/serving/hybrid-ssm.md) → [AFD](vllm/blog/serving/afd.md)

**第三波：** [插件](vllm/blog/architecture/plugin-system.md) → [硬件插件](vllm/blog/architecture/hardware-plugin.md) → [TT 插件](vllm/blog/architecture/tt-plugin.md) → [Triton attention](vllm/blog/architecture/triton-attn.md) → [SHM IPC](vllm/blog/serving/shm-ipc.md) → [PegaFlow](vllm/blog/serving/pegaflow.md) → [TurboQuant](vllm/blog/performance/turboquant.md) → [Native RL](vllm/blog/serving/native-rl.md) → [Ray symmetric-run](vllm/blog/serving/ray-symmetric.md)
