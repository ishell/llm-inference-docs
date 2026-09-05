---
source: https://vllm.ai/blog/2025-10-26-sleep-mode
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# Sleep Mode：换模型不必把房子拆了重建

英文对照：[en/vllm/blog/architecture/sleep-mode.md](../../../../en/vllm/blog/architecture/sleep-mode.md)  
原文：https://vllm.ai/blog/2025-10-26-sleep-mode

2025-10-26。Embedded LLM。学习译文，不是官方译本。数字是 **vLLM 0.11.0**、`cudagraph_mode: FULL_AND_PIECEWISE`，A100 与 A4000。原文大量 Plotly 交互图，笔记不收脚本，把能从原文表格和图表数据里抽出的秒数写成表。

两套都能单独装进一张卡、却装不进同一张卡：要么占 **2×** 显存，要么每次切换 **30–100+ 秒** 冷加载。Sleep Mode 是第三条路——进程还活着，模型去冬眠。几秒睡下，醒来也快：既有按需加载的省，也有常驻 serving 的速度。

本地图（原文版权仍归原站；学习对照用）：

![sleepmode](../../../../assets/vllm/blog/architecture/sleep-mode/01-sleepmode.png)

**图注（原文）。** vLLM Sleep Mode。

## Introduction

**多模型 serving 的困境：** 两只 LLM 各自能进 GPU，一起不行。传统两条路都难看：

1. **两套都常驻** → 要 2× GPU 显存（贵，常常根本装不下）
2. **用到再加载** → 每次切换 30–100+ 秒（慢，浪费）

Sleep Mode 让模型在几秒内冬眠、醒来也快。

### Two Sleep Levels for Different Needs

- **Level 1：** 权重卸到 CPU RAM（醒来最快）
- **Level 2：** 权重整份丢掉（醒来几乎一样快，RAM 占用极小）

两档都声称比完整 reload 快 **18–200×**，和 Tensor Parallelism（TP）、Pipeline Parallelism（PP）、Expert Parallelism（EP）一起用。

### Why Sleep Mode Beats Fast Weight Loaders

即便权重加载瞬间完成，每次冷启动仍要付 Sleep Mode 躲开的隐藏税：

| Cost | Description | Fast Weight Loaders | Sleep Mode |
|------|-------------|---------------------|------------|
| 1. VRAM load time | 把权重拷进 GPU | 已优化 | 保住 |
| 2. Memory allocator setup | CUDA allocator 初始化 | 每次都付 | 保住 |
| 3. CUDA graph capture | 录执行图 | 每次都付 | 保住 |
| 4. GPU kernel JIT compilation | DeepGEMM、FlashInfer、TorchInductor | 每次都付 | 初次 warmup 之后保住 |
| 5. Cache warm-up | 第一轮请求的开销 | 每次都付 | 很快再热一下 |

进程还活着，基础设施（#2–4）就还在，不必昂贵地重初始化。所以 benchmark 里：**Sleep Mode 推理比冷启动快 61–88%**。

这篇覆盖：

- 模型从 0.6B 到 235B、GPU 从 A4000 到 A100 的对照
- 为什么会快的技术拆解
- warmup 与 FP8 量化消融
- 选哪一档 Sleep 的决策表

## Quick Start: Using Sleep Mode

### Online Serving API

两台开了 Sleep Mode 的 vLLM：

```bash
# Terminal 1: Start Phi-3-vision
export VLLM_SERVER_DEV_MODE=1
vllm serve microsoft/Phi-3-vision-128k-instruct --enable-sleep-mode --port 8001

# Terminal 2: Start Qwen3-0.6B
export VLLM_SERVER_DEV_MODE=1
vllm serve Qwen/Qwen3-0.6B --enable-sleep-mode --port 8002
```

### Sleep and Wake Models

```bash
# Put Phi-3-vision to sleep (Level 2 - minimal RAM usage)
curl -X POST 'localhost:8001/sleep?level=2'

# Put Qwen3-0.6B to sleep (Level 2)
curl -X POST 'localhost:8002/sleep?level=2'

# Wake up Phi-3-vision for inference
curl -X POST 'localhost:8001/wake_up'
curl -X POST 'localhost:8001/collective_rpc' \
  -H 'Content-Type: application/json' \
  -d '{"method":"reload_weights"}'

# IMPORTANT: Reset prefix cache after waking (Level 2 only)
curl -X POST 'localhost:8001/reset_prefix_cache'

# Now run inference on Phi-3-vision...
# (your inference requests here)

# Put back to sleep when done
curl -X POST 'localhost:8001/sleep?level=2'

# Wake up Qwen3-0.6B
curl -X POST 'localhost:8002/wake_up'
# (Level 1 doesn't need reload_weights or reset_prefix_cache)

# Run inference on Qwen3-0.6B...
```

> **NOTE。** Level 2 醒了必须再调 `reload_weights` 和 `reset_prefix_cache`。Level 1 不必。

> **WARNING。Security：** `/sleep`、`/wake_up`、`/collective_rpc`、`/reset_prefix_cache` 需要 `VLLM_SERVER_DEV_MODE=1`，只该出现在受信网络。这些管理接口能把服务打断，给训练集群或后端用，不要暴露到公网。

## Performance Overview

先看 Sleep Mode 相对传统整份 reload 差多少。

### Sleep Mode L1 vs No Sleep Mode Performance

原文交互图量的是 **5 次模型切换的总时间**：A 推理 → 切 B → B 推理 → 再重复（A→B→A→B→A→B）。

**有 Sleep Mode：** 切换之间 sleep / wake，基础设施留下。  
**没有 Sleep Mode：** 每次切换都是完整 vLLM 重启再加载。

**Model A：** Qwen3-235B-A22B-Instruct-2507-FP8（TP=4）。**Model B：** Qwen3-Coder-30B-A3B-Instruct（TP=1）。GPU：A100。vLLM 0.11.0。Sleep Level 1。Compilation：`cudagraph_mode: FULL_AND_PIECEWISE`。

原文是甘特式 Plotly。从图表数据抽出的事件秒数（含初次加载）：

| 阶段 | WITH Sleep L1 (s) | WITHOUT Sleep (s) |
|------|-------------------|-------------------|
| A Load（+ warmup 2.38） | 97.61 + 2.38 | 97.9 / 97.4 / 97.71（三次加载） |
| B Load（+ warmup 2.42） | 47.63 + 2.42 | 47.33 / 47.47 / 47.46 |
| A Wake | 5.66, 5.29, 5.27 | —（每次整份 Load） |
| B Wake | 2.89, 2.86, 2.85 | — |
| A Prompt | 1.8, 1.7, 0.92 | 3.8, 3.7, 3.72 |
| B Prompt | 1.0, 0.93, 0.54 | 3.7, 2.9, 2.45 |
| A Sleep | 6.01, 5.78, 5.89 | — |
| B Sleep | 2.78, 2.78 | — |
| 事件合计（约） | **~205 s** | **~456 s** |

## Inference Performance Boost

切换变快之外，醒来后的**第一次推理**也更快：模型已经热过，冷启动那笔开销不必再付。

原文口径：**推理时间 = Prefill + Decode（wake / load 后第一条）**。每条请求换题目以免缓存；输出上限 **100** token。误差棒是多次的 min / max。同一套 A100 / vLLM 0.11.0 / Level 1 / `FULL_AND_PIECEWISE`。

从图表数据抽出（三次）：

| 模型 | Wake (s) | Cold (s) | Wake 均值 | Cold 均值 |
|------|----------|----------|-----------|-----------|
| Qwen3-235B-A22B（TP=4） | 1.8, 1.7, 0.92 | 3.8, 3.7, 3.72 | ~1.47 | ~3.74 |
| Qwen3-Coder-30B（TP=1） | 1.0, 0.93, 0.54 | 3.7, 2.9, 2.45 | ~0.82 | ~3.02 |

正文写：醒来后的推理相对冷启动快 **61–88%**。文中点名的对照：wake **0.92 s** vs 冷启动 **3.72 s**（第一次推理大约 **4–7×** 慢）。

#### Why Sleep Mode Improves Inference Speed

这 61–88% **不是**权重 memcpy 变快，是冷启动必须重建的基础设施还在。

**Sleep Mode 保住什么：**

| Component | Preserved? | Cold Start Must Pay |
|-----------|-----------|---------------------|
| Memory allocator（`CuMemAllocator`） | 是 | 每次重初始化 |
| CUDA graphs | 是 | 每次重捕获 |
| Process state（Python、CUDA context） | 是 | 每次重启 |
| GPU kernel JIT cache | 是（初次 warmup 之后） | 每次重编译 |

**关键差别：**

- **不睡：** 卸载就把进程杀死 → **预热没有地方住**。必须重启 Python 与 CUDA context、重做 allocator、重捕获 CUDA graphs、重 JIT（DeepGEMM、FlashInfer、TorchInductor）。**结果：** 第一次推理 **4–7×** 更慢（见上：0.92 s wake vs 3.72 s 冷启动）。
- **睡着：** 进程留下 → **预热能摊销**。Allocator、graphs、进程状态、JIT kernel 在初次 warmup 之后都还在。**结果：** 第一次推理仍大约 1 s，躲开 3–4 s 的冷惩罚。

> **NOTE。** 时间随模型大小、GPU 代数、配置差很多。见 [Impact of Warm-Up](#impact-of-warm-up-on-sleep-mode)：不预热会 **5–7×** 慢。

## Model Switching Performance

最戏剧的一笔是切换时间。叫醒一只睡着的模型，比拉起一套全新 vLLM 快 **18–20×**。

同一套 A100 / Level 1 / `FULL_AND_PIECEWISE`。图表数据（三次）：

| 模型 | Wake (s) | Cold load (s) | Wake 均值 | Cold 均值 | 大约倍数 |
|------|----------|---------------|-----------|-----------|----------|
| Qwen3-235B-A22B（TP=4） | 5.66, 5.29, 5.27 | 97.9, 97.4, 97.71 | ~5.41 | ~97.7 | **~18×** |
| Qwen3-Coder-30B（TP=1） | 2.89, 2.86, 2.85 | 47.33, 47.47, 47.46 | ~2.87 | ~47.4 | **~17×** |

## Hardware Scalability: A4000 GPU Results

不是高端卡才有这笔账。同一形状的活放到 **A4000**、更小的模型上：收益跨硬件档和模型尺寸还在。

**Model A：** Qwen3-0.6B。**Model B：** Phi-3-vision-128k-instruct。GPU：A4000（TP=1）。vLLM 0.11.0。Sleep Level 1。`cudagraph_mode: FULL_AND_PIECEWISE`。

图表数据（含初次加载）：

| 阶段 | WITH Sleep L1 (s) | WITHOUT Sleep (s) |
|------|-------------------|-------------------|
| A Load（+ warmup 2.49） | 21.01 + 2.49 | 21.04 / 20.98 / 20.98 |
| B Load（+ warmup 7.37） | 46.01 + 7.37 | 46.01 / 46.02 / 46.02 |
| A Wake | 0.11, 0.10, 0.10 | — |
| B Wake | 0.80, 0.80, 0.80 | — |
| A Prompt | 0.44, 0.43, 0.43 | 2.64, 2.50, 2.63 |
| B Prompt | 2.04, 1.73, 1.61 | 9.78, 9.01, 9.79 |
| 正文合计（5 次切换） | **85 s** | **226 s**（大约 **62%**） |

### A4000: Inference Performance

推理时间 = Prefill + Decode（wake / load 后第一条）；题目每次不同；输出上限 100 token。

| 模型 | Wake (s) | Cold (s) | 正文改善 |
|------|----------|----------|----------|
| Qwen3-0.6B | 0.44, 0.43, 0.43 | 2.64, 2.50, 2.63 | **83%** 更快 |
| Phi-3-vision-128k（4B） | 2.04, 1.73, 1.61 | 9.78, 9.01, 9.79 | **81%** 更快 |

### A4000: Model Switching Performance

| 模型 | Wake (s) | Cold (s) | 大约倍数 |
|------|----------|----------|----------|
| Qwen3-0.6B | 0.11, 0.10, 0.10 | 21.04, 20.98, 20.98 | **~200×** |
| Phi-3-vision-128k（4B） | 0.80, 0.80, 0.80 | 46.01, 46.02, 46.02 | **~58×** |

**Key Observations on A4000：**

- **推理：** Qwen3-0.6B 快 83%，Phi-3-vision 快 81%
- **切换：** 醒来约 **0.1–0.8 s**，相对冷启动 **58–203×**
- **总时间节省 62%**（85 s vs 226 s，5 次切换）
- 小模型接近瞬间（约 0.1 s），多模型 serving 摸起来像连续的
- Sleep Mode 跨 GPU 档和模型尺寸都成立

## Sleep Levels: Choosing the Right Mode

两档，代价不同：

**Level 1（默认）：** 权重卸到 CPU，丢掉 KV cache

- **醒来最快**（小模型约 0.1–0.8 s，大模型约 3–6 s）
- **要够大的 CPU RAM** 装权重
- **适合：** 内存够、切得勤

**Level 2：** 权重和 KV 都丢掉，CPU 上只留很小的 buffer（RoPE scaling tensors 一类）

- **醒来更慢**（小模型约 0.8–2.6 s），因为要从盘上再搬权重
- **几乎不占 RAM**——只留小 buffer
- **适合：** CPU RAM 紧，或模型多到内存装不下全部权重

### Performance Comparison: Level 1 vs Level 2 vs No Sleep

**Model A：** Qwen3-0.6B。**Model B：** Phi-3-vision-128k-instruct。GPU：A100（TP=1）。vLLM 0.11.0。`FULL_AND_PIECEWISE`。

**Performance Summary：**

| Mode | Total Time | Wake Time (A/B) | CPU RAM | Best For |
|------|------------|-----------------|---------|----------|
| **No Sleep** | 357.1 s | N/A（整份 reload） | 最少 | 单模型、不切 |
| **Level 1** | 112.6 s | 0.26 s / 0.82 s | 高（每模大约 GB） | 切得勤、RAM 够 |
| **Level 2** | 124.6 s | 0.85 s / 2.58 s | 很少（每模大约 MB） | RAM 紧、省成本 |

图表数据里 Level 1 的三次 wake：A 0.25 / 0.28 / 0.25 s，B 0.82 / 0.82 / 0.83 s；Level 2：A 0.91 / 0.78 / 0.85 s，B 2.55 / 2.62 / 2.58 s。

**Key Insights：**

- **Level 1 最快**（比不睡快 68%），但要可观的 CPU RAM
- **Level 2 几乎一样快**（比不睡快 65%），RAM 几乎可以忽略
- **Level 2 醒来大约比 Level 1 慢 3×**（Qwen3-0.6B：0.85 s vs 0.26 s），因为要再搬权重
- 两档相对不睡都是数量级改善

#### Why Level 2 is Still Faster Than No Sleep Mode

乍看不合理：**Level 2 也要从 SSD 搬权重**（和不睡一样），整体却快 **23–45×**？

**答案：搬权重只是五笔成本里的一笔。**

不睡时你要付全部：

| Cost | Level 2 | No Sleep Mode |
|------|---------|---------------|
| 1. Weight load（SSD → VRAM） | 付 | 付 |
| 2. Process initialization | **跳过** | 付 |
| 3. Memory allocator setup | **跳过** | 付 |
| 4. CUDA graph capture | **跳过** | 付 |
| 5. GPU kernel JIT compilation | **保住（已经编过）** | 全编 + warmup |

**Level 2 策略：** 权重仍从 SSD 搬（和不睡一样）；**其余都留下**：进程、allocator 实例、CUDA graphs、编好的 JIT kernel。初次 warmup 编过的 kernel 还在缓存里。**平均每次切换约 2.6 s。**

**不睡的现实：** 同样碰盘；**其余全重建**：进程重启 + allocator + graph 重捕获；JIT 要完整编译 + 显式 warmup（`kernel_warmup()` + dummy runs）。**平均每次切换约 48 s。**

5 次切换的合计：

- **Level 2：** 124.6 s（平均每次约 2.6 s）
- **No Sleep：** 357.1 s（平均每次约 48 s）

两边都从 SSD 搬权重，Level 2 整体仍 **2.9×**，因为它保住了不睡每次都要重建的那几样贵东西。

### Level 2: Inference Performance

A100、TP=1、Sleep Level 2、`FULL_AND_PIECEWISE`。推理 = Prefill + Decode（第一条）；题目每次不同；输出上限 100 token。

图表数据：

| 模型 | Wake (s) | Cold (s) |
|------|----------|----------|
| Qwen3-0.6B | 0.68, 0.46, 0.44 | 4.66, 3.80, 2.56 |
| Phi-3-vision-128k | 0.78, 0.77, 0.72 | 6.55, 6.21, 6.15 |

### Level 2: Model Switching Performance

图表数据：

| 模型 | Wake (s) | Cold (s) |
|------|----------|----------|
| Qwen3-0.6B | 0.91, 0.78, 0.85 | 38.53, 37.21, 38.15 |
| Phi-3-vision-128k | 2.55, 2.62, 2.58 | 58.52, 57.65, 58.20 |

**Key Observations：**

| Metric | No Sleep | Level 2 | Improvement |
|--------|----------|---------|-------------|
| **Total Time（5 switches）** | 357.1 s | 124.6 s | **65%** 更快 |
| **Qwen3-0.6B Switch Time** | 平均 37.6 s | 平均 0.85 s | **45×** |
| **Phi-3-vision Switch Time** | 平均 58.1 s | 平均 2.58 s | **23×** |
| **Qwen3-0.6B Inference** | 平均 3.67 s | 平均 0.53 s | **86%** 更快 |
| **Phi-3-vision Inference** | 平均 6.30 s | 平均 0.76 s | **88%** 更快 |
| **Wake Time vs Level 1** | — | 慢 3–10× | 用 CPU RAM 换速度 |

**When to Use Level 2：**

- **CPU RAM 紧：** 装不下全部权重
- **省云成本：** 更便宜、内存更小的实例
- **模型很多：** 切来切去，内存是约束
- **仍然很值：** 即便要再搬权重，仍比不睡快 23–45×

**Level 1 vs Level 2：**

- Level 1：醒来约 0.1–0.8 s，每模大约要 10–100GB+ CPU RAM
- Level 2：醒来约 0.8–2.6 s，每模只要大约 MB
- 两边都远快过完整 reload（约 20–100 s）

## Ablation Studies

### Impact of Warm-Up on Sleep Mode

跳过 warmup 会怎样？Warmup 在初次加载时预编译 CUDA graphs，可能要好几秒。有 / 没有对照。

**Model A：** Qwen3-0.6B。**Model B：** Phi-3-vision-128k-instruct。A100、TP=1、Level 1、`FULL_AND_PIECEWISE`。

**Key Findings：**

| Metric | With Warm-Up | Without Warm-Up | Difference |
|--------|--------------|-----------------|------------|
| **Initial Load Time** | 108.7 s（含 8.4 s warmup） | 101.1 s（无 warmup） | 开头省 7.6 s |
| **First Inference (A)** | 0.45 s | 2.59 s | 没有则 **5.8×** |
| **First Inference (B)** | 0.93 s | 6.61 s | 没有则 **7.1×** |
| **Subsequent Inferences** | 平均 0.43 s | 平均 0.41 s | 没有差 |
| **Total Time（5 switches）** | 119.5 s | 119.0 s | 几乎一样 |

图表数据：有 warmup 时 A Load 37.65 + Warm Up 2.39，B Load 62.69 + Warm Up 6.0；第一次 A Prompt 0.45、B Prompt 0.93。没有 warmup：A Load 37.91、B Load 63.16；第一次 A Prompt 2.59、B Prompt 6.61；之后的 Prompt 回到 0.41 / 0.70 一带。

**Insights：**

- **Warmup 编一次，睡醒都受益：** 初次加载时 JIT 和 CUDA graph 付一次，后续 sleep / wake 都留下
- **不预热，每次醒来的第一条都要付编译：** 那 5–7× 发生在**每一次** wake 后的第一条，不是一辈子一次
- **编好的 kernel 跨 sleep / wake 还在：** 初次 8.4 s warmup 之后，后续第一次推理是 0.45 s、0.93 s
- **很少的 warmup 就够：** 一条 **1-token** 的推理就能触发完整 JIT + graph capture
- **用初次加载换稳定延迟：** 8.4 s 付一次，摊到所有切换上
- **建议：生产里永远做 warmup**，如果你在意第一条推理稳、快

总时间看起来差不多，是因为 8.4 s 被摊掉了；延迟的**形状**并不一样。

### Impact of Quantization on Sleep Mode

FP8 会不会改 Sleep Mode 的表现？同一套活，A100 上 BF16 vs FP8。

同一对小模型、TP=1、Level 1、`FULL_AND_PIECEWISE`。

### Ablation: Inference Performance (BF16 vs FP8)

图表数据（三次 Prompt）：

| 模型 | BF16 (s) | FP8 (s) |
|------|----------|---------|
| Qwen3-0.6B | 0.41, 0.40, 0.41 | 0.43, 0.43, 0.45 |
| Phi-3-vision-128k | 0.90, 0.74, 0.80 | 0.69, 0.59, 0.44 |

### Ablation: Model Switching (BF16 vs FP8)

图表数据（三次 Wake）：

| 模型 | BF16 (s) | FP8 (s) |
|------|----------|---------|
| Qwen3-0.6B | 0.28, 0.27, 0.27 | 0.18, 0.19, 0.16 |
| Phi-3-vision-128k | 0.89, 0.93, 0.88 | 0.79, 0.77, 0.78 |

**Key Findings：**

| Metric | BF16 | FP8 | Improvement |
|--------|------|-----|-------------|
| **Total Time（5 switches）** | 108.2 s | 113.6 s | −5%（略慢） |
| **Qwen3-0.6B Wake Time** | 平均 0.27 s | 平均 0.18 s | **33%** 更快 |
| **Phi-3-vision Wake Time** | 平均 0.90 s | 平均 0.78 s | **13%** 更快 |
| **Qwen3-0.6B Inference** | 平均 0.41 s | 平均 0.44 s | −7%（略慢） |
| **Phi-3-vision Inference** | 平均 0.81 s | 平均 0.57 s | **30%** 更快 |
| **Initial Load Time** | 90.5 s | 96.9 s | −7%（warmup 更长） |

**Insights：**

- **FP8 醒来更快**（13–33%），因为搬的内存更少
- **较大的模型推理更受益**（Phi-3-vision 快 30%）；极小模型几乎看不出
- **初次加载更长**，warmup 里有量化开销
- 加载完成之后，FP8 的切换更顺、醒来更便宜
- 切得勤时，更快的 wake 仍可能把更长的初次加载赚回来

## Decision Guide: Which Sleep Level to Use?

### Use Sleep Level 1 When:

- CPU RAM 装得下全部权重
- 要最快醒来（0.1–6 s）
- 几秒 / 几分钟就切一次
- 推理延迟必须稳

### Use Sleep Level 2 When:

- CPU RAM 装不下全部权重
- 要更便宜的云主机（内存更小）
- 要管很多模型（10+）

### Skip Sleep Mode When:

- 只有一个模型（不必切）
- 一天 / 一周才切一次
- 两套已经能同时住进 GPU 显存

## Conclusion

Sleep Mode 把多模型 GPU serving 从 30–100 秒的 reload 惩罚，收成亚秒级切换。Benchmark 自己会说话：

- 切换快 **18–200×**（看模型和卡）
- 热过的模型相对冷启动，推理快 **61–88%**
- 完整工作负载总时间省 **65–68%**
- **每一档规模都成立：** 0.6B 到 235B，小卡和大卡

LLM serving 的未来是多模型。Sleep Mode 让这件事今天就能做。

## Acknowledgements

Vensen Mu, Jeff Aw, Jun Kang Chow, Tun Jian Tan, Pin Siang Tan, Amir Balwel, Ye Hur Cheong, Zhiyao Cen, Kaichao You —— Sleep Mode 功能和这篇博客。

[torch.compile](torch-compile.md) 把启动变贵；Sleep 承认这件事，选择**不要把进程杀死**。多 LoRA 挤在同一套权重上是另一条路；Sleep 是「整模替换、进程留下」。
