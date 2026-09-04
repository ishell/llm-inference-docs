---
source: https://vllm.ai/blog/2025-10-26-sleep-mode
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Sleep Mode：换模型不必把房子拆了重建

英文对照：[en/vllm/blog/architecture/sleep-mode.md](../../../../en/vllm/blog/architecture/sleep-mode.md)  
原文：https://vllm.ai/blog/2025-10-26-sleep-mode  
2025-10-26。数字是 A100 / A4000、**vLLM 0.11.0** 上的演示。原文有交互图，本地只留学习用的那一张。

两套都能装进一张卡、却装不进同一张卡：要么占 **2×** 显存，要么每次切换 **30–100+ 秒** 冷加载。Sleep Mode 是第三条路——进程还活着，模型去冬眠。

本地图（原文版权仍归原站；学习对照用）：

![sleepmode](../../../../assets/vllm/blog/architecture/sleep-mode/01-sleepmode.png)

## 两档

- **Level 1（默认）：** 权重卸到 CPU RAM，丢掉 KV。醒得最快。要够大的内存。
- **Level 2：** 权重也丢掉，CPU 上只留很小的 buffer（RoPE scaling 一类）。几乎不占 RAM，醒的时候要从盘上再把权重搬回来。

两档都声称比完整 reload 快 **18–200×**，和 TP / PP / EP 一起用。

### 为什么「权重加载再快」也不够

即便 memcpy 瞬间完成，冷启动仍要付隐藏税：

| 成本 | 快加载器 | Sleep Mode |
| --- | --- | --- |
| 1. 权重进 VRAM | 已优化 | 保住 |
| 2. CUDA allocator 初始化 | 每次 | 保住 |
| 3. CUDA graph 捕获 | 每次 | 保住 |
| 4. GPU kernel JIT（DeepGEMM、FlashInfer、TorchInductor） | 每次 | 初次 warmup 之后保住 |
| 5. 第一轮 cache warmup | 每次 | 很快再热一下 |

进程还在，**2–4** 就还在。所以醒来后的第一次推理比冷启动快 **61–88%**——不是搬运变快了，是基础设施还在。

| 组件 | Sleep | 冷启动 |
| --- | --- | --- |
| 内存分配器（`CuMemAllocator`） | 在 | 重做 |
| CUDA graphs | 在 | 重捕获 |
| 进程（Python、CUDA context） | 在 | 重启 |
| GPU kernel JIT cache | 初次 warmup 后在 | 重编译 |

不睡：卸载就把进程杀死，**预热没有地方住**。文中例子：第一次推理 **4–7×** 更慢（wake **0.92 s** vs 冷启动 **3.72 s**）。睡着：第一次大约 **1 s**，躲开 **3–4 s** 的冷惩罚。时间随模型、卡、配置差很多——warmup 消融里，不预热会 **5–7×** 慢。

## 怎么用（当时）

管理接口要 `VLLM_SERVER_DEV_MODE=1`。`/sleep` `/wake_up` `/collective_rpc` `/reset_prefix_cache` 只该出现在受信网络（训练集群、后端），公开网上不要开。

```bash
export VLLM_SERVER_DEV_MODE=1
vllm serve microsoft/Phi-3-vision-128k-instruct --enable-sleep-mode --port 8001

export VLLM_SERVER_DEV_MODE=1
vllm serve Qwen/Qwen3-0.6B --enable-sleep-mode --port 8002
```

Level 2 的一轮（Phi-3-vision）：

```bash
curl -X POST 'localhost:8001/sleep?level=2'
curl -X POST 'localhost:8002/sleep?level=2'

curl -X POST 'localhost:8001/wake_up'
curl -X POST 'localhost:8001/collective_rpc' \
  -H 'Content-Type: application/json' \
  -d '{"method":"reload_weights"}'
curl -X POST 'localhost:8001/reset_prefix_cache'
# 推理 …
curl -X POST 'localhost:8001/sleep?level=2'

curl -X POST 'localhost:8002/wake_up'
# Level 1 不必 reload_weights / reset_prefix_cache
```

**Level 2 醒了必须** `reload_weights` 和 `reset_prefix_cache`。**Level 1 不必。**

## 数字（vLLM 0.11.0，`cudagraph_mode: FULL_AND_PIECEWISE`）

默认工作形状：**5 次切换**——A 推理 → 切 B → B 推理 → 再重复（A→B→A→B→A→B）。推理时间 = 醒来 / 加载后**第一条**请求的 Prefill + Decode；题目每次不同以免缓存；输出上限 **100** token。原文误差棒是多次的 min/max。

### A100，大模型，Level 1

**A：** Qwen3-235B-A22B-Instruct-2507-FP8（**TP=4**）。**B：** Qwen3-Coder-30B-A3B-Instruct（**TP=1**）。正文写：醒来比全新加载大约 **18–20×**。醒来后的推理相对冷启动，仍落在上面的 **61–88%**。原文五次切换的总时间是交互图，抓取里没有逐格数字。

### A4000，小模型，Level 1

**A：** Qwen3-0.6B。**B：** Phi-3-vision-128k-instruct。A4000，**TP=1**。

- 推理：Qwen3-0.6B 快 **83%**，Phi-3-vision 快 **81%**
- 切换：醒来 **0.1–0.8 s**，相对冷启动 **58–203×**
- 五次切换总时间：**85 s vs 226 s**（大约 **62%**）
- 小模型可以接近瞬间（约 **0.1 s**）

### 三档对比（A100，同一对小模型，TP=1）

档位里的醒来区间：L1 小模型约 **0.1–0.8 s**、大模型约 **3–6 s**；L2 小模型约 **0.8–2.6 s**（要从盘上再搬权重）。

| 模式 | 五次总时间 | 醒来（A / B） | CPU RAM | 适合 |
| --- | --- | --- | --- | --- |
| **No Sleep** | **357.1 s** | 整份 reload | 最少 | 单模型 |
| **Level 1** | **112.6 s** | **0.26 s / 0.82 s** | 高（每模大约 GB） | 切得勤、内存够 |
| **Level 2** | **124.6 s** | **0.85 s / 2.58 s** | 很少（每模大约 MB） | 内存紧、模型多 |

L1 比不睡快 **68%**，L2 快 **65%**。L2 醒来大约比 L1 慢 **3×**（Qwen3-0.6B：0.85 s vs 0.26 s）。

L2 也要从 SSD 搬权重，为什么切换还能 **23–45×**？

| 成本 | Level 2 | 不睡 |
| --- | --- | --- |
| 1. 权重 SSD → VRAM | 付 | 付 |
| 2. 进程初始化 | 跳过 | 付 |
| 3. Allocator | 跳过 | 付 |
| 4. CUDA graph | 跳过 | 付 |
| 5. Kernel JIT | 已经编过，留下 | 全编 + `kernel_warmup()` + dummy |

五次：L2 **124.6 s**（大约每次 **2.6 s**）对不睡 **357.1 s**（大约每次 **48 s**）——整体大约 **2.9×**，尽管两边都碰盘。

Level 2 细表：

| 指标 | 不睡 | Level 2 | 改善 |
| --- | --- | --- | --- |
| 五次总时间 | 357.1 s | 124.6 s | **65%** |
| Qwen3-0.6B 切换 | 平均 37.6 s | 平均 0.85 s | **45×** |
| Phi-3-vision 切换 | 平均 58.1 s | 平均 2.58 s | **23×** |
| Qwen3-0.6B 推理 | 平均 3.67 s | 平均 0.53 s | **86%** |
| Phi-3-vision 推理 | 平均 6.30 s | 平均 0.76 s | **88%** |
| 醒来相对 Level 1 | — | 慢 3–10× | 用 RAM 换速度 |

一句话对照：L1 醒来约 **0.1–0.8 s**，CPU RAM 大约 **10–100 GB+** / 模型；L2 约 **0.8–2.6 s**，只要大约 **MB**；两边都远快过完整 reload（约 **20–100 s**）。

## 消融（A100，TP=1，Level 1，同一对小模型）

### Warm-up

预热在初次加载时把 CUDA graph 编好。

| 指标 | 有 warmup | 没有 | 差 |
| --- | --- | --- | --- |
| 初次加载 | **108.7 s**（含 **8.4 s** warmup） | **101.1 s** | 开头省 **7.6 s** |
| 第一次推理（A） | **0.45 s** | **2.59 s** | 没有则 **5.8×** |
| 第一次推理（B） | **0.93 s** | **6.61 s** | 没有则 **7.1×** |
| 之后的推理 | 平均 0.43 s | 平均 0.41 s | 没有差 |
| 五次总时间 | **119.5 s** | **119.0 s** | 几乎一样 |

JIT 和 graph **只在加载时付一次**，睡醒都还在。不预热，那 **5–7×** 发生在**每一次**醒来后的第一条，不是一辈子一次。一条 **1-token** 的推理就够触发完整 JIT + graph。生产上若在意第一条的稳定，文中建议**永远做 warmup**。总时间看起来差不多，是因为 8.4 s 被摊掉了；延迟的**形状**并不一样。

### FP8 vs BF16

| 指标 | BF16 | FP8 | 变化 |
| --- | --- | --- | --- |
| 五次总时间 | **108.2 s** | **113.6 s** | **−5%**（略慢） |
| Qwen3-0.6B 醒来 | 平均 0.27 s | 平均 0.18 s | **33%** 更快 |
| Phi-3-vision 醒来 | 平均 0.90 s | 平均 0.78 s | **13%** 更快 |
| Qwen3-0.6B 推理 | 平均 0.41 s | 平均 0.44 s | **−7%** |
| Phi-3-vision 推理 | 平均 0.81 s | 平均 0.57 s | **30%** 更快 |
| 初次加载 | **90.5 s** | **96.9 s** | **−7%**（warmup 更长） |

FP8 醒来搬得少；两个模型里较大的那个推理更受益；初次加载更长（warmup 里要量化）。切得勤时，更快的醒来仍可能把更长的初次加载赚回来。

## 怎么选（原文）

**Level 1：** CPU RAM 装得下所有权重；要最快醒来（**0.1–6 s**）；几秒 / 几分钟就切一次；推理延迟必须稳。

**Level 2：** RAM 装不下全部权重；要更便宜的云主机；模型 **10+**。

**不必 Sleep：** 只有一个模型；一天 / 一周才切一次；两套已经能同时住进 VRAM。

收束：切换 **18–200×**；醒来后的推理相对冷启动 **61–88%**；A100 小模型那套总时间 **65–68%**；规模从 **0.6B 到 235B**，卡从 A4000 到 A100。致谢名单见英文对照。

[torch.compile](torch-compile.md) 把启动变贵；Sleep 承认这件事，选择**不要把进程杀死**。多 LoRA 挤在同一套权重上是另一条路；Sleep 是「整模替换、进程留下」。
