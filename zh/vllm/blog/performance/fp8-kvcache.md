---
source: https://vllm.ai/blog/2026-04-22-fp8-kvcache
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# FP8 KV cache：长上下文时把记忆砍半

英文对照：[en/vllm/blog/performance/fp8-kvcache.md](../../../../en/vllm/blog/performance/fp8-kvcache.md)  
原文：https://vllm.ai/blog/2026-04-22-fp8-kvcache  
2026-04-22。`--kv-cache-dtype fp8` 把 KV **和** QK / ScoreV 的注意力乘都打成 FP8（文中全程 **e4m3**）。长上下文（128k+）时 KV 往往占满显存，Decode 每步还要把它读一遍；砍半，并发或窗口才能在同一张卡上再长一截——前提是精度还站得住。

```bash
# 所有层都走 FP8 KV
vllm serve meta-llama/Llama-3.1-8B --kv-cache-dtype fp8

# 混合注意力（有 sliding window）建议跳过那些层：
vllm serve gpt-oss-20b --kv-cache-dtype fp8 --kv-cache-dtype-skip-layers sliding_window
```

文中精度数字用的是 **未校准、per-tensor、scale=1.0**——最差下限，任何人加 `--kv-cache-dtype fp8` 就能复现。校准和 per-head scale 只会更好。他们在 decoder-only 与 MoE、Hopper（FA3，vLLM 的 flash-attention fork）与 Blackwell（FlashInfer）上都压过一遍。`head_dim` 64 / 128 时 Prefill 和 Decode 都可以加速；memory-bound Decode 最好可以把每 token 的 KV 代价压到 BF16 的 **54%**。`head_dim=256` 的 Decode 仍能降 ITL，默认 Prefill 却仍略慢于 BF16。

本地图（原文版权仍归原站；学习对照用）：

![fig1 niah before after plot](../../../../assets/vllm/blog/performance/fp8-kvcache/01-fig1_niah_before_after_plot.png)

Figure 1：Hopper 上 128k needle-in-a-haystack。两级累加把精度从悬崖边拉回 BF16 附近；优化过的 FP8 路径仍保住 Decode 速度。

## 他们修掉的两座坑

`--kv-cache-dtype fp8` 在 vLLM 里已经存在一段时间。压测仍翻出两类问题。

**精度。** Hopper 上 FA3 的 FP8 路径，长上下文会把累加精度吃掉：128k NIAH 从 BF16 的 **91%** 掉到 **13%**。官方说 Tensor Core 往 FP32 寄存器里累，收缩维一大，中间结果其实不准——DeepSeek-V3 训练也撞过（技术报告 Figure 7(b)）。推理里那一维就是上下文长度：`Softmax(AttnScore) * V`。

**速度。** gpt-oss-20b 这类 hybrid，sliding window 层的 KV 有上限，量化摊不回税。全层 FP8 的 ITL 斜率几乎等于 BF16（**96%**），盈亏点 **741,565** token——比大多数人会开的窗口还远。

## Kernel 与 vLLM 侧的修补

**两级累加。** 把部分和写进真正的 FP32 寄存器（[SageAttention2](https://arxiv.org/abs/2411.10958)；[flash-attention#104](https://github.com/vllm-project/flash-attention/pull/104)），NIAH 回到 **89%**。代价是寄存器更挤，Prefill 变慢。[flash-attention#125](https://github.com/vllm-project/flash-attention/pull/125) 用新 tiling 给 `head_dim` 64 / 128 补回一部分；**大于 128，Prefill 仍落后 BF16**。

**跳过某些层。** `--kv-cache-dtype-skip-layers`（[vllm#33695](https://github.com/vllm-project/vllm/pull/33695)）允许混合数值格式。GPT-OSS 的 sliding window 例如只看 128 个 token，FP8 的固定开销摊不掉，留 BF16 反而更快。对量化过敏的层也可以用同一面旗跳过。

**Per-head scale。** FA3 本来就能收一组 scale，每个 KV head 一个。接到 vLLM 需要推广静态量化的 group-shape（[vllm#30833](https://github.com/vllm-project/vllm/pull/30833)），以及让 `reshape_and_cache_flash` 吃数组而不是一个标量（[vllm#30141](https://github.com/vllm-project/vllm/pull/30141)）。

**Query 量化外移。** 从 attention backend 搬到一段普通 torch，让 `torch.compile` 融进周围的 op（[vllm#24914](https://github.com/vllm-project/vllm/pull/24914)），去掉每 token 那点固定税。

**FA3 FP8 的 tile。** Prefill 针对 `head_dim=64` / `128` 减两级累加带来的 register spill（#125）；Decode 另有一套 tile，专门压 ITL 斜率（[flash-attention#96](https://github.com/vllm-project/flash-attention/pull/96)、[#91](https://github.com/vllm-project/flash-attention/pull/91)）。

Hopper / Blackwell 上 FP8 FLOPs 是 BF16 的两倍，Prefill 按理也该快。实践里这笔账不是开箱即有的。

## 单请求：看 ITL 斜率

并发 1，好把 attention 行为看干净。ITL 随 input length 线性涨，因为每个新 token 都要扫完整份 KV：

`ITL = slope × input_len + intercept`

Prefill 是二次的：

`TTFT = a × input_len² + b × input_len + c`

斜率就是「每多缓存一个 token，Decode 贵多少」。**盈亏点**：FP8 ITL 低于 BF16 的上下文长度。

设置：单卡 H100，FA3（vLLM fork，原生 FP8 KV + online softmax rescaling）。`vllm bench serve`，concurrency 1，128 个输出 token，输入从 256 扫到 125k。

![fig2 llama 8b](../../../../assets/vllm/blog/performance/fp8-kvcache/02-fig2_llama_8b.png)

Figure 2：Llama-3.1-8B。斜率几乎腰斩，截距几乎不动，盈亏点大约 7k；TTFT 差不多，长上下文甚至略好。

| | BF16 | FP8 |
|---|---|---|
| ITL slope (ms/token) | `4.37e-05` | `2.37e-05` |
| intercept (ms) | 6.44 | 6.58 |
| 斜率相对 BF16 | 100% | **54%** |
| 截距差 | — | +0.14 ms |
| 盈亏点 | — | **约 7k** |

![fig3 gptoss 20b](../../../../assets/vllm/blog/performance/fp8-kvcache/03-fig3_gptoss_20b.png)

Figure 3：gpt-oss-20b。全局层 + sliding window（窗口 128）。skip-SW 是赢家：那些层的 KV 有上限，量化只交税、不省长上下文的房子。

| | BF16 | FP8（全层） | FP8 skip-SW |
|---|---|---|---|
| ITL slope (ms/token) | `8.94e-06` | `7.14e-06` | `6.34e-06` |
| intercept (ms) | 约 4.03–4.07 | 同一簇 | 同一簇 |
| 斜率相对 BF16 | 100% | 80% | **71%** |
| 盈亏点 | — | 22,109 | **7,659** |

他们建议 hybrid 模型用 skip-SW。

### 改进前后的 ITL 斜率表

原文 Table 3。before = v0.10.2，after = v0.19.1。

| 模型 | 版本 | 变体 | 盈亏点 (tokens) | 斜率 vs BF16 |
|---|---|---|---|---|
| Llama-3.1-8B | v0.10.2 | FP8 | 24,889 | 63% |
| Llama-3.1-8B | v0.19.1 | FP8 | **7,010** | **54%** |
| gpt-oss-20b | v0.10.2 | FP8 | 741,565 | 96% |
| gpt-oss-20b | v0.19.1 | FP8 | 22,109 | 80% |
| gpt-oss-20b | v0.19.1 | FP8 skip-SW | **7,659** | **71%** |

H100 上 Decode 重、上下文长：Llama 一类已经值得开；hybrid 则先跳过那些很小的 sliding window 层。

## 有负载时的吞吐

上面把每 token 的 attention 税隔离出来了。更像线上： **150 条请求、concurrency 8**，约 20k in / 2k out（±15%）。

Table 4：Llama-3.1-8B。FP8 输出吞吐 **+14.9%**、总时长 **−13.0%**、中位 ITL **−14.8%**。

| Config | Median TTFT (ms) | Median ITL (ms) | Total duration (s) | Output tok/s |
|---|---|---|---|---|
| BF16 | 763.6 | 15.18 | 672.6 | 450.3 |
| FP8 | 742.8 | 12.93 | 585.2 | **517.5** |

Table 5：gpt-oss-20b。skip-SW 输出吞吐 **+4.8%**、时长 **−4.6%**、中位 ITL **−4.8%**。

| Config | Median TTFT (ms) | Median ITL (ms) | Total duration (s) | Output tok/s |
|---|---|---|---|---|
| BF16 | 468.9 | 8.09 | 364.2 | 831.6 |
| FP8 | 451.7 | 7.90 | 355.1 | 853.0 |
| FP8 skip-SW | 456.4 | 7.70 | 347.4 | **871.8** |

Llama 在 c=1 时斜率砍到 54%，到 c=8 变成 +14.9% 输出吞吐——token 更快，KV 房子又小一半，调度器能多塞人。gpt-oss 的 sliding window 限制了省房子的幅度，skip-SW 把税交得最少。更高并发或更长上下文，BF16 会先 OOM 或更凶地驱逐；那时 FP8 的房子优势才真正显形。

## `head_dim=256`：Prefill 会退步

flash-attention#104 之后两级累加是**默认开**的，免得有人默默吃下 91%→13% 的悬崖。大 `head_dim` 上，这笔默认税会让 TTFT 慢于 BF16。

![fig4 gemma](../../../../assets/vllm/blog/performance/fp8-kvcache/04-fig4_gemma.png)

Figure 4：gemma-4-E2B，H100，`head_dim=256`。四层里三层 sliding window **512**（gpt-oss 的 128 的四倍，窗口里的数据够摊量化税，SW 层**值得**量化）。Decode 赢，Prefill 输。

| | BF16 | FP8 |
|---|---|---|
| ITL slope (ms/token) | `5.30e-05` | `3.60e-05`（**68%**） |
| TTFT 二次项 (ms/token²) | `6.93e-07` | `1.12e-06`（**约 1.6×**） |

两条出路：

1. **关掉两级累加**——Prefill 回来，64 / 128 的 Prefill 还会更快。**必须自己拿真实负载验精度。**
2. 每 N 步累加一次：开放中的 [flash-attention#122](https://github.com/vllm-project/flash-attention/pull/122)，功能已能跑，Prefill 速度能回来。

## B200 + FlashInfer

累加是 Hopper FA3 的病。B200 上硬件层面没了，不必两级累加。当时 B200 **还不能 skip-SW**，所以只比 BF16 vs FP8。

![fig5 llama b200](../../../../assets/vllm/blog/performance/fp8-kvcache/05-fig5_llama_b200.png)

Figure 5：Llama-3.1-8B。斜率仍约 BF16 的 54%，截距几乎不动，盈亏点大约 **4k**。

| | BF16 | FP8 |
|---|---|---|
| ITL slope (ms/token) | `1.80e-05` | `9.72e-06` |
| intercept (ms) | 3.93 | 3.96 |

![fig6 gptoss b200](../../../../assets/vllm/blog/performance/fp8-kvcache/06-fig6_gptoss_b200.png)

Figure 6：gpt-oss-20b。斜率砍得比 H100 更狠，但截距仍要更长的上下文才赚回来，盈亏点大约 **13k**。

| | BF16 | FP8 |
|---|---|---|
| ITL slope (ms/token) | `3.56e-06` | `2.06e-06` |
| intercept (ms) | 3.15 | 3.17 |

## 精度

模型：Llama-3.3-70B-Instruct、Qwen3-30B-A3B-Instruct-2507、Qwen3-30B-A3B-Thinking-2507、Qwen3.5-27B。

- 长上下文（Prefill 重）：`openai/mrcr`，最长到 1M。每个长度桶平均 pass@1，**5** 次重复；跨长度用 **AUC**（[Context Arena](https://contextarena.ai/)）。
- 推理题（Decode 重）：AIME25、GPQA:Diamond、MATH500、LiveCodeBench-v6。pass@1：AIME25 与 LiveCodeBench-v6 **10** 次，GPQA 与 MATH500 **5** 次。
- 采样：各模型作者建议的**非 greedy** 默认，假装真在线上。

**全部数字都是 per-tensor、未校准、scale=1.0。** 校准走 [`vllm-project/LLM-Compressor`](https://github.com/vllm-project/llm-compressor)；更细的 per-head scale 见 [vllm#30141](https://github.com/vllm-project/vllm/pull/30141) 和 vLLM 量化 KV 的 examples。

### 推理题

![fig7 Qwen3 30B A3B Thinking 2507 reasoning combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/07-fig7_Qwen3-30B-A3B-Thinking-2507_reasoning_combined_plot.png)

Figure 7：Qwen3-30B-A3B-Thinking-2507，BF16 模型与 FP8 权重量化两套。短 Prefill、长 Decode（常常上万 token）。FP8 KV + FP8 attention 大约 **1–2 分**，最低回收 **97%**（GPQA:Diamond，BF16 模型）。

![fig8 Qwen3.5 27B reasoning combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/08-fig8_Qwen3.5-27B_reasoning_combined_plot.png)

Figure 8：Qwen3.5-27B。最多 **0.7 分**，最低回收 **99%**（AIME25，BF16 模型）。

### 长上下文 MRCR

![fig9 Llama 3.3 70B Instruct openai mrcr 2 needles combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/09-fig9_Llama-3.3-70B-Instruct_openai_mrcr_2_needles_combined_plot.png)

Figure 9：Llama-3.3-70B-Instruct，8k 到 128k（模型上限）。AUC@128k 回收 **97–98%**（BF16 / FP8 两套权重都是）。

![fig10 Qwen3 30B A3B Instruct 2507 openai mrcr 2 needles combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/10-fig10_Qwen3-30B-A3B-Instruct-2507_openai_mrcr_2_needles_combined_plot.png)

Figure 10：Qwen3-30B-A3B-Instruct-2507 MoE，到 256k。最长的桶比 Llama 裂得更开。AUC 回收大约 **94%**（BF16 模型）/ **98%**（FP8 模型）。桶间抖动有一部分来自基线自己（32k > 8k/16k；128k > 64k）。

![fig11 Qwen3.5 27B openai mrcr 4 needles combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/11-fig11_Qwen3.5-27B_openai_mrcr_4_needles_combined_plot.png)

Figure 11：Qwen3.5-27B 到 **1M**。聚合 AUC@1M **对齐**基线；最长的桶仍有可见抖动。

### B200 FlashInfer 上的精度

不必两级累加。配方与 Hopper 相同。

![fig12 Qwen3 30B A3B Instruct 2507 openai mrcr 2 needles combined B200 plot](../../../../assets/vllm/blog/performance/fp8-kvcache/12-fig12_Qwen3-30B-A3B-Instruct-2507_openai_mrcr_2_needles_combined_B200_plot.png)

Figure 12：MRCR AUC 回收约 **93%**（BF16 模型）/ **96%**（FP8 模型）。能打，但不如 Hopper/FA3 最好的几条那么贴。

![fig13 Qwen3 30B A3B Thinking 2507 reasoning combined B200 plot](../../../../assets/vllm/blog/performance/fp8-kvcache/13-fig13_Qwen3-30B-A3B-Thinking-2507_reasoning_combined_B200_plot.png)

Figure 13：推理题，平均大约差 **一分或更少**。

他们的结论：许多 Decode 重、被 KV 显存按住的长上下文部署，FP8 KV 可以当**默认起点**。例外：`head_dim=256` 且 Prefill / TTFT 要紧；很小的 sliding window 层应留 BF16；未校准就系统性下跌的 backend / 模型——去校准。

### 何时该校准

不是所有模型都吃 scale=1.0。Kimi-K2.5 走 **FlashMLA**（不是 FA3 / FlashInfer），H200。

![fig14 Kimi K2.5 openai mrcr 4 needles H200 plot](../../../../assets/vllm/blog/performance/fp8-kvcache/14-fig14_Kimi-K2.5_openai_mrcr_4_needles_H200_plot.png)

Figure 14：未校准 FP8 KV + FP8 attention。聚合 AUC 跌得不多，误差带还重叠，但每个长度桶都在往下——这是**系统性**，不是某几个桶的噪声。先开未校准；看见这种形状，再拿目标数据校准。非 FA3 / FlashInfer 的路径尤其要看一眼。

## 何时别开

- 上下文短于大约 **7k**：截距那点税可能让 BF16 的 ITL 更快。
- `head_dim=256` 且 TTFT / Prefill 要紧：两级累加把 TTFT 二次项抬到约 **1.6×**。关掉能换速度，必须自己验精度。
- 未校准精度掉到你的 **95%** 以下：Kimi-K2.5 + FlashMLA 是现成例子，用目标数据校准。
- 很多很小的 sliding window 层：不要全层 FP8，用 `--kv-cache-dtype-skip-layers sliding_window`。

和 TensorRT-LLM 手册第 5 章是同一类税：房子变小，质量自己签字。
