---
source: https://vllm.ai/blog/2026-04-22-fp8-kvcache
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# FP8 KV cache：长上下文时把记忆砍半

英文对照：`en/vllm/blog/performance/fp8-kvcache.md`  
原文：https://vllm.ai/blog/2026-04-22-fp8-kvcache  
2026-04-22。`--kv-cache-dtype fp8` 把 KV **和** QK / ScoreV 的注意力乘都打成 FP8（文中全程 **e4m3**）。长上下文（128k+）时 KV 往往占满显存，decode 每步还要把它读一遍；砍半，并发或窗口才能在同一张卡上再长一截——前提是精度还站得住。

```bash
vllm serve meta-llama/Llama-3.1-8B --kv-cache-dtype fp8
# 混合注意力（有 sliding window）建议跳过那些层：
vllm serve gpt-oss-20b --kv-cache-dtype fp8 --kv-cache-dtype-skip-layers sliding_window
```

文中精度数字用的是 **未校准、per-tensor、scale=1.0**——最差下限。校准和 per-head scale 只会更好。


本地图（原文版权仍归原站；学习对照用）：

![fig1 niah before after plot](../../../../assets/vllm/blog/performance/fp8-kvcache/01-fig1_niah_before_after_plot.png)

![fig2 llama 8b](../../../../assets/vllm/blog/performance/fp8-kvcache/02-fig2_llama_8b.png)

![fig3 gptoss 20b](../../../../assets/vllm/blog/performance/fp8-kvcache/03-fig3_gptoss_20b.png)

![fig4 gemma](../../../../assets/vllm/blog/performance/fp8-kvcache/04-fig4_gemma.png)

![fig5 llama b200](../../../../assets/vllm/blog/performance/fp8-kvcache/05-fig5_llama_b200.png)

![fig6 gptoss b200](../../../../assets/vllm/blog/performance/fp8-kvcache/06-fig6_gptoss_b200.png)

![fig7 Qwen3 30B A3B Thinking 2507 reasoning combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/07-fig7_Qwen3-30B-A3B-Thinking-2507_reasoning_combined_plot.png)

![fig8 Qwen3.5 27B reasoning combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/08-fig8_Qwen3.5-27B_reasoning_combined_plot.png)

![fig9 Llama 3.3 70B Instruct openai mrcr 2 needles combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/09-fig9_Llama-3.3-70B-Instruct_openai_mrcr_2_needles_combined_plot.png)

![fig10 Qwen3 30B A3B Instruct 2507 openai mrcr 2 needles combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/10-fig10_Qwen3-30B-A3B-Instruct-2507_openai_mrcr_2_needles_combined_plot.png)

![fig11 Qwen3.5 27B openai mrcr 4 needles combined plot](../../../../assets/vllm/blog/performance/fp8-kvcache/11-fig11_Qwen3.5-27B_openai_mrcr_4_needles_combined_plot.png)

![fig12 Qwen3 30B A3B Instruct 2507 openai mrcr 2 needles combined B200 plot](../../../../assets/vllm/blog/performance/fp8-kvcache/12-fig12_Qwen3-30B-A3B-Instruct-2507_openai_mrcr_2_needles_combined_B200_plot.png)

![fig13 Qwen3 30B A3B Thinking 2507 reasoning combined B200 plot](../../../../assets/vllm/blog/performance/fp8-kvcache/13-fig13_Qwen3-30B-A3B-Thinking-2507_reasoning_combined_B200_plot.png)

![fig14 Kimi K2.5 openai mrcr 4 needles H200 plot](../../../../assets/vllm/blog/performance/fp8-kvcache/14-fig14_Kimi-K2.5_openai_mrcr_4_needles_H200_plot.png)

## 他们修掉的两座坑

Hopper 上 FA3 的 FP8 路径，长上下文会把精度吃掉：128k needle-in-a-haystack 从 BF16 的 **91%** 掉到 **13%**。原因是 Tensor Core 中间累加在 contraction 维很大时丢精度（DeepSeek-V3 训练也撞过）。对策：**两级累加**写回真正的 FP32（SageAttention2 / flash-attention#104），精度回到 **89%**。代价是寄存器更挤，prefill 变慢；`head_dim` 64/128 用新 tiling 补回一部分，**256 的 prefill 仍落后 BF16**。

另一座：gpt-oss-20b 这类 hybrid，sliding window 层的 KV 有上限，量化摊不回税。全层 FP8 的 ITL 斜率几乎等于 BF16（96%），盈亏点到 **70 万+ token**。`--kv-cache-dtype-skip-layers sliding_window` 让那些层留 BF16。

另外还有：per-head scale、query 量化搬出 attention 让 `torch.compile` 融合、decode 专用 tile。

## 单请求：看 ITL 斜率

并发 1，H100，FA3。`ITL = slope × input_len + intercept`。斜率就是「每多缓存一个 token，decode 贵多少」。盈亏点：FP8 ITL 低于 BF16 的上下文长度。

Llama-3.1-8B：斜率 BF16 `4.37e-05` → FP8 `2.37e-05`（**54%**），截距几乎不动，盈亏点大约 **7k**。TTFT 在长上下文上甚至略好。

gpt-oss-20b：跳过 sliding window 最好（斜率 71% of BF16，盈亏点约 **7.7k**）；全层 FP8 是 80% / 22k。

| 模型 | 版本 | 变体 | 盈亏点 | 斜率 vs BF16 |
|---|---|---|---|---|
| Llama-3.1-8B | v0.10.2 | FP8 | 24,889 | 63% |
| Llama-3.1-8B | v0.19.1 | FP8 | 7,010 | 54% |
| gpt-oss-20b | v0.10.2 | FP8 | 741,565 | 96% |
| gpt-oss-20b | v0.19.1 | FP8 | 22,109 | 80% |
| gpt-oss-20b | v0.19.1 | skip-SW | 7,659 | 71% |

并发 8、约 20k in / 2k out：Llama 输出吞吐 **+14.9%**、中位 ITL **−14.8%**；gpt-oss skip-SW 吞吐 **+4.8%**。更高并发或更长上下文，BF16 会先 OOM，FP8 的房子优势才真正显形。

`head_dim=256`（gemma-4-E2B）：decode 仍赢（斜率 68%），TTFT 二次项大约 **1.6×**——两级累加把 prefill 拖慢。可关两级累加（必须自己验精度），或等「每 N 步累加一次」的 PR。窗口 512 比 128 大，sliding window 层量化反而划算。

B200 + FlashInfer：累加问题硬件上没了。Llama 斜率仍约 54%，盈亏点约 **4k**；gpt-oss 约 13k（当时 B200 还不能 skip-SW）。

## 精度

推理题（AIME / GPQA / MATH500 / LiveCodeBench）：大约 **1–2 分**，最低回收约 97%。长上下文 MRCR：Llama-3.3-70B AUC 约 **97–98%**；Qwen3 MoE 到 256k 约 **94–98%**；Qwen3.5-27B 到 **1M** 聚合 AUC 对齐。B200 FlashInfer 略松一点（约 93–96% AUC）。

**该校准的时候：** Kimi-K2.5 + FlashMLA，未校准会出现**系统性**下移——不是某几个桶的噪声。非 FA3/FlashInfer 的 backend 更要拿真实负载看一眼。

## 何时别开

- 上下文短于大约 **7k**：截距那点税可能让 BF16 更快。
- `head_dim=256` 且 TTFT 要紧。
- 未校准精度掉到你的 95% 以下。
- 很多很小的 sliding window 层：跳过它们。

结论：decode 重、被 KV 显存按住的负载，FP8 KV 可以当**默认起点**。和 TensorRT-LLM 手册第 5 章是同一类税：房子变小，质量自己签字。
