---
source: https://vllm.ai/blog/2026-05-11-turboquant
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# TurboQuant：KV 再压到 3–4 bit 之前，先读完这篇对照

英文对照：[en/vllm/blog/performance/turboquant.md](../../../../en/vllm/blog/performance/turboquant.md)  
原文：https://vllm.ai/blog/2026-05-11-turboquant  
2026-05-11。Eldar Kurtić、Michael Goin、Alexandre Marques（Red Hat AI）。数字来自 **vLLM 0.20.2**（commit `6ec9bbec3`）。接在 [FP8 KV](fp8-kvcache.md) 后面读。论文：[TurboQuant](https://arxiv.org/pdf/2504.19874)。当时文档：[quantization/turboquant](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/)。

[TurboQuant](https://arxiv.org/pdf/2504.19874) 把 KV 压到 3–4 bit，广告上是一大截 GPU 显存。和 [FP8 KV](https://vllm.ai/blog/fp8-kvcache)（`--kv-cache-dtype fp8`）不一样：FP8 连 **存储和 attention 计算** 都走硬件 FP8 Tensor Core；TurboQuant 只压缩**存储**，算的时候再反量化回 BF16。省的是房间，付的是反量化。精度和速度都从这条裂缝里长出来。

以前报的数字，多半是小模型、短上下文，KV 量化几乎不被逼到墙角。这篇量了四个模型（稠密和 MoE，**30B 到 200B+**），五套基准：Prefill 偏重的长上下文检索，Decode 偏重的推理。

```bash
# FP8 KV-cache for all layers
vllm serve MiniMaxAI/MiniMax-M2.7 --kv-cache-dtype fp8

# TurboQuant KV-cache, skipping the first and last two layers
vllm serve MiniMaxAI/MiniMax-M2.7 --kv-cache-dtype turboquant_4bit_nc
```

本地图（原文版权仍归原站；学习对照用）：

![llama 70b pareto](../../../../assets/vllm/blog/performance/turboquant/01-llama_70b_pareto.png)

Figure 1：Llama-3.3-70B-Instruct 在 4×H100 上的 Pareto。FP8 压过全场：burst 吞吐比 BF16 高 **2.6×**，KV 容量 **2×**。TurboQuant 各变体都是用吞吐换再多一点房间。

![qwen3 30b a3b pareto](../../../../assets/vllm/blog/performance/turboquant/02-qwen3_30b_a3b_pareto.png)

Figure 2：Qwen3-30B-A3B-Instruct-2507 在 2×H100 上。FP8 吞吐打平 BF16，容量 **2×**。TurboQuant 把容量拉到 **2.3–3.7×**，吞吐却掉 **40–52%**。

## TL;DR（当时）

- **FP8**（`--kv-cache-dtype fp8`）仍是默认：KV 容量大约 **2×**，精度损失可忽略，多数指标打平 BF16，显存紧的 serving 上还明显更好。
- TurboQuant **`k8v4`** 几乎没有胜过 FP8 的理由：容量大约 **2.4× vs 2×**，吞吐和延迟却稳定变差。
- TurboQuant **`4bit-nc`** 是 TQ 里最可能用的：KV 紧的时候多一点房间，换中等的精度、延迟、吞吐。边缘、房间极度不够时可以考虑。
- TurboQuant **`k3v4-nc`** 和 **`3bit-nc`**：推理和超长上下文掉得能看见，延迟/吞吐也一并变差。不当生产默认。

## Experimental Setup

**量化方案。** 四个 TurboQuant dtype，对照未量化 BF16 和 FP8：

| `--kv-cache-dtype` | Keys / values | 说明 |
| --- | --- | --- |
| *（无 / BF16）* | BF16 | 未量化基线 |
| `fp8` | Q、K、V 都是 FP8 | **attention 计算** 也走 Tensor Core |
| `turboquant_k8v4` | K 8 bit、V 4 bit | 名字里没有 norm correction |
| `turboquant_4bit_nc` | K、V 都是 4 bit | 带 norm correction |
| `turboquant_k3v4_nc` | K 3 bit、V 4 bit | 带 norm correction |
| `turboquant_3bit_nc` | K、V 都是 3 bit | 带 norm correction |

TurboQuant 只压存储，attention 前要反量化回 BF16。变体细节见 [论文](https://arxiv.org/pdf/2504.19874) 和 [vLLM TurboQuant 文档](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/)。FP8 背景见 [fp8-kvcache.md](fp8-kvcache.md)。

**基准。** 五套，Prefill 重、Decode 重都有。长上下文检索：`openai/mrcr`（多轮上下文检索），测到每个模型自己的最大长度。推理：AIME25、GPQA:Diamond、MATH500、LiveCodeBench-v6。采样一律用模型作者建议的**默认非 greedy**，用来贴近真实部署。

**模型。** 稠密和 MoE，小的和大的：`Llama-3.3-70B-Instruct`、`Qwen3-30B-A3B-Instruct-2507`、`Qwen3-30B-A3B-Thinking-2507`、`MiniMax-M2.7`。

**当时还不能做什么。** TurboQuant 当时只支持**标准注意力**（GQA 一类）。滑动窗口、混合注意力 **当时还没有**。

## Accuracy Results

### Long-context Retrieval

`openai/mrcr`。每个长度桶平均 pass@1，**5** 次重复；再拿各长度上的 Area-Under-Curve（AUC）当总账（[Context Arena](https://contextarena.ai/)）。

![Llama 3.3 70B Instruct openai mrcr 2 needles plot](../../../../assets/vllm/blog/performance/turboquant/03-Llama-3.3-70B-Instruct_openai_mrcr_2_needles_plot.png)

Figure 3：Llama-3.3-70B-Instruct 长上下文检索，图上到 64k。到 **128k**（模型上限），BF16 基线塌到 **10% 以下**。

Llama-3.3-70B-Instruct 上，高一点 bit 的 TQ（`k8v4`、`4bit-nc`）检索还站得住，AUC 大约 **52%**。`k3v4-nc`（**48.6%**）和 `3bit-nc`（**50.3%**）每个长度都差一截；64k 处裂口最大，可到 **8 分**。

![Qwen3 30B A3B Instruct 2507 openai mrcr 2 needles plot](../../../../assets/vllm/blog/performance/turboquant/04-Qwen3-30B-A3B-Instruct-2507_openai_mrcr_2_needles_plot.png)

Figure 4：Qwen3-30B-A3B-Instruct-2507 长上下文检索，到 256k。

Qwen3-30B-A3B-Instruct-2507 能到 **256k**，差别更刺眼。BF16（**45.8%**）、FP8（**43.1%**）、TQ `k8v4`（**43.0%**）还在彼此标准差里。TQ `4bit-nc`（**42.3%**）也还像样。狠压的变体就不像了：TQ `k3v4-nc` AUC **33.5%**，TQ `3bit-nc` **31.2%**——相对 BF16 大约 **−30%**。伤集中在 **128k–256k**：低 bit 的 KV 误差会跟长度一起攒。

**Takeaway：** 长上下文检索上，`k8v4` 和 `4bit-nc` 还算安全。`k3v4-nc` 和 `3bit-nc` 不安全，尤其是特别长的上下文。FP8 打平高 bit 的 TQ，后面会看到它 serving 更快。

### Reasoning

Decode 偏重：AIME25、GPQA:Diamond、MATH500、LiveCodeBench-v6。平均 pass@1：AIME25 和 LiveCodeBench-v6 重复 **10** 次；GPQA:Diamond 和 MATH500 重复 **5** 次。

![Qwen3 30B A3B Thinking 2507 reasoning plot](../../../../assets/vllm/blog/performance/turboquant/05-Qwen3-30B-A3B-Thinking-2507_reasoning_plot.png)

Figure 5：Qwen3-30B-A3B-Thinking-2507 的推理。狠压的 TQ（`k3v4-nc`、`3bit-nc`）在 AIME25 和 LiveCodeBench-v6 上掉得很深。

Qwen3-30B-A3B-Thinking-2507 上层次清楚。FP8 和 TQ `k8v4` 贴近 BF16，平均精度恢复 **>98%**。TQ `4bit-nc` 稍差，**96%**。TQ `k3v4-nc` 和 `3bit-nc` 平均掉大约 **20 分**。连相对容易的 MATH500 也大约 **4 分**——狠压的 TQ 不适合长生成推理。

![MiniMax M2.7 reasoning plot](../../../../assets/vllm/blog/performance/turboquant/06-MiniMax-M2.7_reasoning_plot.png)

Figure 6：MiniMax-M2.7 的推理。更大的模型通常更扛量化；狠压的 TQ 仍然伤，尤其 AIME25 和 LiveCodeBench-v6。

MiniMax-M2.7（**200B+**）同一套阶梯。FP8 和 TQ `k8v4` 恢复 **>99%**；TQ `4bit-nc` 中等下降；`k3v4-nc` / `3bit-nc` 仍掉，AIME25 和 LiveCodeBench-v6 上可到大约 **8 分**。

**Takeaway：** 狠压的 TQ（`k3v4-nc`、`3bit-nc`）在硬数学和代码上伤得明显。`4bit-nc` 是中等一刀。`k8v4` 打平未量化 BF16。FP8 也打平未量化基线，而且（下面）比任何 TQ 变体都快一截。

## Performance Results

只盯两台：`Qwen3-30B-A3B-Instruct-2507` 在 **2×H100**，`Llama-3.3-70B-Instruct` 在 **4×H100**。延迟、离线吞吐、在线 serving（TPOT 和 TTFT），若干请求速率。vLLM **0.20.2**，commit `6ec9bbec3`。

### Latency

`vllm bench latency`。合成请求写死：输入 **1024**，输出 **256**；batch **1、8、32、64**。warmup **10** 轮，测量 **30** 轮。相对 BF16 的慢多少（越低越好）。

![qwen3 30b a3b latency](../../../../assets/vllm/blog/performance/turboquant/07-qwen3_30b_a3b_latency.png)

Figure 7：Qwen3-30B-A3B-Instruct-2507 相对 BF16 的延迟税。FP8 几乎没有，batch 一大就更看不见。TQ 按变体和 batch，最多大约 **60%**。

![llama 70b latency](../../../../assets/vllm/blog/performance/turboquant/08-llama_70b_latency.png)

Figure 8：Llama-3.3-70B-Instruct。FP8 可忽略；TQ **10–68%**。

FP8 在两个模型、所有 batch 上都几乎不加延迟——attention 本身走 FP8 Tensor Core，没有反量化这一拍。TQ 全都加得出来：Qwen3-30B 大约 **10–60%**；Llama-3.3-70B 大约 **10–68%**。70B 上 TQ 的税还随 batch **变大**——想用来扛并发的人最不想看见的方向。低 bit 存着、算前再反量化回 BF16，摸到的 KV 越多，这一拍越贵。

### Throughput

`vllm bench throughput`。**200** 条 prompt；三组输入/输出：**256/256**、**1024/512**、**4096/256**。相对 BF16 吞吐的百分比（越高越好）。

![qwen3 30b a3b throughput](../../../../assets/vllm/blog/performance/turboquant/09-qwen3_30b_a3b_throughput.png)

Figure 9：Qwen3-30B-A3B-Instruct-2507 平均吞吐相对 BF16。FP8 保住 BF16；TQ 全在下面。KV 存得便宜，并不等于 serving 更快。

![llama 70b throughput](../../../../assets/vllm/blog/performance/turboquant/10-llama_70b_throughput.png)

Figure 10：Llama-3.3-70B-Instruct。同一件事。

FP8 在两台上打平 BF16。TQ 全部严格低于 BF16：Qwen3-30B 从 **80%**（`k8v4`）到 **73%**（`3bit-nc`）；Llama-70B 从 **75%**（`k8v4` 和 `4bit-nc`）到 **66%**（`3bit-nc`）。打包越狠，吞吐越差。反量化的价钱跟着 packing 复杂度涨。

### Serving Speed

`vllm bench serve`。合成输入 **1024**、输出 **512**；测量 **300** 条，warmup **5** 条。请求速率 **2**、**8**、以及 `inf`（能发多快发多快）。看 **TPOT**（Time Per Output Token——Decode 有多快）和 **P99 TTFT**（Time To First Token——请求多久才开始吐字）。

![qwen3 30b a3b serve](../../../../assets/vllm/blog/performance/turboquant/11-qwen3_30b_a3b_serve.png)

Figure 11：Qwen3-30B-A3B-Instruct-2507 的 serving TPOT。

![llama 70b serve](../../../../assets/vllm/blog/performance/turboquant/12-llama_70b_serve.png)

Figure 12：Llama-3.3-70B-Instruct 的 serving TPOT。

TPOT 和延迟、吞吐是同一张脸：FP8 在每个速率上跟上或超过 BF16；TQ 给每个 token 加税，负载越大税越重。Llama-70B burst 时，FP8 几乎比 BF16 快 **2×**；TQ 变体慢 **1.5× 到 2.5×**。

![qwen3 30b a3b ttft](../../../../assets/vllm/blog/performance/turboquant/13-qwen3_30b_a3b_ttft.png)

Figure 13：Qwen3-30B-A3B-Instruct-2507 的 P99 TTFT。

![llama 70b ttft](../../../../assets/vllm/blog/performance/turboquant/14-llama_70b_ttft.png)

Figure 14：Llama-3.3-70B-Instruct 的 P99 TTFT。burst 时 BF16 的 TTFT 冲到大约 **17 s**（KV 饱和、门口排队）；TurboQuant 压在 **3.5 s** 内；FP8 在 **1.5 s** 内（正文写大约 **1.3 s**）。

Qwen3-30B 在 2×H100 上 KV 还比较宽敞，FP8 的 TTFT 每个速率都打平 BF16。TQ 一直更慢，burst 时可到 **2×**。Llama-3.3-70B 在 4×H100 上 KV 房间窄，burst 时 BF16 的 P99 TTFT 冲到大约 **17 s**——KV 满了，新请求只能在门口等。TQ 各变体都压在 **3.5 s** 内，大约 **5×**：压缩后的 KV 让更多在飞的请求不用排队。FP8 仍是最低 TTFT，大约 **1.3 s**，也稳赢所有 TQ。

**Takeaway：** 吞吐和每 token 延迟上，TQ 稳定不如 BF16 和 FP8。可是 serving 一旦被显存卡住，KV 压缩能挡住饱和，burst 时的 TTFT 相对 BF16 会好看一截。这才是 TQ 的卖点：**用每 token 的速度，换「请求不会在门口排队」**。两边都要的话，FP8 已经把这件事做了：吞吐打平或超过 BF16，延迟税可忽略，burst TTFT 还明显更好。

## Key Findings and Recommendations

**默认仍是 FP8（`--kv-cache-dtype fp8`）。** KV 容量 **2×**，不付吞吐，精度损失可忽略，量化后的 attention 有时还更快。绝大多数负载上最稳、最好预期。细节见 [fp8-kvcache.md](fp8-kvcache.md)。

**TurboQuant `k8v4` 几乎没有胜过 FP8 的理由。** 容量只多到 **2.4× vs 2×**，吞吐和延迟却稳定变差。

**TurboQuant `4bit-nc` 是用房间换吞吐的那一档。** KV 容量最高大约 **3.4×**（Qwen 的 Pareto 图上，TQ 各变体一起看能到 **3.7×**），多数基准上精度大约掉 **1–4 分**。显存紧、burst TTFT 的改善盖过其他指标的恶化时，才值得。上线前要在**目标负载**上自己验精度。

**没有彻底验证，不要用 `k3v4-nc` 和 `3bit-nc`。** 硬数学和代码上精度可掉到 **20 分**。反量化步骤也更绕，不当生产默认。

**GPU 显存不是瓶颈时，留 BF16。** 短上下文、低并发、HBM 够用：BF16 仍是精度–性能的默认，不必承担量化伪影。
