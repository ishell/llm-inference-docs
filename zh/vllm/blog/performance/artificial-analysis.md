---
source: https://vllm.ai/blog/2026-05-11-vllm-tops-artificial-analysis
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Artificial Analysis 榜：三只模型三只瓶颈，融合和 draft 都在 main

英文对照：[en/vllm/blog/performance/artificial-analysis.md](../../../../en/vllm/blog/performance/artificial-analysis.md)  
原文：https://vllm.ai/blog/2026-05-11-vllm-tops-artificial-analysis  
2026-05-11。署名 **vLLM Team**。学习笔记。2026-05 DigitalOcean / Artificial Analysis 当天的板，不是你的 SLA。Pareto 亲戚：[gpt-oss-optimizations.md](gpt-oss-optimizations.md) / [qwen35-25k-tps.md](../serving/qwen35-25k-tps.md)。V3.2 稀疏路径：[deepseek-v32.md](../architecture/deepseek-v32.md)。后来复用：[deepseek-v4.md](../architecture/deepseek-v4.md)。Draft 训练：[eagle-3-1.md](eagle-3-1.md) / [speculators-v050.md](speculators-v050.md)。MRV2：[mrv2.md](../architecture/mrv2.md)。系统 TPS ≠ 每用户 TPS。

适用：看三只模型各自卡在哪、哪几个 fusion / draft 进了 main。不适合：把页上的 **230 TPS** 当 SLA。

**原文 TL;DR。**

- DigitalOcean [发了](https://www.digitalocean.com/blog/how-we-built-fastest-deepseek-minimax-qwen-on-blackwell-ultra) 三只前沿开源权重在 Blackwell Ultra 上的部署。底下引擎是开源 vLLM。对专有栈的说法：同一块硅，榜上第一。
- DeepSeek V3.2：低 batch 被 **launch** 钉死；attention 路径 ~33 kernel → ~10，bs=1 约 **1.28×**（85.8→109.3 tok/s，4×GB200，无 MTP）。单 8×B300 cc=1：无 MTP TP8 **125**；MTP=1 **234**（接受率 ~90%）；P/D TP4+TP4+MTP=3 **262**。router GEMM 再约 **6%**；indexer TopK 单 graph，128K Decode 最高约 **17%** TPOT。
- MiniMax-M2.5：TorchSpec EAGLE3 + `fuse_minimax_qk_norm`。天花板（合成 100% 接受）TP4 **326 tok/s**。
- Qwen 3.5 397B：漏掉的 `allreduce_rms` 让 Decode 一半时间耗在未融合跨卡 reduce；修完 + post-conv fusion + dual-stream。TEP=8 cc=1 **163 tok/s**，cc=256 **6.69→7.33 req/s**。
- 改动在 vLLM `main` 或在飞。页上标题数字：DeepSeek V3.2 最佳每用户输出 **230 TPS**（多数 provider 的 4× 以上）；Qwen 3.5 397B 12 家里第一，1 万 token prompt 的 TTFT 不到 1 s。

![hero](../../../../assets/vllm/blog/performance/artificial-analysis/01-hero_image.png)

*How vLLM built the leading deployments of DeepSeek V3.2, MiniMax-M2.5, and Qwen 3.5 397B.*（页上封面）

## 怎么快起来的

一只模型一只瓶颈：

1. **DeepSeek V3.2：** 低 batch 狠做 kernel fusion（也是 [DeepSeek V4](../architecture/deepseek-v4.md) 的底座）。
2. **MiniMax-M2.5：** 定点 fusion + 自训 EAGLE3 draft，开源 [TorchSpec](https://github.com/torchspec-project/TorchSpec) 和 vLLM。同一只 draft 能上 M2.7（架构一样）。
3. **Qwen 3.5 397B：** 对着线性注意力和归一化路径做 fusion。

## DeepSeek V3.2：低 batch 的 kernel fusion

低 batch 时 V3.2 被 **kernel launch** 钉死，不是算力。每层几十个小 kernel（norm、RoPE、quant），GPU 微秒就跑完，launch 税占满墙钟。

Attention 路径上的 op fusion：Q / KV norm、Q / KV 的 RoPE、indexer 的 layer-norm + RoPE、FP8 quant、KV-cache 写入——收成一对 fused kernel，attention 和 MoE 以外都盖进去。每层 kernel 数约 **33 → ~10**。

![DSv3.2 attention-path fusion](../../../../assets/vllm/blog/performance/artificial-analysis/02-figure1.png)

**Figure 1。** Attention 路径融合：~33 次 launch → ~10。batch=1 约 **1.28×**。

单靠 fusion：bs=1 **1.28×**（85.8 → 109.3 tok/s，4× GB200，无 MTP）。单 8× B300、concurrency 1：

- 无 MTP（TP=8）：**125 tok/s**
- MTP=1（TP=8）：**234 tok/s**（草稿接受率 ~90%）
- Prefill/Decode 拆分（TP=4 + TP=4 + MTP=3）：**262 tok/s**

Fusion 之后还有两只模型专用 kernel：

- 对着 DSv3 MoE routing 维、小 Decode batch 的 router GEMM——batch=1 再 **6%**（[#34302](https://github.com/vllm-project/vllm/pull/34302)）。
- 稀疏注意力 indexer TopK：按行、按序列长度选算法，所有情况进 **一张 CUDA graph**。128K 上下文 Decode 上每 token 延迟最高约 **17%**（[#37421](https://github.com/vllm-project/vllm/pull/37421)）。

同一套活现在托着 DeepSeek V4（复用 Q RoPE + quant 和 QK-norm fusion）。

![DeepSeek V3.2 Non-Reasoning](../../../../assets/vllm/blog/performance/artificial-analysis/03-figure2.png)

**Figure 2。** DeepSeek V3.2 Non-Reasoning，各家输出速度。来源：[Artificial Analysis](https://artificialanalysis.ai/models/deepseek-v3-2/providers#output-speed)，2026-05。

![DeepSeek V3.2 Reasoning](../../../../assets/vllm/blog/performance/artificial-analysis/04-figure3.png)

**Figure 3。** DeepSeek V3.2 Reasoning，各家输出速度。来源：[Artificial Analysis](https://artificialanalysis.ai/models/deepseek-v3-2-reasoning/providers#output-speed)，2026-05。

## MiniMax-M2.5：EAGLE3 和更多 fusion

[Inferact](https://inferact.ai) 用 [TorchSpec](https://github.com/torchspec-project/TorchSpec) 训自定义 EAGLE3：FSDP 训 draft，同时 vLLM 跑 target。Draft 吃的是活的 vLLM hidden，数据是 MiniMax-M2.5 重生的回复——对齐基座的 token 分布，不是一份泛 SFT。

让这事能跑的 MRV2 投机管道：修 draft metadata，抬后面几位的接受（[#38311](https://github.com/vllm-project/vllm/pull/38311)）；draft prefill 走 CUDA graph（[#37588](https://github.com/vllm-project/vllm/pull/37588)）。

Fusion：`fuse_minimax_qk_norm` 对付非标准 attention norm——Q 和 K 的方差先在 TP rank 之间 reduce，**再** 上 per-channel scale（[#37045](https://github.com/vllm-project/vllm/pull/37045)）。

![fuse_minimax_qk_norm](../../../../assets/vllm/blog/performance/artificial-analysis/05-figure4.png)

**Figure 4。** 四路 TP 上 `fuse_minimax_qk_norm` 的解剖。

再加上 `fuse_norm_quant`、`fuse_act_quant`、`fuse_gemm_comms`，**天花板**实验（合成 100% 接受，把 fusion 和 draft 质量拆开）：

- concurrency 1 **326 tok/s**（TP=4，EAGLE3 + 3 枚投机 token）。

![MiniMax-M2.5 providers](../../../../assets/vllm/blog/performance/artificial-analysis/06-figure5.png)

**Figure 5。** MiniMax-M2.5，各家输出速度。来源：[Artificial Analysis](https://artificialanalysis.ai/models/minimax-m2-5/providers#output-speed)，2026-05。

## Qwen 3.5 397B：线性注意力和 fusion 缺口

Qwen 3.5 用线性注意力，attention block 里的归一化也不标准。两件事都跟 vLLM 现成 fusion 别扭：投影后的 convolution 是线性注意力独有；归一化变体对不上 `allreduce_rms`。

Profiler：漏掉 `allreduce_rms` → Decode 大约 **一半** 耗在未融合跨卡 reduce。数字没错，多付了 HBM 往返。

四块：

- 让 `allreduce_rms` 认出 Qwen 那一档 norm——batch > 1 时 TPOT 约 **5%**。
- qk-norm + RoPE 路径的 kernel 级优化。
- 对着线性注意力架构的 post-conv fusion（[#37813](https://github.com/vllm-project/vllm/pull/37813)）。
- Dual-stream 叠独立计算枝。

![Qwen 3.5 fusion](../../../../assets/vllm/blog/performance/artificial-analysis/07-figure6.png)

**Figure 6。** vLLM 里 Qwen 3.5 397B 的 kernel fusion。

TP=8 + expert parallelism，生产部署：

- concurrency 1 **163 tok/s**（TEP=8，post-conv fusion）
- concurrency 256 **7.33 req/s**，基线 **6.69 req/s**（**+10%**）

已进 vLLM `main`。

![Qwen 3.5 providers](../../../../assets/vllm/blog/performance/artificial-analysis/08-figure7.png)

**Figure 7。** Qwen 3.5 397B，各家输出速度。来源：[Artificial Analysis](https://artificialanalysis.ai/models/qwen3-5-397b-a17b/providers#output-speed)，2026-05。

## 这意味着什么 / 开源默认

DSv3.2 attention 路径 fusion、MiniMax EAGLE3 训练菜谱、Qwen 3.5 fusion：已经在上游或在路上。现成 vLLM 就能拿到同一套加速。

页上的说法：历史上最快的推理栈是专有的。这几块 Artificial Analysis 板上，他们量到的最快推理是开源。

## 致谢（页上点名）

Inferact、DigitalOcean、NVIDIA、Red Hat，以及 vLLM 开源社区。
