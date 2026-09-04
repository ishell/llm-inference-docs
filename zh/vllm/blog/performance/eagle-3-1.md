---
source: https://vllm.ai/blog/2026-05-26-eagle-3-1
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# EAGLE 3.1：压住 attention drift

英文对照：[en/vllm/blog/performance/eagle-3-1.md](../../../../en/vllm/blog/performance/eagle-3-1.md)  
原文：https://vllm.ai/blog/2026-05-26-eagle-3-1  
2026-05-26。署名 **EAGLE Team、vLLM Team、and TorchSpec Team**。学习笔记。仓库：[SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)、[vllm-project/vllm](https://github.com/vllm-project/vllm)、[lightseekorg/TorchSpec](https://github.com/lightseekorg/TorchSpec)。验收数学仍是 [spec-decode](spec-decode.md)。一次前向猜 K 个字的亲戚：[P-EAGLE](p-eagle.md)。EAGLE-3 训练底座：[speculators-v030](speculators-v030.md)。页上的吞吐是他们的 SPEED-Bench，不是你的 SLA。

EAGLE 系列（1 / 2 / 3）已经是研究和生产里用得最广的投机解码家族之一。这篇是三家一起推的 **EAGLE 3.1**：更稳、更敢上 serving。

## 创新

受控评测里投机解码可以很好看；换 chat template、拉长上下文、换 OOD system prompt，接受长度就掉。

EAGLE 团队把这种脆归结为 [attention drift](https://arxiv.org/pdf/2605.09992)：猜得越深，草稿注意力离开 sink token，盯住自己刚吐的字。

底下两处。一是融合输入越来越不平衡，高层 hidden 把草稿输入占满。二是未归一化的 residual 让 hidden 幅度跨投机步膨胀。合在一起，越深越不稳。

本地图（原文版权仍归原站；学习对照用）：

![pre norm vs post norm](../../../../assets/vllm/blog/performance/eagle-3-1/01-pre-norm-vs-post-norm.png)

**Figure 1。** EAGLE 3 对 EAGLE 3.1。3.1：每路 target hidden 进 FC **之前** 做 FC normalization；下一步吃 **post-norm** hidden。

两处结构改动：

- 每路 target hidden 之后、FC 之前做 **FC normalization**
- 下一步解码吃 **post-norm** hidden

直觉上，post-norm 更像跨步 **递归调用** 草稿，而不是往 target 后面再叠层。

相对 EAGLE 3，原文声称：

- 训练时到推理时的外推更好
- 长上下文更稳
- 更扛得住 chat template / system prompt 变化
- 不同 serving 环境里接受长度更稳

长上下文负载上，EAGLE 3.1 的接受长度相对 EAGLE 3 最多约 **2×**。

## 用 TorchSpec 训

[TorchSpec](https://github.com/lightseekorg/torchspec) 现在能训 [EAGLE 3.1](https://github.com/lightseekorg/TorchSpec/pull/97)，也给后面的投机算法留门。训练税更低，试新算法更快。

基于 TorchSpec 和 vLLM，他们训并开源了 Kimi K2.6 的 EAGLE 3.1 草稿：

https://huggingface.co/lightseekorg/kimi-k2.6-eagle3.1-mla

这只模型是「TorchSpec 训、vLLM 服」落在真实 serving 模型上的例子。

## 接到 vLLM

EAGLE 3.1 在 vLLM 里是现有 EAGLE 3 实现上的 **config-driven** 扩展（[PR #42764](https://github.com/vllm-project/vllm/pull/42764)）。

接入包括：

- FC normalization
- post-norm hidden 回灌
- 拿掉对 target hidden 的硬编码假设

旧 EAGLE 3 checkpoint **仍然能用**。3.1 草稿走同一条投机解码路径，例如：

```bash
vllm serve nvidia/Kimi-K2.6-NVFP4 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --tool-call-parser kimi_k2 \
  --enable-auto-tool-choice \
  --reasoning-parser kimi_k2 \
  --attention-backend tokenspeed_mla \
  --speculative-config '{"model":"lightseekorg/kimi-k2.6-eagle3.1-mla","method":"eagle3","num_speculative_tokens":3}' \
  --language-model-only
```

生产里换草稿可以平滑。当时已合入 vLLM main，走 nightly，以及即将到来的 **v0.22.0**。

早期数据点：Kimi K2.6 EAGLE 3.1 草稿，底座 Kimi-K2.6-NVFP4，vLLM **TP=4**、**GB200**、非分离，SPEED-Bench coding。相对无投机 baseline：并发 **1** 时每用户输出吞吐 **2.03×**；并发上去仍有数——**C=4 为 1.71×**，**C=16 为 1.66×**。

![tpot baseline vs eagle31](../../../../assets/vllm/blog/performance/eagle-3-1/02-tpot_baseline_vs_eagle31.png)

**Figure 2。** 每用户输出吞吐（TPS）：Kimi-K2.6-NVFP4，vLLM，TP=4，GB200，SPEED-Bench coding。EAGLE 3.1-MLA 对无投机 baseline。

## 开源协作

算法研究（EAGLE）、系统优化（vLLM）、训练基础设施（TorchSpec）三家合在同一条线上。EAGLE 继续推算法；vLLM 把创新送进规模化推理；TorchSpec 让下一只投机算法更好训、更好试。

NVIDIA 提供 GPU 和持续合作：开发、验证、把 3.1 从算法接到能部署的评测，都靠这层支持。

他们希望把投机解码的基线再抬一截，让更广的 LLM 生态在 token 效率上往前走。
