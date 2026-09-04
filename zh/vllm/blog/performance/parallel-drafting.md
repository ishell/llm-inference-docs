---
source: https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 并行草稿：一路并行到验收

英文对照：[en/vllm/blog/performance/parallel-drafting.md](../../../../en/vllm/blog/performance/parallel-drafting.md)  
原文：https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting  
2026-07-28。署名 **Alexandre Marques、Megan Flynn、Helen Zhao、Krishna Teja Chitty Venkata、Chibueze Ukachi（Red Hat AI）**。学习笔记。Speculators + vLLM 给三只并行 drafter 开源支持：[P-EAGLE](https://arxiv.org/abs/2602.01469)、[DFlash](https://arxiv.org/abs/2602.06036)、[DSpark](https://arxiv.org/abs/2607.05147)。Checkpoint 在 [Speculators Collection](https://huggingface.co/collections/RedHatAI/speculator-models)。

**勘误（2026-07-29），文末另有一节：** Figure 1 的图后来改过。原先数字和声称的评测环境对不上（环境配错了）。模型之间的 **相对** 排名仍一致，文中结论没改。Markdown 原文 **没有** 把那些曲线的 TPS / ITL / OTPS 写成表——不要从图里编数字。

P-EAGLE 细节和 B200 表：[p-eagle](p-eagle.md)。DSpark 按信心改验收预算：[dspark-adaptive](dspark-adaptive.md)。验收数学仍是 [spec-decode](spec-decode.md)。后来 EAGLE 注意力漂移的修法：[eagle-3-1](eagle-3-1.md)。训练用的 hidden 导出：[extract-hidden-states](../architecture/extract-hidden-states.md)。

验收仍是 **rejection sampling**。投机解码精确保持 verifier 的输出分布，和普通解码数学上同分布。这篇改的是 **草稿怎么长出来**，不是验收公式。

## 1. Introduction

投机解码已经是 serving 里对付 memory-bandwidth 墙的核心手段：一次 verifier 前向验收一串候选，生产上能换到可观的加速。

基础设施往前走，传统投机栈的顶却卡在 **draft token 怎么生成**。这篇是 Speculators + vLLM 给 P-EAGLE、DFlash、DSpark 三条并行草稿开源支持。

![compare interactivity qwen38b math](../../../../assets/vllm/blog/performance/parallel-drafting/01-compare_interactivity_qwen38b_math.png)

![compare interactivity qwen330b humaneval](../../../../assets/vllm/blog/performance/parallel-drafting/02-compare_interactivity_qwen330b_humaneval.png)

![compare interactivity gemma431b humaneval](../../../../assets/vllm/blog/performance/parallel-drafting/03-compare_interactivity_gemma431b_humaneval.png)

**Figure 1。** 并行草稿（P-EAGLE、DFlash、DSpark）对照自回归 EAGLE-3，三套负载。Checkpoint：[Speculators Collection](https://huggingface.co/collections/RedHatAI/speculator-models)。**看 2026-07-29 更新后的图**；见勘误。帖子正文没有把曲线值制成表。

## 2. The Limits of Recursive Drafting

[EAGLE](https://arxiv.org/abs/2401.15077) 和 [MTP](https://arxiv.org/abs/2404.19737) 换过一次范式：speculator 直接读 verifier 的 hidden，不再只盯表面文字，接受率跳了一截。

[EAGLE-3](https://arxiv.org/abs/2503.01840) 仍是 **自回归草稿**。一串候选 = **每个 draft token 一次前向**。

生产上两笔税：

- **模型不能大。** 草稿成本跟投机长度线性涨，speculator 只能极瘦，免得把 verifier 刚省下的时间吃回去。
- **参数要跟人盯。** 线性缩放实际上把 K 卡住。最优投机长度变成运维旋钮，要按用例和实时负载拧。

![ar vs parallel](../../../../assets/vllm/blog/performance/parallel-drafting/04-ar_vs_parallel.jpg)

**Figure 2。** 并行草稿：一步多个 draft token。自回归草稿：一步一个。

## 3. The Shift to Parallel Drafting

并行草稿把草稿阶段的顺序执行拆掉：一整块候选 **同时** 预测。草稿压成一次前向，出候选的延迟就和猜多少个 **脱钩**。

原文点名的两件事：

- **表达力有地方放。** Speculator 每块只跑一次，可以用更大、更深的 draft。上下文更够，接受率更高，却没有一串顺序延迟。
- **调参简单。** 草稿成本不再跟着块长走，不必对着波动的负载去拧 K。

这想法并不新：[Medusa](https://arxiv.org/abs/2401.10774)、[PARD](https://arxiv.org/abs/2504.18583)。P-EAGLE、DFlash、DSpark 是在并行执行上再叠 **深层 verifier 状态条件**——EAGLE 真正管用的那一insight。

## 4. Under the Hood: Inference & Training Architecture

三只都吃 verifier hidden，都并行吐 draft token，路不一样。Figure 3 并排画。

![diagram](../../../../assets/vllm/blog/performance/parallel-drafting/05-diagram.jpg)

**Figure 3。** P-EAGLE 把 verifier hidden 当 speculator **输入**。DFlash 把 hidden **投影进** speculator 的 **KV-cache**。DSpark 在 DFlash 骨架上再加顺序修正和信心估计。

共同的训练麻烦：并行 speculator 要在训练序列的 **每一个** token 位置做 next-K。长度 N、lookahead K：对整张矩阵算 loss，显存和算力都会炸。三只各自稀疏。

### P-EAGLE

底座仍是 EAGLE：verifier hidden 当输入特征。不是一个一个吃，而是把它们铺到多个未来位置，一次并行吐整段候选。

**这篇** 写的训练手法：**draft block sparsification**——沿 lookahead 维 K 按 **衰减率** 丢掉 token，loss 集中在最近的位置，远处未来从 loss 里剪掉。（单独的 [P-EAGLE 帖](p-eagle.md) 写的是长 N 上的 sequence partition；两篇各记各的，不要并成一句。）

### DFlash

Verifier 特征走另一条。不当普通输入，而是 **投影** 之后 **注入 speculator 的 KV-cache**。Attention 被 verifier 状态紧紧条件住，**输入序列并不变长**。候选块用 **block diffusion** 生成。

训练：**sequence length sparsification**。不要在长度 N 的每个位置算 block loss；沿时间轴抽随机 **anchor**，只在这些交点上做块预测。省显存，覆盖仍有代表性。和 [speculators-v050](speculators-v050.md) 同一族想法。

### DSpark

DFlash 的并行骨架，再叠两件：

1. 一只轻量 **自回归修正头**，让后面的 token 更强地条件在已经出来的 token 上。并行吞吐加上顺序连贯。
2. 一只 **confidence head**：draft token **进 verifier 之前** 打分，只把像会被接受的送去。并行草稿可以很便宜地猜很多，verifier 仍要为每一个付钱。信心用来少做冤枉验收。Serving 侧用这只头排预算：[dspark-adaptive](dspark-adaptive.md)。

## 5. Inference Performance

Figure 1 是对照 EAGLE-3。三组模型 × 算法：

| Model | Algorithm | Use case | Hardware |
| --- | --- | --- | --- |
| Qwen3-8B | [P-EAGLE](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.peagle) | Math reasoning (GSM8k) | 1×A100 |
| Qwen3-30B-A3B | [DFlash](https://huggingface.co/RedHatAI/Qwen3-30B-A3B-speculator.dflash) | Coding (HumanEval) | 2×A100 |
| gemma-4-31B-it | [DSpark](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.dspark) | Coding (HumanEval) | 2×A100 |

勘误之后原文仍说：三组里并行草稿都 **明显** 好过 EAGLE-3；环境修好后相对排名还在。绝对值 **没有** 写进正文。模型、任务、硬件都会变——作者让你用自己的负载去量。

## 6. Production Serving with vLLM and Speculators

Speculators 是训和评的统一生态，接到 vLLM。并行投机引擎：初始化时把配置传进去。原文 DFlash 例子：

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 2 \
  --reasoning-parser qwen3 \
  --speculative-config '{
    "model": "RedHatAI/Qwen3-30B-A3B-speculator.dflash",
    "num_speculative_tokens": 7,
    "method": "dflash"
  }'
```

P-EAGLE 在 vLLM 里是 `method: eagle3` 再加 `"parallel_drafting": true`（见 [p-eagle](p-eagle.md)）。DSpark 在姊妹篇里的 method 名是 `"dspark"`（[dspark-adaptive](dspark-adaptive.md)）。

块级并行草稿：管线一路并行到底。相对普通解码仍靠 rejection sampling 无损。

## 7. Get Started

原文当时的判断：并行草稿已经完全支持、开源、能进生产。按文档路径自己训，在 vLLM 里原生测。

- 仓库：[Speculators](http://github.com/vllm-project/speculators)
- 预训练：[Speculators Collection on HuggingFace](https://huggingface.co/collections/RedHatAI/speculator-models)
- 训练指南：[Speculator tutorials](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/tutorials/index.md)

## Errata（2026-07-29）

Figure 1 的图在 **2026-07-29** 更新。原先图上的数字和声称的评测环境不一致，原因是 **环境配错**。模型之间的相对行为一致；**文中结论没有改**。勘误正文没有另给一张替换数字表。
