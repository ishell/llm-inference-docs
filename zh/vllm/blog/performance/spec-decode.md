---
source: https://vllm.ai/blog/2024-10-17-spec-decode
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 投机解码：最多 2.8×，高 QPS 时会反过来咬人

英文对照：[en/vllm/blog/performance/spec-decode.md](../../../../en/vllm/blog/performance/spec-decode.md)  
原文：https://vllm.ai/blog/2024-10-17-spec-decode  
2024-10-17，整理自双周 Office Hours。下面的构造函数参数是**当时的**；今日请以 [speculative-decoding.md](../../features/speculative-decoding.md)（`--speculative-config`）为准。V1 的 2025 年 1 月 alpha 把投机解码列成 **V1 尚未支持**；这一篇是 V0 时代的接法。幻灯：[Google 文档](https://docs.google.com/presentation/d/1wUoLmhfX6B7CfXy3o4m-MdodRL26WvY3/edit#slide=id.p1)。录像：[YouTube](https://youtu.be/eVJBFajJRIU)。报名当时走 Neural Magic 的 community office hours。

本地图（原文版权仍归原站；学习对照用）：

![figure8](../../../../assets/vllm/blog/performance/spec-decode/01-figure8.png)

![figure1](../../../../assets/vllm/blog/performance/spec-decode/02-figure1.png)

![figure9](../../../../assets/vllm/blog/performance/spec-decode/03-figure9.png)

![figure2](../../../../assets/vllm/blog/performance/spec-decode/04-figure2.png)

![figure3](../../../../assets/vllm/blog/performance/spec-decode/05-figure3.png)

![figure4](../../../../assets/vllm/blog/performance/spec-decode/06-figure4.png)

![figure5](../../../../assets/vllm/blog/performance/spec-decode/07-figure5.png)

![figure6](../../../../assets/vllm/blog/performance/spec-decode/08-figure6.png)

![figure7](../../../../assets/vllm/blog/performance/spec-decode/09-figure7.png)

![figure10](../../../../assets/vllm/blog/performance/spec-decode/10-figure10.png)

## 它在干什么（Leviathan et al., 2023）

论文：[arXiv:2211.17192](https://arxiv.org/abs/2211.17192)。小模型（draft）一个一个猜；大模型（target）**一次前向**核对这一串。对的留下，错的从第一个错处切开，错处由 target 改正。无损：最终分布仍是大模型的。

延迟为什么可能掉下去：传统 Decode 是自回归的——T1、T2、T3 各付一次前向。投机把「一步一个字」变成「一串提议、一次核对」。

三拍：

1. **Draft** 便宜地、逐个提出 token。
2. **Target** 用一次前向核对整段：命中确认，第一个 miss 改写。
3. 草稿够准时，核对那一次前向摊得起，一步就能提交多个 token。

图 08 的例子：draft 提出 `["I", "like", "cooking", "and", "traveling"]`。target 发现第三个该是 `"playing"`。这一步只收下 `["I", "like", "playing"]`——五次草稿，一次大模型，换来三个被接受的 token（miss 被改正，不是整段作废）。

## 在 vLLM 里怎么接

Continuous batching 还在：不同请求挤在同一个 batch。

两套 runner：

- **Draft Runner** 跑小模型，提出候选。
- **Target Runner** 跑大模型，核对。

引擎里有两处必须改：

1. **Scheduler**——一次前向里给多枚 token 留槽，提议和核对才能同一步。
2. **Memory manager**——同时养**两份 KV**（draft 与 target）。

## 三种草稿（当时支持的）

### 1. 独立 draft 模型

最常见。文中例子：**Llama 68M** 给 **Llama 2 70B** 打草稿。要足够小以免拖死，又要足够准才有净收益。

硬约束：draft 与 target **词表必须相同**。Llama 3 一类常常因此找不到合身的小草稿——于是才有下面两条「不用第二份权重、或把头长在大模型上」的路。

### 2. Prompt lookup / n-gram

也叫 n-gram matching。摘要、一部分问答里，答案常常从 prompt 里抄。

图 03：把 prompt 里所有 **2-gram** 当查找键，值是键后面的 **三个 token**。生成时当前 2-gram 若命中，就把后面那几个提出来。没有第二份模型。

### 3. Medusa / EAGLE / MLPSpeculator

在 **target 自己**身上加头（或层），一次前向猜后面几个位置。仍没有单独的 draft 权重。

图 04 来自 [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa)：三个头，输入都是最后一块 transformer 的输出。Head 1 为位置 1 提出 `["is", "'", "the"]`；Head 2 为位置 2 提出 `["difficult", "is", "'"]`；Head 3 为位置 3 提出 `["not", "difficult", "a"]`。他们说这还**初步**，kernel 越好，越像正路。

## 何时快、何时慢

**低 QPS**（图上是 **QPS = 1**），Llama-3-70B，**4×H100**：

- ShareGPT + draft `turboderp/Qwama-0.5B-Instruct`：最多约 **1.5×**。
- CNN/DailyMail + n-gram：最多约 **2.8×**。

**高 QPS**，同一套 70B / 4×H100：

- ShareGPT：大约 **1.4× 变慢**。
- CNN/DailyMail：大约 **1.8× 变慢**。

系统已经 **compute-bound** 时，再为草稿和核对付钱，是往已经满员的车厢里塞人。多出来的前向是税，不是礼物。

## 当时路线图：dynamic speculative decoding

要让投机解码在高负载下也敢开：**dynamic speculative decoding**（[arXiv:2406.14066](https://arxiv.org/abs/2406.14066)；另见 [RFC #4565](https://github.com/vllm-project/vllm/issues/4565)）。他们把它写成 vLLM 正在做的研究方向。

文中的经验规则：系统忙时**缩短提议长度**；平均 token **接受率**高时，少砍一点（图 10）。目标是每一步自动改投机程度，让人不必先猜自己的 QPS 再决定开不开。**这篇发布时还没落地。**

## 当时的离线 API

当时文档：[v0.6.0 spec_decode](https://docs.vllm.ai/en/v0.6.0/models/spec_decode.html)。下面是构造函数关键字——不是今天的 `--speculative-config` JSON。

**Draft 模型**，每次猜 5 个 token：

```python
from vllm import LLM

llm = LLM(
    model="facebook/opt-6.7b",
    speculative_model="facebook/opt-125m",
    num_speculative_tokens=5,
)
outputs = llm.generate("The future of AI is")

for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

**n-gram**（`speculative_model="[ngram]"`）：

```python
from vllm import LLM

llm = LLM(
    model="facebook/opt-6.7b",
    speculative_model="[ngram]",
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,
    ngram_prompt_lookup_min=1,
)
outputs = llm.generate("The future of AI is")

for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

**草稿用更小的 TP**，少通信。例子：target `tensor_parallel_size=4`，draft `speculative_draft_tensor_parallel_size=1`，草稿 `ibm-fms/llama3-70b-accelerator`：

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_model="ibm-fms/llama3-70b-accelerator",
    speculative_draft_tensor_parallel_size=1,
)
outputs = llm.generate("The future of AI is")

for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

他们预期后续（同一篇论文 + RFC）会自动选择 `num_speculative_tokens`。

## 原文的收束

投机解码在 **低 QPS** 上是大便宜。动态长度是他们对 **高 QPS** 的赌注。这一篇负责把不等式说清楚：**低 QPS 像魔法，高 QPS 像税**——直到提议长度能跟着负载和接受率一起动。这条线后来会长到 DSpark、EAGLE-3、MTP；起点是这里。
