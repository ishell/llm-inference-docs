---
source: https://vllm.ai/blog/2024-10-17-spec-decode
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# 投机解码：最多 2.8×，高 QPS 时会反过来咬人

英文对照：[en/vllm/blog/performance/spec-decode.md](../../../../en/vllm/blog/performance/spec-decode.md)  
原文：https://vllm.ai/blog/2024-10-17-spec-decode  
2024-10-17。署名 **vLLM Team**。学习译文，不是官方译本。整理自双周 Office Hours。下面的构造函数参数是**当时的**；今日请以 [speculative-decoding.md](../../features/speculative-decoding.md)（`--speculative-config`）为准。V1 的 2025 年 1 月 alpha 把投机解码列成 **V1 尚未支持**；这一篇是 V0 时代的接法。幻灯：[Google 文档](https://docs.google.com/presentation/d/1wUoLmhfX6B7CfXy3o4m-MdodRL26WvY3/edit#slide=id.p1)。录像：[YouTube](https://youtu.be/eVJBFajJRIU)。报名当时走 [Neural Magic community office hours](https://neuralmagic.com/community-office-hours/?utm_campaign=vLLM%20Office%20Hours&utm_source=vllm-blog)。

投机解码让小模型和大模型搭伙，加速吐 token。下文拆开：它在 vLLM 里怎么工作，以及能换来多少性能。

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

## An Introduction to Speculative Decoding

投机解码（[Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)）是砍 token 生成延迟的一把刀。小模型处理简单的预测；大模型核对或改写。加速，但不牺牲准确——相对大模型的分布 **无损**。

**为什么延迟可能掉下去？** 传统 LLM 自回归：给一个 prompt，吐 T1、T2、T3，各付一次前向。投机解码把这件事改成：一串提议、一次核对。

三拍：

1. **Draft Model**：更小、更便宜的模型，一个一个提出 token。
2. **Target Model Verification**：大模型 **一次前向** 核完整串。对的留下，错的从第一个错处改正。
3. **Multiple Tokens in One Pass**：不再一步一个字；一次前向可以提交多个 token，延迟下来。

**图 08 的例子。** draft 提出 `["I", "like", "cooking", "and", "traveling"]`，交给 target 并行核对。第三个 `"cooking"` 该是 `"playing"`。这一步只收下 `["I", "like", "playing"]`——五次草稿，一次大模型，换来三个被接受的 token（miss 被改正，不是整段作废）。

他们把这条路写成：小规模、大规模部署都能用。

## How Speculative Decoding Works in vLLM

Continuous batching 还在：不同请求挤在同一个 batch，吞吐才上得去。两套组件：

- **Draft Runner**：跑小模型，提出候选。
- **Target Runner**：跑大模型，核对。

系统按这条路接好，投机解码才能和 continuous batching 一起转。

**图 01。** draft runner 与 target runner 在 vLLM batching 里怎么碰头。

引擎里有两处必须改：

1. **Scheduler**——一次前向里给多枚 token 留槽，提议和核对才能同一步。
2. **Memory Manager**——同时养**两份 KV**（draft 与 target）。

**图 09。** vLLM 里投机解码的系统架构。

## Types of Speculative Decoding Supported in vLLM

当时三种，对着不同负载。

### Draft Model-Based Speculative Decoding

最常见：小模型猜下一串 token，大模型核对。文中例子：**Llama 68M** 给 **Llama 2 70B** 打草稿。草稿要足够小以免拖死，又要足够准才有净收益。选对草稿，才是效率的关键。

选起来却不容易。Llama 3 一类常常找不到合身的小草稿——**词表必须相同**。词表对不上，这条路就走不通。所以下面两条「不用第二份权重、或把头长在大模型上」。

**图 02。** draft 模型那条路。

### Prompt Lookup Decoding

也叫 n-gram matching。摘要、一部分问答里，答案常常从 prompt 里抄。不用小模型提议，直接拿 prompt 里已经有的信息来猜。大模型爱在答案里重复 prompt 时，这条路特别灵。

**图 03。** 把 prompt 里所有 **2-gram** 当查找键，值是键后面的 **三个 token**。生成时当前 2-gram 若命中，就把后面那几个提出来。

### Medusa / EAGLE / MLPSpeculator

在 **target 自己**身上加层（或头），一次前向猜后面几个位置。仍没有单独的 draft 权重，靠大模型自己的容量做并行吐字。他们说这还**初步**，kernel 越好，越像正路。

**图 04** 来自 [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa)：三个头，输入都是最后一块 transformer 的输出。Head 1 为位置 1 提出 `["is", "'", "the"]`；Head 2 为位置 2 提出 `["difficult", "is", "'"]`；Head 3 为位置 3 提出 `["not", "difficult", "a"]`。

## Speculative Decoding Performance Insights: Speedups and Trade-offs

**低 QPS** 上是大便宜。ShareGPT 上，draft 模型那条路，token 生成最多约 **1.5×**。摘要数据集（CNN/DailyMail）上，prompt lookup 最多约 **2.8×**。

**图 05–06。** **QPS = 1**，Llama-3-70B，**4×H100**：ShareGPT + draft `turboderp/Qwama-0.5B-Instruct` 最多约 **1.5×**；CNN/DailyMail + n-gram 最多约 **2.8×**。

**高 QPS** 时，提议和核对付的那笔算力，可能反过来咬人——系统已经 **compute-bound**，请求变密，税比礼物重。

**图 07。** 高 QPS：ShareGPT 大约 **1.4× 变慢**；CNN/DailyMail 大约 **1.8× 变慢**（同一套 70B / 4×H100）。

## On the Roadmap: Dynamic Adjustments for Better Performance

要让投机解码在高负载下也敢开：**dynamic speculative decoding**。论文：[arXiv:2406.14066](https://arxiv.org/abs/2406.14066)。他们把它写成 vLLM 正在做的研究方向。按系统负载和草稿准不准，改投机 token 的个数。经验规则：系统忙时**缩短提议长度**；平均 token **接受率**高时，少砍一点（图 10）。

以后每一步自动改投机程度，让人不必先猜自己的 QPS 再决定开不开。**这篇发布时还没落地。**

## How to Use Speculative Decoding in vLLM

当时说：起 vLLM server 时带上旗标，指定投机模型、token 数、tensor parallel size。下面三块却是**离线** `LLM(...)` 构造函数——不是今天的 `--speculative-config` JSON。当时文档：[v0.6.0 spec_decode](https://docs.vllm.ai/en/v0.6.0/models/spec_decode.html)。

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

有时希望草稿用更小的 TP，少占资源、少通信，重活留给 target。例子：target `tensor_parallel_size=4`，draft `speculative_draft_tensor_parallel_size=1`，草稿 `ibm-fms/llama3-70b-accelerator`：

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

后续（同一篇论文 + [RFC #4565](https://github.com/vllm-project/vllm/issues/4565)）会自动选择 `num_speculative_tokens`，少一次手工配置。

文档当时指向上面那页 spec_decode。问题与反馈走双周 office hours。

## Conclusion: The Future of Speculative Decoding in vLLM

投机解码在 **低 QPS** 上是大便宜。动态长度落地以后，他们希望高 QPS 也能用——延迟下来、效率上去，变成 serving 里一件常开的工具。

这一篇负责把不等式说清楚：**低 QPS 像魔法，高 QPS 像税**——直到提议长度能跟着负载和接受率一起动。这条线后来会长到 DSpark、EAGLE-3、MTP；起点是这里。
