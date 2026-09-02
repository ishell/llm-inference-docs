---
source: https://vllm.ai/blog/2024-10-17-spec-decode
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# 投机解码：最多 2.8×，高 QPS 时会反过来咬人

英文对照：`en/vllm/blog/performance/spec-decode.md`  
原文：https://vllm.ai/blog/2024-10-17-spec-decode  
2024-10-17，整理自 Office Hours。CLI 旗标是当时的；今日请以文档 `speculative_decoding` 为准。图、幻灯、录像在原网页。


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

## 它在干什么

Leviathan et al., 2023。小模型（draft）一个一个猜；大模型（target）**一次前向**核对这一串。对的留下，错的从第一个错处切开，错处由 target 改正。无损：最终分布仍是大模型的。

例子：draft 提出 `["I", "like", "cooking", "and", "traveling"]`，target 发现第三个该是 `"playing"`，这一步只收下 `["I", "like", "playing"]`。五次草稿，一次大模型，换来三个被接受的 token。

传统 decode 一步一个字。投机把「一步」变成「一串」。延迟掉下去的条件是：草稿足够准，而核对的那一次前向摊得起。

## 在 vLLM 里怎么接

Continuous batching 还在：不同请求挤在同一个 batch。两套 runner——**Draft Runner** 跑小模型，**Target Runner** 跑大模型。调度器要能在一次前向里给多枚 token 留槽；内存管理器要同时养两份 KV（draft 与 target）。

## 三种草稿

**独立 draft 模型。** 最常见。例如 Llama 68M 给 Llama 2 70B 打草稿。要足够小以免拖死，又要足够准才有净收益。draft 与 target **词表必须相同**——Llama 3 一类，常常因此找不到合身的草稿，于是才有下面两条「不用第二份模型」的路。

**Prompt lookup / n-gram。** 摘要、问答里，答案常常从 prompt 里抄。把 prompt 里的 n-gram 当查找表，命中就提出后续若干 token。没有第二份权重。

**Medusa / EAGLE / MLPSpeculator。** 在大模型自己身上加头，一次前向猜后面几个位置。仍在长，kernel 越好，越像正路。

## 何时快、何时慢

低 QPS（他们举 QPS=1）：ShareGPT 上 Llama 3-70B、4×H100、draft=`turboderp/Qwama-0.5B-Instruct`，最多约 **1.5×**；CNN/DailyMail 上 n-gram 最多约 **2.8×**。

高 QPS：同一套 70B / 4×H100，ShareGPT 大约 **1.4× 变慢**，CNN/DailyMail 大约 **1.8× 变慢**。系统已经 compute-bound 时，再为草稿和核对付钱，会把已经满员的车厢再塞人。

路线图上的 **dynamic speculative decoding**：按负载和接受率改提议长度——忙时缩短，接受率高时少砍。目标是让人敢默认打开，而不必先猜自己的 QPS。当时还在做。

## 当时的 API（离线）

Draft 模型：

```python
from vllm import LLM
llm = LLM(
    model="facebook/opt-6.7b",
    speculative_model="facebook/opt-125m",
    num_speculative_tokens=5,
)
```

n-gram：

```python
llm = LLM(
    model="facebook/opt-6.7b",
    speculative_model="[ngram]",
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,
    ngram_prompt_lookup_min=1,
)
```

草稿可以用更小的 TP，减少通信：

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_model="ibm-fms/llama3-70b-accelerator",
    speculative_draft_tensor_parallel_size=1,
)
```

必读里后头还有 FP8 KV、生产级 CI；投机解码这条线会一直长到 DSpark、EAGLE-3、MTP。这一篇负责把「为什么低 QPS 像魔法、高 QPS 像税」说清楚。
