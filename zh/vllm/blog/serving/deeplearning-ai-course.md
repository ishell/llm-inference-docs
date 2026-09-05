---
source: https://vllm.ai/blog/2026-06-03-deeplearning-ai-vllm-course
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# DeepLearning.AI 课：压缩 → serve → GuideLLM 压测，不是新 kernel

英文对照：[en/vllm/blog/serving/deeplearning-ai-course.md](../../../../en/vllm/blog/serving/deeplearning-ai-course.md)  
原文：https://vllm.ai/blog/2026-06-03-deeplearning-ai-vllm-course  
2026-06-03。**Cedric Clyburn**（Red Hat）与 [DeepLearning.AI](https://www.deeplearning.ai/) / [Andrew Ng](https://en.wikipedia.org/wiki/Andrew_Ng)。课：[Fast & Efficient LLM Inference with vLLM](https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm)。免费。学的是 **compress → serve → benchmark** 闭环，不是 vLLM 内核新机制。LLM Compressor 也出现在 [laguna-xs2.md](../performance/laguna-xs2.md)；GuideLLM 进 UI 见 [playground.md](playground.md)。GuideLLM ≠ AIPerf。

> "Deploying open-source LLMs efficiently, for many users, with low latency and reasonable cost, is challenging. This course shows you how." — Andrew Ng

原文有课程封面、结构图、KV / 量化示意、三张 lab 截图；本仓库没有那些原图的副本。

## 课是怎么拼起来的

vLLM 生态已经不只是引擎：压缩走 [LLM Compressor](https://github.com/vllm-project/llm-compressor)，部署压测走 [GuideLLM](https://github.com/vllm-project/guidellm)。这门课要演示的是：规模化部署时这几块怎么咬在一起。

和 Mountain View 的 Andrew Ng 团队合作，材料按许多部署会走的工作流来切：**压缩**模型以适配硬件，用 vLLM **serve**，再 **benchmark** 速度–成本–精度的交易。动手 lab 之前先铺推理与内存：为什么 continuous batching、PagedAttention、prefix caching 会帮忙。

**原文图 caption（未收录）：** *The course covers hardware requirements, memory hierarchy, and optimization techniques before diving into hands-on labs.*

## 他们把力气花在哪

很大一块花在**可视化**：推理内部、KV cache、GPU 内存层次。

拆 inference 时的 transformer：token 怎么流、每层算什么、瓶颈实际住在哪。KV：它在 GPU 内存里长什么样、每生成一个 token 怎么涨、并发用户为什么会把内存压得很凶。

**原文图 caption（未收录）：** *Visualizing how the KV cache grows during autoregressive generation in the course.*

量化：从默认发布的 **FP16** 权重走到 **INT8** 或 **INT4**，好处和代价。页上**没有**体积比或 perplexity 数字。

**原文图 caption（未收录）：** *Breaking down weight-only vs. weight-and-activation quantization and the GPU memory hierarchy.*

## 课里有什么

三截。每一截都有 JupyterLab lab，对着真模型和一台正在跑的 vLLM server。

### Compress

拿一份全精度 **Qwen**，用 LLM Compressor 量化。比量化前后体积；用 **perplexity** 量精度交易。体会部署时怎样减 GPU 内存。页上没有具体体积或 perplexity 值。

**原文图 caption（未收录）：** *Quantizing a Qwen model with LLM Compressor in the course lab.*

### Serve

用 [vLLM](https://github.com/vllm-project/vllm) 部署，走 **OpenAI-compatible API**。从 vLLM 指标里看 continuous batching：并发上来时内存怎么变；请求共享系统提示时 prefix caching 怎样少算一遍。

**原文图 caption（未收录）：** *Watching vLLM's serving metrics live as concurrent requests hit the server.*

### Benchmark

用 GuideLLM 模拟流量：负载下的延迟和吞吐。再用 [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) 确认压缩后的模型还过精度。结尾是对真模型跑完一轮 load/accuracy。帖子里没有 GuideLLM 百分位表。

**原文图 caption（未收录）：** *Running GuideLLM to benchmark a vLLM deployment under simulated traffic in the course lab.*

## 课的规格

- **Course**： [Fast & Efficient LLM Inference with vLLM](https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm/)
- **Instructor**：[Cedric Clyburn](https://www.linkedin.com/in/cedricclyburn)，Red Hat Senior Developer Advocate
- **Duration**：约 **1.5 小时**，**9** 节视频，**3** 个动手 lab
- **Level**：Intermediate（假定会 Python、懂一点 LLM）

DeepLearning.AI 上免费。已经在本地或规模上跑过模型、想看见水面下的人；或者只是想动手摸一次 vLLM。

## 致谢

Red Hat：Saša Zelenović、Michael Goin、Sawyer Bowerman（设计、技术内容、lab）。DeepLearning.AI：Hawraa Salami（课程与制作）。Andrew Ng 的合作，以及在目录里给开源推理工具留的位置。
