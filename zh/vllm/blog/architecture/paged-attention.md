---
source: https://vllm.ai/blog/2023-06-20-vllm
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# vLLM：用 PagedAttention 让 serving 便宜下来

英文对照：[en/vllm/blog/architecture/paged-attention.md](../../../../en/vllm/blog/architecture/paged-attention.md)  
原文：https://vllm.ai/blog/2023-06-20-vllm  
2023-06-20。作者：**Woosuk Kwon\***、**Zhuohan Li\***、Siyuan Zhuang、Ying Sheng、Lianmin Zheng、Cody Yu、Joey Gonzalez、Hao Zhang、Ion Stoica（\* 同等贡献）。学习译文，不是官方译本。论文稍后：[arXiv:2309.06180](https://arxiv.org/pdf/2309.06180.pdf)。原文页上还有 [GitHub](https://github.com/vllm-project/vllm) 和当时的 [Documentation](https://vllm.readthedocs.io/en/latest/)（Read the Docs）。营销 Logo 不收录。后来的 Anatomy、V1、prefix cache，都还站在这张页表上。

LLM 答应改写所有行业。真正把模型端上桌，却常常慢得像在昂贵的硬件上散步。这一天他们放出 **vLLM**：开源库，做快的推理和能给人用的 serving。核心不是换模型结构，而是一种管 attention keys and values 的算法——**PagedAttention**。立项时他们写：相对 HuggingFace Transformers，吞吐最高大约 **24×**；相对当时的 SOTA HuggingFace Text Generation Inference（TGI），最高大约 **3.5×**。

在 UC Berkeley 做出来。两个月里，已经在 [Chatbot Arena 和 Vicuna Demo](https://chat.lmsys.org) 上值班。LMSYS 那种算力并不宽裕的研究组，靠它才付得起「让几百万人排队说话」。当时劝人去 GitHub 用一条命令试试。

本地图（原文版权仍归原站；学习对照用）。分页比喻另有一张学习图。

## Beyond State-of-the-art Performance

对照：[HuggingFace Transformers (HF)](https://huggingface.co/docs/transformers/main_classes/text_generation)（当时最常用的 LLM 库）和 [HuggingFace Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)（当时的上一任 SOTA）。两档硬件：**LLaMA-7B on NVIDIA A10G**，**LLaMA-13B on NVIDIA A100（40GB）**。请求的进出长度从 ShareGPT 采样。实验里：相对 HF 最高 **24×**，相对 TGI 最高 **3.5×**。

![perf a100 n1 light](../../../../assets/vllm/blog/architecture/paged-attention/01-perf_a100_n1_light.png)
![perf a10g n1 light](../../../../assets/vllm/blog/architecture/paged-attention/02-perf_a10g_n1_light.png)

**图注（原文）。** 每个请求只要 **一份** completion。vLLM 相对 HF **14×–24×**，相对 TGI **2.2×–2.5×**。

![perf a100 n3 light](../../../../assets/vllm/blog/architecture/paged-attention/03-perf_a100_n3_light.png)
![perf a10g n3 light](../../../../assets/vllm/blog/architecture/paged-attention/04-perf_a10g_n3_light.png)

**图注（原文）。** 每个请求要 **三路并行** 输出。相对 HF **8.5×–15×**，相对 TGI **3.3×–3.5×**。并行采样更吃 KV；PagedAttention 正好会分享 prompt 的那几页。

## The Secret Sauce: PagedAttention

他们认定：LLM serving 的瓶颈是**记忆**，不是 FLOPs。自回归每吐一个字，输入 token 都要产出 attention 的 key / value 张量，留在 GPU 上好生成下一个 token。这份 KV cache：

- **大：** LLaMA-13B 一条序列可以到 **1.7GB**。
- **活：** 长度随序列走，高度可变、没法预先量体裁衣。

管这份 cache 很难。当时的系统因为碎片和超额预留，浪费 **60%–80%** 的显存。屋子看起来很大，真正能坐下的人很少。

**PagedAttention** 把操作系统的虚拟内存 / 分页搬进注意力。和传统 attention 不同：连续的 K/V 可以躺在**不连续**的物理空间里。每条序列的 KV 切成 **block**，每块固定数量 token 的 K/V。计算时，kernel 按表去找、去取这些块。

![annimation0](../../../../assets/vllm/blog/architecture/paged-attention/05-annimation0.gif)

**图注（原文）。** PagedAttention：KV cache 切成块。块在物理上不必连续。

块不必连续，就可以像 OS 管虚拟内存那样灵活：页 ≈ KV **block**，字节 ≈ token，进程 ≈ 一条序列。一条序列连续的 **logical blocks**，经 **block table** 映射到不连续的 **physical blocks**。物理块按需分配：新 token 长出来，再要一页。

学习对照（OS 比喻）：

![分页比喻](../../../../assets/vllm/blog/architecture/paged-attention/zh/01-os-metaphor.png)

| OS | PagedAttention |
|---|---|
| 页 | KV **block**（固定数量 token 的 K/V） |
| 字节 | token |
| 进程 | 一条序列 |
| 页表 | **block table**：逻辑块 → 物理块 |

![annimation1](../../../../assets/vllm/blog/architecture/paged-attention/06-annimation1.gif)

**图注（原文）。** 一条请求在 PagedAttention 下的生成过程。

浪费几乎只发生在**最后一块没填满**的地方，实践中不到 **4%**。显存一紧，就能多塞几条序列，GPU 才真正忙起来——上面那些倍数，多半是从这里来的。

另一件礼物：**共享**。并行采样时，好几路输出共用同一段 prompt。计算和显存都可以分。

![annimation2](../../../../assets/vllm/blog/architecture/paged-attention/07-annimation2.gif)

**图注（原文）。** 并行采样的例子。

block table 让不同序列指向同一物理块——像进程共享物理页。再加引用计数和 **Copy-on-Write**，共享才安全。

![annimation3](../../../../assets/vllm/blog/architecture/paged-attention/08-annimation3.gif)

**图注（原文）。** 一条请求采样多路输出时的生成过程。

复杂采样（并行、beam search）的显存开销可以砍到大约 **55%**，吞吐再涨到大约 **2.2×**。这些算法从「实验室里玩得起」变成「服务里用得起」。

PagedAttention 是 vLLM 的核心：多种模型、高性能、接口好用。当时技术细节指向 GitHub，论文还在路上。

## The Silent Hero Behind LMSYS Vicuna and Chatbot Arena

2023 年 4 月，[LMSYS](https://lmsys.org) 放出 Vicuna，公开可用。此后 Vicuna 在 [Chatbot Arena](https://arena.lmsys.org/) 上给几百万人值班。FastChat 最初用 HF Transformers 当 [serving backend](https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/model_worker.py)。流量翻了几倍，HF 成了门厅里的堵车。LMSYS 和 vLLM 很快接上 FastChat-vLLM，用 vLLM 当 [新后端](https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/vllm_worker.py)，好接住大约 **5×** 的高峰。LMSYS 早期 [内部微基准](https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/test_throughput.py)：相对最初的 HF 后端，吞吐可以到大约 **30×**。

4 月中起，Arena 上最热的 Vicuna、Koala、LLaMA，前端 FastChat（多模型 chat）、后端 vLLM。大学赞助的那几张卡，撑住了百万用户——高吞吐、低延迟。LMSYS 当时还在把 Databricks Dolly、LAION OpenAssistant、Stability AI StableLM 接进来；[更多模型支持](https://vllm.readthedocs.io/en/latest/models/supported_models.html) 写在「即将到来」。

![lmsys traffic](../../../../assets/vllm/blog/architecture/paged-attention/09-lmsys_traffic.png)

**图注（原文）。** 4–5 月 Chatbot Arena 上 FastChat-vLLM 接到的请求。一半以上走 vLLM。

GPU 用量砍掉大约 **50%**。日均约 **3 万** 请求，峰值 **6 万**。稳健，不只是实验室数字。

## Get started with vLLM

当时的安装（[installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html)）：

```bash
pip install vllm
```

离线和在线都能用。离线：Python 里导入 `LLM`：

```python
from vllm import LLM

prompts = ["Hello, my name is", "The capital of France is"]  # Sample prompts.
llm = LLM(model="lmsys/vicuna-7b-v1.3")  # Create an LLM.
outputs = llm.generate(prompts)  # Generate texts from the prompts.
```

在线：开一台 OpenAI API 兼容的 server（当时的入口；现在更常见的是 `vllm serve`）：

```bash
python -m vllm.entrypoints.openai.api_server --model lmsys/vicuna-7b-v1.3
```

查询格式与 OpenAI API 相同：

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "lmsys/vicuna-7b-v1.3",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

更多用法当时指向 [quickstart](https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html)。

## Credits

正文：Woosuk Kwon、Zhuohan Li（UC Berkeley）。Hao Zhang：vLLM 与 FastChat 的对接，以及这一节。致谢整支队伍——Siyuan Zhuang、Ying Sheng、Lianmin Zheng（UC Berkeley），Cody Yu（Independent Researcher），Joey Gonzalez（UC Berkeley），Hao Zhang（UC Berkeley & UCSD），Ion Stoica（UC Berkeley）。

下一篇必读是 [Anatomy](anatomy.md)：同一张页表，长成了一整座城。立项文只负责告诉你，城墙为什么要按页来砌。
