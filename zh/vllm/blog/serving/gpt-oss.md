---
source: https://vllm.ai/blog/2025-08-05-gpt-oss
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# gpt-oss Day-0：MXFP4 MoE + 满/滑窗 1:1 attention + hybrid KV

英文对照：[en/vllm/blog/serving/gpt-oss.md](../../../../en/vllm/blog/serving/gpt-oss.md)  
原文：https://vllm.ai/blog/2025-08-05-gpt-oss  
2025-08-05。署名 **The vLLM Team**。20B / 120B。Blackwell、Hopper、MI300x / MI355x。当时 `vllm==0.10.1+gptoss` 或 `vllm/vllm-openai:gptoss`。Pareto 后续见 [gpt-oss-optimizations.md](../performance/gpt-oss-optimizations.md)。Hybrid KV 论文和 [qwen3-next.md](qwen3-next.md) 同一篇。这家人上 Multi-LoRA：[multi-lora.md](multi-lora.md)。内置浏览/Python：Responses API 或外部 MCP——**不是**普通 `/chat/completions` tool parser。

**原文 TL;DR：**

- 稀疏 MoE：120B = 128 expert，20B = 32；每 token **4** 只，**没有** shared expert。MoE 权重 MXFP4；attention 和其他层 BF16。
- MXFP4 压到约 **63 GB**（120B）/ 约 **14 GB**（20B）——单卡能跑，原文说「often not recommended」冲性能。
- Attention：GQA 64/8，head dim **64**，满 attention 与 window=128 **1:1**，每 query head 有 attention sink。
- Hybrid KV allocator 让满层和滑窗层共享物理页。

## Quick start

容器：

```
docker run --gpus all \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:gptoss \
    --model openai/gpt-oss-20b
```

或当时的 wheel：

```
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

vllm serve openai/gpt-oss-120b
```

用户指南：[GPT-OSS recipe](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)。

## MXFP4 MoE

[MXFP4](https://arxiv.org/abs/2310.10537)：分组量化浮点。每个权重是 fp4 e2m1；每组 **32** 个连续 fp4 一个 2 的幂 scale。两个 fp4 打进一个 8-bit；matmul 里现场拆。

两套 kernel，和 OpenAI、NVIDIA 一起：

- **Blackwell（如 B200）：** FlashInfer MoE，NVIDIA，原生 MXFP4 Tensor Core。
- **Hopper（如 H100、H200）：** OpenAI Triton [`matmul_ogs`](https://github.com/triton-lang/triton/tree/main/python/triton_kernels)——swizzling + 内置启发式，不用手调。

## Efficient attention

GQA：64 query head，8 KV head。满 attention 和滑窗（**128**）1:1 交错。Head size **64**（常規 128 的一半）。每个 query head 有训练好的 attention-sink 向量。

Kernel：FlashInfer（Blackwell）、FlashAttention 3（Hopper）、AMD 上增强过的 Triton。Hybrid KV allocator（[论文](https://arxiv.org/abs/2503.18292)）：满层和滑窗层共享物理页；碎片声称压到零。

## 内置工具：agent loop 和 MCP tool server

内置网页浏览和 Python 解释器。模型自己决定何时调；vLLM 解析、调用、把结果喂回去。

原生路径：[OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) + gpt-oss toolkit——解析 tool call，跑搜索 / 代码解释器，解析输出，送回模型。

另一条：MCP 合规的外部 tool server；vLLM 跟 server 说话，不直接用 toolkit。新工具库不必改 vLLM 内核。

## Looking ahead

当时 roadmap：把 Responses API 拧硬；attention DP 和 MoE EP；砍 CPU 开销换吞吐。

## Acknowledgement

vLLM：Yongye Zhu, Woosuk Kwon, Chen Zhang, Simon Mo, Kaichao You。

Jay Shah（Colfax International）：attention-sink 适配；gpt-oss 上的 FA3 优化。

OpenAI：Zhuohan Li, Xiaoxuan Liu, Philippe Tillet, Mario Lezcano-Casado, Dominik Kundel, Casey Dvorak, Vol Kyrylov。

NVIDIA（Blackwell 性能 + 精度）：Duncan Moss, Grace Ho, Julien Demouth, Minseok Lee, Siyuan Fu, Zihao Ye, Pen Chung Li。

AMD：Hongxia Yang, Ali Zaidy；Peng Sun, Vinayak Gokhale, Andy Luo 支援。

Hugging Face：Lysandre, Hugo, Marc, vb, Arthur, Mohamed, Andrien。

点名的伙伴：AWS、Cloudflare、Snowflake、Databricks、Together、Fireworks、Cerebras。
