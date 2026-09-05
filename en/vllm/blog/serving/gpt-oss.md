---
source: https://vllm.ai/blog/2025-08-05-gpt-oss
lang: en
fetched: 2026-09-04
---

# gpt-oss day-0: MXFP4 MoE + 1:1 full/sliding attention + hybrid KV

Chinese: [zh/vllm/blog/serving/gpt-oss.md](../../../../zh/vllm/blog/serving/gpt-oss.md)

2025-08-05. **The vLLM Team**. 20B / 120B. Blackwell, Hopper, MI300x / MI355x. Then `vllm==0.10.1+gptoss` or `vllm/vllm-openai:gptoss`. Later Pareto: [gpt-oss-optimizations.md](../performance/gpt-oss-optimizations.md). Hybrid KV paper same as [qwen3-next.md](qwen3-next.md). Multi-LoRA on this family: [multi-lora.md](multi-lora.md). Built-in browse/Python is Responses API or external MCP — **not** a generic `/chat/completions` tool parser.

**TL;DR from the page:**

- Sparse MoE: 120B = 128 experts, 20B = 32; **4** per token, **no** shared expert. MoE weights MXFP4; attention and other layers BF16.
- MXFP4 shrinks to ~**63 GB** (120B) / ~**14 GB** (20B) — runnable on one GPU, “often not recommended” for best performance.
- Attention: GQA 64/8, head dim **64**, full vs window=128 at **1:1**, attention sink per query head.
- Hybrid KV allocator shares physical pages across full and sliding-window layers.

## Quick start

Container:

```
docker run --gpus all \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:gptoss \
    --model openai/gpt-oss-20b
```

Or the then-wheel:

```
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

vllm serve openai/gpt-oss-120b
```

User guide: [GPT-OSS recipe](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html).

## MXFP4 MoE

[MXFP4](https://arxiv.org/abs/2310.10537): group-quantized float. Each weight is fp4 e2m1; a power-of-two scale per group of **32** consecutive fp4 values. Two fp4 packed into one 8-bit unit; unpacked on the fly inside the matmul.

Two kernels, with OpenAI and NVIDIA:

- **Blackwell (e.g. B200):** FlashInfer MoE, NVIDIA, native MXFP4 Tensor Cores.
- **Hopper (e.g. H100, H200):** OpenAI Triton [`matmul_ogs`](https://github.com/triton-lang/triton/tree/main/python/triton_kernels) — swizzling + built-in heuristics, no manual tuning.

## Efficient attention

GQA: 64 query heads, 8 KV heads. Full attention interleaved with sliding-window (**128**) at 1:1. Head size **64** (half of the usual 128). Each query head has a trained attention-sink vector.

Kernels: FlashInfer (Blackwell), FlashAttention 3 (Hopper), enhanced Triton on AMD. Hybrid KV allocator ([paper](https://arxiv.org/abs/2503.18292)): full and sliding-window layers share physical pages; fragmentation claimed down to zero.

## Built-in tool support: agent loop and tool server via MCP

Built-in web browse and Python interpreter. Model decides when to call; vLLM parses, invokes, feeds results back.

Native path: [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) + gpt-oss toolkit — parse tool call, run search / code interpreter, parse outputs, send back.

Alternative: MCP-compliant external tool server; vLLM talks to the server instead of the toolkit. No internal vLLM change for new tool libraries.

## Looking ahead

Then-roadmap: harden Responses API; attention DP and MoE EP; cut CPU overhead for throughput.

## Acknowledgement

vLLM: Yongye Zhu, Woosuk Kwon, Chen Zhang, Simon Mo, Kaichao You.

Jay Shah (Colfax International): attention-sink adaptation; FA3 optimizations for gpt-oss.

OpenAI: Zhuohan Li, Xiaoxuan Liu, Philippe Tillet, Mario Lezcano-Casado, Dominik Kundel, Casey Dvorak, Vol Kyrylov.

NVIDIA (Blackwell perf + accuracy): Duncan Moss, Grace Ho, Julien Demouth, Minseok Lee, Siyuan Fu, Zihao Ye, Pen Chung Li.

AMD: Hongxia Yang, Ali Zaidy; support from Peng Sun, Vinayak Gokhale, Andy Luo.

Hugging Face: Lysandre, Hugo, Marc, vb, Arthur, Mohamed, Andrien.

Partners named: AWS, Cloudflare, Snowflake, Databricks, Together, Fireworks, Cerebras.
