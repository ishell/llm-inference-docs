---
source: https://vllm.ai/blog/2025-08-05-gpt-oss
lang: en
fetched: 2026-09-01
---

# gpt-oss day-0: MXFP4 MoE + 1:1 full/sliding attention + hybrid KV

Chinese: [zh/vllm/blog/serving/gpt-oss.md](../../../../zh/vllm/blog/serving/gpt-oss.md)  
20B / 120B. Blackwell, Hopper, MI300x/MI355x. Then `vllm==0.10.1+gptoss` or `vllm/vllm-openai:gptoss`. Later Pareto: [gpt-oss-optimizations](../performance/gpt-oss-optimizations.md).

120B: 128 experts, 4 per token, no shared; MXFP4 ~**63 GB**. 20B ~14 GB. Blackwell: FlashInfer native MXFP4 Tensor Cores; Hopper: OpenAI Triton `matmul_ogs`. Attention: GQA 64/8, head dim **64**, full vs window=128 at 1:1, attention sink per query head. Hybrid KV allocator shares physical pages across layer types. Built-in browse/Python: Responses API or external MCP, not a generic `/chat/completions` tool parser.
