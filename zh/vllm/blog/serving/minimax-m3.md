---
source: https://vllm.ai/blog/2026-06-12-minimax-m3-vllm
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# MiniMax M3：1M 上下文靠 MSA 选 128-token 块，不是满 attention

英文对照：`en/vllm/blog/serving/minimax-m3.md`  
原文：https://vllm.ai/blog/2026-06-12-minimax-m3-vllm  
BF16 / MXFP8。验证过 H200 / GB200 / B300；AMD MI350/MI300。图在原网页。

MiniMax Sparse Attention：给 128-token KV 块打分，每 query / KV group 选 top，再 GQA。`--block-size 128` 必须对齐这块粒度。NVIDIA 默认 MSA backend，vision 走 `--mm-encoder-attn-backend FLASHINFER`、`--mm-processor-cache-type shm`、`--mm-encoder-tp-mode data`。AMD：`--attention-backend TRITON_ATTN`，vision `ROCM_AITER_FA`。MXFP8 MoE：Blackwell DeepGEMM，Hopper Marlin。parser：`--tool-call-parser minimax_m3` `--reasoning-parser minimax_m3`。Day-0 EAGLE3：`Inferact/MiniMax-M3-EAGLE3`。NeMo RL 里 GRPO 用 vLLM 做 generation。菜谱在 Recipes，不在这篇抄全 CLI。
