---
source: https://vllm.ai/blog/2026-03-13-p-eagle
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# P-EAGLE：一次前向猜 K 个字

英文对照：`en/vllm/blog/performance/p-eagle.md`  
原文：https://vllm.ai/blog/2026-03-13-p-eagle  
vLLM ≥0.16.0，PR#32887。数字是 **一块 B200**、GPT-OSS-20B 上的演示。图在原网页。

EAGLE 草稿是自回归的：猜 K 个 token 要 K 次 draft 前向。草稿越好、K 越大，这段税越贵。P-EAGLE 一次前向吐出全部 K 个。相对公开 EAGLE-3 checkpoint，低并发约 **1.55–1.69×** TPS，c=64 仍约 **1.05–1.25×**。峰值常在 K=7；EAGLE-3 峰值常在 K=3——并行草稿的深度几乎免费，线性草稿不是。

## 结构

Prefill 与普通 EAGLE 一样：target 走完 prompt，留下 `h_prompt` / `h_context`。Drafter 把每个位置的 embedding 和 hidden 拼在一起一次过 N 层：

- 位置 1（NTP）：新 token + `h_context`，和自回归 EAGLE 相同。
- 位置 2…K（MTP）：还不存在的 token/hidden 用训练出来的 **mask embedding** 和 **共享 hidden** 填。

验收长度（AL）也更高：K=7 时 HumanEval 3.94 vs EAGLE-3 的 3.03。同一场前向里猜得更深，被接受的也更多。

## 引擎里别扭的地方

并行草稿的 batch 形状和 verification 不再一样：要插入 MASK 位，重建 slot map。他们用一只融合 Triton kernel 在 GPU 上展开 token / position / mask 元数据；hidden 更大，另开 copy kernel 把学到的 placeholder 广播进 mask 槽。被拒绝的 token 写 `PADDING_SLOT_ID (-1)`，免得脏 KV。CUDA graph capture 范围要加 `K × max_num_seqs`。

## 怎么开

HF 上有 GPT-OSS 20B/120B、Qwen3-Coder 30B 的预训练头。`"parallel_drafting": true`。当时 GPT-OSS-20B + EAGLE 还要一笔补丁 PR#36684。

```bash
vllm serve openai/gpt-oss-20b \
  --speculative-config '{"method":"eagle3","model":"amazon/gpt-oss-20b-p-eagle","num_speculative_tokens":5,"parallel_drafting":true}'
```

训练长序列时 N×K 个位置会把 attention 撑爆；他们用序列内切块 + 跨块正确 attention 依赖（论文）。和 [投机解码主线](spec-decode.md)、[并行草稿总览](parallel-drafting.md) 一起读。
