---
source: https://docs.vllm.ai/en/stable/features/speculative_decoding/
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# Speculative decoding（功能页）

英文对照：[en/vllm/features/speculative-decoding.md](../../../en/vllm/features/speculative-decoding.md)  
原文：https://docs.vllm.ai/en/stable/features/speculative_decoding/  
原理与 2024 年那组数字：[博客 spec-decode](../blog/performance/spec-decode.md)。文中「还不支持」是历史。

每步先让一个更便宜的提议者猜几个 token，目标模型一次核对。猜对了就少付几次完整 forward。CLI 是一份 JSON：

```bash
vllm serve <target> --speculative-config '{
  "method": "draft_model",
  "model": "<draft>",
  "num_speculative_tokens": 5
}'
```

Python：`speculative_config={...}`。

常用字段：

- `method`：`draft_model` / `ngram` / `suffix` / `mtp` / `eagle3` / `dflash`
- `model`：draft 或 EAGLE head
- `num_speculative_tokens`：每步提议几个
- `parallel_drafting`：仅 EAGLE / draft-model
- `use_heterogeneous_vocab`：仅 draft_model，取词表交集

方法名会继续变（P-EAGLE、DSpark、EAGLE 3.1 各有一篇 CATALOG 里的后续文）。这份功能页只负责「旗标长什么样」；要理解 scheduler 如何把提议嵌进 continuous batching，读 2024-10-17 那篇。格式约束（JSON / 工具参数）是另一间房：[structured decoding](../blog/performance/struct-decode.md)。
