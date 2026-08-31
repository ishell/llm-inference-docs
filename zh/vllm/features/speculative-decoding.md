---
source: https://docs.vllm.ai/en/stable/features/speculative_decoding/
lang: zh
fetched: 2026-08-31
---

# vLLM Speculative Decoding

`--speculative-config` JSON，或 Python `speculative_config={...}`。

常用字段：

- `method`：`draft_model` / `ngram` / `suffix` / `mtp` / `eagle3` / `dflash`
- `model`：draft 或 EAGLE head
- `num_speculative_tokens`：每步提议几个 token
- `parallel_drafting`：仅 EAGLE / draft-model
- `use_heterogeneous_vocab`：仅 draft_model，取词表交集

```bash
vllm serve <target> --speculative-config '{
  "method": "draft_model",
  "model": "<draft>",
  "num_speculative_tokens": 5
}'
```

博客「最高约 2.8×」见 `blog/CATALOG.md` 里 2024-10-17 那篇。
