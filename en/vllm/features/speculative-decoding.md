---
source: https://docs.vllm.ai/en/stable/features/speculative_decoding/
lang: en
fetched: 2026-08-31
---

# vLLM Speculative Decoding

CLI: `--speculative-config` JSON (same keys on `LLM(..., speculative_config={...})`).

Common keys:

- `method`: `draft_model` | `ngram` | `suffix` | `mtp` | `eagle3` | `dflash` (often inferred)
- `model`: draft / EAGLE head
- `num_speculative_tokens`: proposals per step
- `parallel_drafting`: EAGLE / draft-model only
- `use_heterogeneous_vocab`: draft_model only; intersect vocabs

```bash
vllm serve <target> --speculative-config '{
  "method": "draft_model",
  "model": "<draft>",
  "num_speculative_tokens": 5
}'
```

See also vLLM blog: How Speculative Decoding Boosts vLLM (up to ~2.8×) — listed in `blog/CATALOG.md`.
