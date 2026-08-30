---
source: https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/
lang: en
fetched: 2026-08-30
---

# Automatic Prefix Caching

Automatic Prefix Caching (APC) caches the KV cache of existing queries. A new query can reuse KV cache if it shares a prefix with an existing query, skipping computation of the shared part.

Technical details: https://docs.vllm.ai/en/stable/design/prefix_caching/

## Enabling APC

Set `enable_prefix_caching=True` in the vLLM engine.

## Example workloads

APC helps a lot when:

- **Long document query:** the same long document (manual, annual report) is queried repeatedly with different questions. Process the document once; later requests reuse its KV cache.
- **Multi-round conversation:** reuse chat-history KV across turns instead of recomputing the whole history.

## Limits

APC generally does not hurt performance. It only speeds up **prefill**, not **decode**. It does not help when:

- most time is spent generating long answers, or
- new queries do not share a prefix with any cached query.
