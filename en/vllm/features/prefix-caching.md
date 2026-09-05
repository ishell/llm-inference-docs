---
source: https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/
lang: en
fetched: 2026-09-04
---

# Automatic Prefix Caching

Chinese: [zh/vllm/features/prefix-caching.md](../../../zh/vllm/features/prefix-caching.md)  
Technical details: [prefix-caching-design.md](prefix-caching-design.md) (https://docs.vllm.ai/en/stable/design/prefix_caching/)

Automatic Prefix Caching (APC) caches the KV cache of existing queries. A new query can reuse KV cache if it shares a prefix with an existing query, skipping computation of the shared part.

## Enabling APC

Set `enable_prefix_caching=True` in the vLLM engine, or `--enable-prefix-caching` on `vllm serve`. Hash default `sha256` (`--prefix-caching-hash-algo`). Cross-environment reproducibility: `sha256_cbor`. `xxhash` / `xxhash_cbor` are faster with weaker collision isolation — multi-tenant readers should see the security note on the [design page](prefix-caching-design.md).

## Example workloads

APC helps a lot when:

- **Long document query:** the same long document (manual, annual report) is queried repeatedly with different questions. Process the document once; later requests reuse its KV cache.
- **Multi-round conversation:** reuse chat-history KV across turns instead of recomputing the whole history.

## Limits

APC generally does not hurt performance. It only speeds up **prefill**, not **decode**. It does not help when:

- most time is spent generating long answers, or
- new queries do not share a prefix with any cached query.
