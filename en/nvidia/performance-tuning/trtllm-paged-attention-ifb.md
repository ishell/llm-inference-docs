---
source: https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html
lang: en
fetched: 2026-08-31
---

# Paged Attention, IFB, and Request Scheduling

Same machinery as handbook chapter 3 (`trtllm-max-batch.md`), written as a feature page. Figures stay on the official HTML.

## In-flight batching (IFB)

Also continuous / iteration-level batching. Context-phase and generation-phase sequences run in the **same** iteration. Requires packed (unpadded) inputs — padding generation’s single token out to max prompt length is waste.

Current constraint: in the packed tensor, **context sequences must appear before generation sequences**. May be relaxed later.

## Three size knobs

| Flag | Meaning |
|---|---|
| `max_batch_size` | Max concurrent requests. Build high enough not to bottleneck; tune runtime `max_batch_size` without rebuild. |
| `max_seq_len` | Max tokens of one request. Since v0.11 (`--remove_input_padding` + `--context_fmha`) replaces `max_input_len`/`max_output_len`. Default `max_position_embeddings`. Shrink only if even one request at that length cannot fit. |
| `max_num_tokens` | Max **packed** tokens per iteration. Default **8192** since v0.11. No effect if padding is kept. Sets workspace and a GEMM dimension. |

Do not size `max_num_tokens` to the longest prompt: real prompts are shorter, and IFB generation contributes at most `beam_width` tokens per step. A realistic value leaves memory for KV. Raise it for GPU math; past saturation, TTFT and e2e latency suffer. Meet SLO (TTFT / TPOT), then stop.

Sweep recipe: `trtllm-max-batch.md`.

## Chunked context (chunked prefill)

Split the prompt across iterations so leftover token budget mixes with decode. Needs **FMHA paged KV**. Except the last chunk, size must be a multiple of the KV block size. Prompt need not fit in `max_num_tokens`. Average TTFT usually drops; a few short “lucky” prompts that would have finished in one shot may get slightly worse TTFT.

## KV: contiguous vs paged

One cache per transformer layer.

- **Contiguous:** `[max_batch_size * max_beam_width, 2, num_heads, max_seqlen, hidden_dim_per_head]` — short sequences pay for the full seat.
- **Paged:** a cache manager hands out blocks (`tensorrt_llm.runtime.KVCacheManager`; production is C++ Batch Manager). See `trtllm-kvcache.md`.
