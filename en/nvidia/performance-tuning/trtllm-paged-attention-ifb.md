---
source: https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html
lang: en
fetched: 2026-08-31
---

# Paged Attention, IFB, and Request Scheduling

## In-flight batching (IFB)

Also called continuous / iteration-level batching. Context-phase and generation-phase sequences run in the **same** iteration so GPUs stay busy and queuing latency drops. Requires packed (unpadded) inputs. Current constraint: context sequences must appear **before** generation sequences in the packed tensor.

## Three size knobs

| Flag | Meaning |
|---|---|
| `max_batch_size` | Max requests the engine can handle. Build high enough not to bottleneck; tune runtime `max_batch_size` without rebuild. |
| `max_seq_len` | Max tokens of one request. Since v0.11 (with `--remove_input_padding` + `--context_fmha`) this replaces `max_input_len`/`max_output_len`. Default = `max_position_embeddings`. Only shrink if even one request at that length cannot fit. |
| `max_num_tokens` | Max **packed** tokens per batch. Default 8192 since v0.11. Only bites when padding is removed. Sets workspace + a GEMM dimension. Realistic (not max-prompt) values free KV memory. Go high enough for GPU math, not so high that TTFT/TPOT miss SLO. |

## Chunked context (chunked prefill)

Split a prompt across iterations so leftover token budget can mix with decode tokens. Needs **FMHA paged KV**. Chunk size (except the last) must be a multiple of the KV block size. Removes “prompt must fit in `max_num_tokens`” and usually **lowers average TTFT**; a few short prompts that would have finished in one shot may get slightly worse TTFT.

## KV cache

One cache per transformer layer.

- **Contiguous:** shape `[max_batch * max_beam, 2, num_heads, max_seqlen, hidden_per_head]` — wastes memory on short sequences.
- **Paged:** blocks handed out by a cache manager (`tensorrt_llm.runtime.KVCacheManager`; production path is C++ Batch Manager).

Scheduler visualization and the “always enable paged context attention” argument are the same as `trtllm-max-batch.md`.
