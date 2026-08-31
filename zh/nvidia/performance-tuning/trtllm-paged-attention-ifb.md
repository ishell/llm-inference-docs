---
source: https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html
lang: zh
fetched: 2026-08-31
---

# Paged Attention、IFB 与请求调度

## In-flight batching（IFB）

也叫 continuous / iteration-level batching。Context 阶段和 generation 阶段的序列在**同一次** iteration 里跑，GPU 更满、排队更短。要求 packed（去 padding）输入。当前实现：packed tensor 里 **context 序列必须排在 generation 前面**。

## 三个尺寸旋钮

| 旗标 | 含义 |
|---|---|
| `max_batch_size` | 引擎最多同时处理多少请求。编译时设够高以免成为瓶颈；运行时还能再调，不用重建。 |
| `max_seq_len` | 单请求最长。v0.11 起（`--remove_input_padding` + `--context_fmha`）可代替 `max_input_len`/`max_output_len`。默认 `max_position_embeddings`。只有连一条这么长的请求都塞不下时才往下砍。 |
| `max_num_tokens` | 每 batch **打包后**最多多少 token。v0.11 默认 8192。只有去 padding 才生效。决定 workspace 和某一维 GEMM。用接近真实负载而不是「最长 prompt」的值，KV 才能多占显存。够高吃满算力，但别高到 TTFT/TPOT 破 SLO。 |

## Chunked context（chunked prefill）

把 prompt 拆到多次 iteration，剩余 token 预算可以和 decode 混打。需要 **FMHA paged KV**。除最后一块外，chunk 大小必须是 KV block 的整数倍。不再要求 prompt ≤ `max_num_tokens`。平均 TTFT 通常更好；少数「本来一次就能做完」的短 prompt 可能略差。

## KV cache

每层一份。

- **Contiguous：** `[max_batch * max_beam, 2, num_heads, max_seqlen, hidden_per_head]` —— 短序列浪费显存。
- **Paged：** cache manager 按块分配（Python 示意 `KVCacheManager`；生产是 C++ Batch Manager）。

调度示意图和「务必开 paged context attention」见 `trtllm-max-batch.md`。
