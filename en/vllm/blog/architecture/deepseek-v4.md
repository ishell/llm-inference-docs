---
source: https://vllm.ai/blog/2026-04-24-deepseek-v4
lang: en
fetched: 2026-09-05
---

# DeepSeek V4 in vLLM: Efficient Long-context Attention

Chinese: [zh/vllm/blog/architecture/deepseek-v4.md](../../../../zh/vllm/blog/architecture/deepseek-v4.md)  
Source: https://vllm.ai/blog/2026-04-24-deepseek-v4

2026-04-24. **vLLM Team**. Study rewrite, not an official reprint. Hugging Face: [`deepseek-ai/DeepSeek-V4-Pro`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) (1.6T) and [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (285B). Both claim up to **one million** tokens of context. Image in the recipes: `vllm/vllm-openai:deepseekv4-cu130`. First support drop; further opts were still in flight on the page.

Earlier sparse-attention stack: [deepseek-v32](deepseek-v32.md). FP8 KV / attention: [fp8-kvcache](../performance/fp8-kvcache.md). Wide-EP serving: [large-scale](../serving/large-scale.md). GB200 scoreboard: [gb200-wideep](../serving/gb200-wideep.md). Plugin door: [plugin-system](plugin-system.md) / [hardware-plugin](hardware-plugin.md).

**TL;DR from the page:**

- Same 1M-context attention implementation. Optional: FP4 indexer, MTP. `--block-size 256` is the allocator’s logical unit.
- Four structural moves: share K and V (~**2×**, correctness needs **inverse RoPE** on the attention output); compress KV across tokens (`c4a` / `c128a`, ~**4× to 128×**); DSA attends only to top-$k$ compressed tokens; short sliding window **128** on uncompressed tokens.
- With bf16 KV at 1M context: V4 is **9.62 GiB** per sequence vs **83.9 GiB** for a 61-layer V3.2-style stack — about **8.7×**. Production uses **fp4** indexer and **fp8** attention cache, another ~**2×** off the bf16 estimate. Arithmetic in the appendix.
- vLLM: logical block = 256 native positions; compressor residual as sliding-window KV; five cache kinds into three page-size buckets; kernel fusion + multi-stream. Implementation [vllm#40760](https://github.com/vllm-project/vllm/pull/40760).

Original sections: Running DeepSeek V4 on vLLM (DeepSeek-V4-Pro / DeepSeek-V4-Flash) → DeepSeek V4's Attention Mechanism Explained → vLLM's Implementation of DeepSeek V4 (Keeping the KV Cache Memory Packed: (1) A single logical block size / (2) Compressor state as a sliding window / (3) Unifying page sizes; Keeping the GPU Busy: (1) Kernel Fusion / (2) Multi-stream) → Planned Work → Acknowledgments → Appendix: The Math behind DeepSeek V4's Attention Mechanism (inverse RoPE / position ranges / $k$ in c4a and c128a / short sliding window / 8.7× savings arithmetic).

The new attention looks intricate on first reading; the principles are straightforward once examined systematically. Three blocks: a quickstart; a first-principles walk through the architecture; then the vLLM systems work — hybrid KV cache, kernel fusion, disaggregated serving. The write-up is meant to explain both the mechanism and why the implementation choices look the way they do.

## Running DeepSeek V4 on vLLM

Two models, same attention implementation sized for 1M context. Optional extras named on the page: FP4 indexer and MTP. The docker snippets are for easy single-node testing, not a cluster recipe. Disaggregated serving and other GPU layouts live on the [V4-Pro recipes](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro) and [V4-Flash recipes](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash).

### DeepSeek-V4-Pro

Runnable on **8×B200** or **8×B300**:

```bash
docker run --gpus all \
  --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:deepseekv4-cu130 deepseek-ai/DeepSeek-V4-Pro \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --data-parallel-size 8 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops":["all"]}' \
  --attention_config.use_fp4_indexer_cache=True \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4
```

### DeepSeek-V4-Flash

Same flags, smaller DP. Runnable on **4×B200** or **4×B300**:

```bash
docker run --gpus all \
  --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:deepseekv4-cu130 deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --data-parallel-size 4 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops":["all"]}' \
  --attention_config.use_fp4_indexer_cache=True \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4
```

`--block-size 256` is the logical unit in the allocator section below, not an arbitrary page guess.

## DeepSeek V4's Attention Mechanism Explained

Long-context inference has two usual walls:

- **KV cache memory growth.** Cache still scales linearly with context. [MLA](https://arxiv.org/abs/2405.04434) is already much cheaper than MHA / MQA; one million tokens still does not fit comfortably in GPU memory.
- **Attention computation cost.** Even with [DSA](http://arxiv.org/abs/2512.02556), the matmul remains a bottleneck.

The V4 design compresses the cache **and** the compute:

1. **Share key and value vectors** (~**2×** memory). Correctness then needs an **inverse RoPE** on the attention output — full algebra in the appendix.
2. **Compress the KV cache across multiple tokens** (~**4× to 128×**). Two recipes:
   - **`c4a`:** compress by roughly 1/4. One compressed token is a weighted sum of **8** uncompressed tokens, **stride 4**.
   - **`c128a`:** compress by roughly 1/128. One compressed token is a weighted sum of **128** uncompressed tokens, **stride 128**.
3. **DSA, bounded compute.** After `c4a`, a 1M-token sequence still has **250k** compressed tokens. DSA attends only to the top-$k$ compressed tokens.
4. **Preserving locality: short sliding window.** Window size **128**, on **uncompressed** tokens, so a query can still see local context before it crosses a compression boundary.

The original page animates `c4a` on **13** tokens (and ships an [interactive hover view](https://vllm.ai/assets/interactive_pages/c4a.html); this note keeps the static gif, no HTML widget). `c128a` is the same diagram with a coarser stride.

![c4a animation](../../../../assets/vllm/blog/architecture/deepseek-v4/01-c4a_animation.gif)

**Figure.** Animation of `c4a` attention (study copy; copyright remains with the original site).

With **bf16** KV, DeepSeek V4 is **9.62 GiB** per sequence at 1M context — about **8.7×** smaller than the **83.9 GiB** estimate for a **61-layer** DeepSeek V3.2-style stack. Production on the page uses **fp4** for the indexer cache and **fp8** for the attention cache, which cuts that bf16 estimate by roughly another **2×**. Arithmetic and interpretation are in the appendix.

![kv cache comparison](../../../../assets/vllm/blog/architecture/deepseek-v4/02-kv-cache-comparison.svg)

**Figure.** Per-layer KV state: DeepSeek V3.2 versus DeepSeek V4.

## vLLM's Implementation of DeepSeek V4

The structural savings are real; turning them into a serving path is still a systems problem:

- Same split as V3.2: the attention kernel uses **bfloat16** KV for Prefill and **partially token-wise fp8** for Decode.
- Mix of `c4a` and `c128a`, plus some layers that are **pure sliding window** with no compression. Heterogeneous types make KV management much harder.
- A batched set of sequences can sit at **different compression-boundary states**.
- Native **fp4 MoE** weights need special handling.

Architecture extras named and then skipped: [Manifold-Constrained Hyper-Connections](http://arxiv.org/abs/2512.24880), and some MoE-module deltas. The post calls those simpler model changes.

vLLM’s answer is two fronts: memory packing, then kernel efficiency.

### Keeping the KV Cache Memory Packed

The allocator still has to work with prefix caching, Prefill/Decode disaggregation, CUDA graphs, and the rest of the serving path. Three choices.

#### (1) A single logical block size

Layers compress at 1/4 (`c4a`), 1/128 (`c128a`), or 1/1 (SWA). Sizing each layer’s block around a round number of *compressed* entries would give every layer its own page layout.

Instead the logical block is fixed at **256 native token positions** for every compressed layer. A `c4a` block then physically holds `256 / 4 = 64` compressed entries; a `c128a` block holds `256 / 128 = 2`. Allocating a block always means the next 256 native positions of that request, whoever owns the layer. Slot mapping, scheduler accounting, and prefix-hit detection share that unit and do not branch on `compress_ratio`.

#### (2) Compressor state as a sliding window

Each compressor layer keeps a small rolling residual per request: an **8-token** (overlapped) partial for C4, a **128-token** partial for C128. A per-request side buffer works in isolation and then fights the rest of the stack.

With a side buffer, prefix cache would have to snapshot the residual at every cacheable boundary, key it with the prefix hash, and restore it on a hit. Disaggregated Prefill would need a second transfer path that ships residuals from Prefill workers to Decode workers alongside the KV blocks. Each requirement is manageable; together they create another state-management path to maintain across features.

vLLM treats compressor state like sliding-window KV. Same invariant: fixed size per request, advanced as Decode proceeds, state outside the window discarded or handled through caching. It is registered under the sliding-window KV cache spec with `sliding_window = coff * compress_ratio` (**8** for C4, **128** for C128), in SWA-style blocks under the same hybrid KV manager.

Reuse:

- **Prefix caching.** A hit lands on a KV block boundary (the 256-position unit). Compressor state at that boundary is already the handoff.
- **Disaggregated Prefill.** Compressor state ships like SWA state: only blocks inside the window, no residual-specific path, transfer-size savings kept.
- **CUDA graphs** and **MTP** follow the same SWA integration pattern; metadata stays compressor-specific.

#### (3) Unifying page sizes

The two choices above are still not enough. A C4 indexer block, a `c128a` KV block, and a `c4a` compressor-state block still have different **page sizes** (bytes per block). Separate pools would fragment each other.

Page size is `block_size * compress_ratio * per_entry_size`, and all three factors are chosen so the five-way cache stack collapses into **three** page-size buckets. Each pool is sized once at load time; allocation is a bucket lookup. No runtime repartitioning, no per-kind accounting, no cross-kind fragmentation.

- *Largest bucket:* `c4a` main KV, SWA KV, `c4a` compressor state, `c128a` compressor state.
- *Middle bucket:* C4 indexer KV, C4 indexer compressor state.
- *Smallest bucket:* `c128a` main KV.

A commented details block in the source sketches concrete bytes for a **61-layer** V4 with the standard C4/C128 mix: **1,728 B**, **8,640 B**, and **37,440 B** per block, each a multiple of FlashMLA’s **576 B** alignment. The public post does not finish the per-kind clustering table.

### Keeping the GPU Busy

FlashMLA and FlashInfer cover attention and MoE. This model still launches many small, mostly memory-bound kernels. Extra launches and HBM round-trips would stall Decode.

![decode path](../../../../assets/vllm/blog/architecture/deepseek-v4/03-decode-path.svg)

**Figure.** `c4a` Decode path: operator graph with kernel fusions (colored outlines) and multi-stream partitioning (default stream = blue band, indexer stream = amber band).

#### (1) Kernel Fusion

Three fusions, the colored outlines in the figure:

- **Compressor + RMSNorm + RoPE + cache insertion.** After compression, compressed K goes through RMSNorm, RoPE, and insertion into the next attention’s KV cache (main or indexer). Almost entirely elementwise → one kernel. Separate kernels for indexer K cache vs main-attention K cache so parallelization can still match each head dim. About **~1.4–3×** over the unfused baseline.
- **Inverse RoPE + fp8 quant.** After main attention, inverse RoPE then fp8 batched matmul for the `o_lora` projection. Fusing them skips a back-to-back HBM round trip and raises arithmetic intensity: **~2–3×** over unfused.
- **Fused Q norm + KV RoPE + K insert.** Before main attention, KV insertion for both the compressed path and the sliding-window path. Compressed path is already the first fusion; what remains is elementwise work on queries and uncompressed SWA keys. Horizontal fuse with static `warpID` dispatch: each warp independently on a Q head or a K head, no cross-warp communication. **10–20×** over the naive unfused kernels.

They also reuse V3.2 fusions: Q RoPE + quant + weight multiply, and horizontal QK norm right after QK projection at the start of attention.

#### (2) Multi-stream

Work before main attention splits three ways: indexer, main-attention KV compression, SWA token insertion. After the initial projection those branches are almost independent, so they overlap on CUDA streams. Blue band = default stream; amber = indexer stream.

- **`c128a` layers** (no indexer): main KV compression in parallel with SWA token insertion.
- **`c4a` layers:** full indexer pipeline on its own stream, in parallel with main KV compression and SWA insertion (those last two stay serial with each other).

Observed: **5–6%** end-to-end latency reduction at **low batch sizes** — less idle Decode. CUDA graphs still cut launch overhead on Decode, as for every other model.

Implementation PR: [vllm#40760](https://github.com/vllm-project/vllm/pull/40760).

## Planned Work

In flight on the page:

- DeepGEMM MegaMoE kernel
- Paged Prefill kernel

The drop targets NVIDIA **Hopper** and **Blackwell**; recipes on the [recipe site](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro). Hardware vendors can add support through the plugin door. Named independently: [vllm-ascend](https://github.com/vllm-project/vllm-ascend), [vllm-mlu](https://github.com/Cambricon/vllm-mlu).

## Acknowledgments

DeepSeek for open-sourcing V4, and DeepSeek leadership for trust in vLLM. Model support credited to [Inferact Inc.](https://inferact.ai/).

## Appendix: The Math behind DeepSeek V4's Attention Mechanism

### Why inverse RoPE is needed when key and value are shared

Query at position $i$ after [RoPE](http://arxiv.org/abs/2104.09864): $<q_i, i> = R(i)q_i$, with $R(i)$ the rotation matrix whose angles are parameterized by position $i$. Basic properties:

- $R(i)R(j) = R(i+j)$
- $R(i)^{-1} = R(i)^T = R(-i)$
- $R(i)$ is orthogonal: $R(i)R(i)^T = I$

Keys at $j_1, j_2, j_p, \ldots, j_n$ after RoPE: $<k_{j_1}, j_1> = R(j_1)k_{j_1}$, $<k_{j_2}, j_2> = R(j_2)k_{j_2}$, $\ldots$, $<k_{j_p}, j_p> = R(j_p)k_{j_p}$, $\ldots$, $<k_{j_n}, j_n> = R(j_n)k_{j_n}$.

Values usually do **not** get RoPE: $<v_{j_1}, j_1> = v_{j_1}$, $\ldots$, $<v_{j_p}, j_p> = v_{j_p}$, $\ldots$, $<v_{j_n}, j_n> = v_{j_n}$.

Attention output (scaling omitted):

$$
a_i = \sum_{p=1}^n \frac{\exp(<q_i, i>^T <k_{j_p}, j_p>)}{\sum_{r=1}^n \exp(<q_i, i>^T <k_{j_r}, j_r>)} <v_{j_p}, j_p> = \sum_{p=1}^n \frac{\exp(q_i^T R(j_p - i)k_{j_p})}{\sum_{r=1}^n \exp(q_i^T R(j_r - i)k_{j_r})} v_{j_p}
$$

Translation invariance: the position-dependent factors $R(j_p -i)$ and $R(j_r -i)$ depend only on relative position. Shift query and key by the same amount, output unchanged.

If key and value are **shared**, the output becomes:

$$
a_i = \sum_{p=1}^n \frac{\exp(<q_i, i>^T <k_{j_p}, j_p>)}{\sum_{r=1}^n \exp(<q_i, i>^T <k_{j_r}, j_r>)} <k_{j_p}, j_p> = \sum_{p=1}^n \frac{\exp(q_i^T R(j_p -i)k_{j_p})}{\sum_{r=1}^n \exp(q_i^T R(j_r -i)k_{j_r})} R(j_p) k_{j_p}
$$

Now $R(j_p)$ leaks **absolute** position. Fix: inverse RoPE on the attention output:

$$
R(-i) a_i = R(-i) \sum_{p=1}^n \frac{\exp(<q_i, i>^T <k_{j_p}, j_p>)}{\sum_{r=1}^n \exp(<q_i, i>^T <k_{j_r}, j_r>)} <k_{j_p}, j_p> = \sum_{p=1}^n \frac{\exp(q_i^T R(j_p -i)k_{j_p})}{\sum_{r=1}^n \exp(q_i^T R(j_r -i)k_{j_r})} R(j_p -i) k_{j_p}
$$

Only $R(j_p -i)$ remains; translation invariance is back. Related discussion: https://kexue.fm/archives/10862.

### Implementation details: exact position ranges and causality conditions

For each compressed index $j$: combine a fixed local group of original tokens, apply RoPE **once** at the compressed token’s **anchor** position, then store that compressed token.

- **`c4a`:** the $j$-th compressed token is a weighted sum over $[4j - 4, 4j + 3]$ ($j$ from 0; negative indices treated as tokens with value 0). RoPE position: $4j$.
- **`c128a`:** weighted sum over $[128j, 128j + 127]$ ($j$ from 0). RoPE position: $128j$.

Causality: a query at $i$ may only see information produced by tokens in $[0, i]$. So for query $i$ and compressed index $j$: $i \ge 4j + 3$ (`c4a`) or $i \ge 128j + 127$ (`c128a`).

### Implementation details: The exact value of $k$ in c4a and c128a

Defaults in DeepSeek V4: $k = 512$ for `c4a`, $k = 8192$ for `c128a`. (DeepSeek V3.2 default $k = 2048$.)

`c128a` is more compressed: a 1M-token context has at most **8k** compressed tokens. That is small enough for **full** attention over the compressed tokens. Implementation still frames it as sparse attention whose top-$k$ is 8192.

### Implementation details: why the short sliding window is needed

With `c128a`, a query at position **100** cannot attend to any compressed token: the first compressed token holds positions $0$–$127$, and causality forbids attending past 100. The short sliding window lets that query attend to uncompressed tokens in $[0, 100]$, so local information is still there.

### Arithmetic behind the estimates for the 8.7× savings

Sequence length 1M (`1,048,576` tokens).

DeepSeek V3.2 with bf16 KV:

- MLA cache per token per layer: $(512 + 64) \times 2 = 1152$ bytes.
- Indexer cache per token per layer: $128 \times 2 = 256$ bytes.
- Total cached state per token per layer: $1152 + 256 = 1408$ bytes.
- At 1,048,576 tokens: $1{,}048{,}576 \times 1408 \approx 1.375$ GiB per layer.
- Over **61** layers: about **83.9 GiB**.

DeepSeek V4, 61 layers, bf16 KV:

- Each shared-KV cached entry: $512 \times 2 = 1024$ bytes.
- Each `c4a` indexer cached entry: $128 \times 2 = 256$ bytes.
- `c4a` layer: shared-KV $(128 + 1{,}048{,}576 / 4) \times 1024$ plus indexer $(1{,}048{,}576 / 4) \times 256$, about **320.1 MiB**.
- `c128a` layer: $(128 + 1{,}048{,}576 / 128) \times 1024 \approx 8.1$ MiB.
- Total across **30** `c4a` layers and **31** `c128a` layers: about **9.62 GiB**.
