---
source: https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg
lang: en
fetched: 2026-09-05
---

# Disaggregated Serving for Hybrid SSM Models in vLLM

Chinese: [zh/vllm/blog/serving/hybrid-ssm.md](../../../../zh/vllm/blog/serving/hybrid-ssm.md)  
Original: https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg  
2026-04-21. Nicolò Lucchesi, Zhanqiu Hu (Red Hat), and the vLLM team. `vllm>=0.20.0`. Study extract, not an official reprint.

Hybrid architectures that interleave Mamba-style SSM layers with full-attention (FA) layers — [NVIDIA Nemotron-H](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8) is the running example — mix linear-time state-space efficiency with attention’s expressiveness. vLLM already does disaggregated prefill/decode (P/D) for standard transformers through its [NIXL-based KV connector](https://blog.vllm.ai/2025/01/27/v0-disagg-prefill.html): a prefill instance computes KV blocks and a decode instance pulls them over RDMA. Extending that to hybrids is not a uniform-KV story. FA and SSM store different state, in different layouts and sizes, while the block manager and NIXL connector were built around one cache format.

This post extends the NIXL connector for hybrid SSM-FA in disaggregated mode. Three ideas:

- **Dual descriptor views** — two NIXL block-descriptor lists over the same physical memory, with different offsets and lengths: one for FA, one for SSM.
- **Physical/logical block bridging** — the logical block the block manager sees vs the physical block attention kernels require.
- **3-descriptor conv transfer** — a decomposition of the Mamba conv state that lets heterogeneous tensor-parallel transfers happen without reshuffling on the sender.

None of this changes the existing workflow for standard transformers. Additive only: it activates when the model has SSM layers. Builds on the [HMA interface for NIXL](https://github.com/vllm-project/vllm/pull/35758):

- [#36687](https://github.com/vllm-project/vllm/pull/36687) — dual descriptors and homogeneous-TP support for hybrid SSM-FA
- [#37416](https://github.com/vllm-project/vllm/pull/37416) — DS conv-state layout for Mamba kernels
- [#37635](https://github.com/vllm-project/vllm/pull/37635) — heterogeneous-TP 3-descriptor conv transfer
- [#37310](https://github.com/vllm-project/vllm/pull/37310) — N-1 prefill for Mamba P/D

Local figures (copyright remains with the original site; study copies):

![transfer volume vs isl](../../../../assets/vllm/blog/serving/hybrid-ssm/01-transfer-volume-vs-isl.png)

![disagg vs colocated](../../../../assets/vllm/blog/serving/hybrid-ssm/02-disagg-vs-colocated.png)

## Background: the NIXL KV transfer workflow

Four phases for a standard transformer:

1. **Register memory regions** — each worker registers its KV tensors with NIXL for RDMA.
2. **Create block descriptors** — per registered region, per-block `(address, length, device_id)`. The transfer unit is a block, not a whole region.
3. **Handshake** — when a decode (D) worker first pulls from a prefill (P) worker, they exchange metadata: agent handles, block counts, block lengths. Once per P–D pair.
4. **Transfer** — the scheduler tells D which blocks to pull. D maps `block_id → descriptor_id`, issues an RDMA READ, polls for completion.

For a standard model with `M` registered regions and `N` blocks:

```text
+----------------------------------+
| Region 0: desc_0 ... desc_{N-1}  |
| Region 1: desc_0 ... desc_{N-1}  |
| ...                              |
| Region M: desc_0 ... desc_{N-1}  |
+----------------------------------+
```

Block `b` in region `r` → descriptor index `r * N + b`.

Hybrids break the uniform scheme: FA and SSM want different descriptor sizes and different block counts.

## The challenge: FA and SSM state are different animals

In a standard transformer every layer’s KV has the same shape: `[num_blocks, 2, block_size, num_kv_heads, head_dim]` (or a layout variant). Same block size, page size, and block count per layer.

Mamba stores a collapsed **conv state** and a **temporal SSM state** — no token axis:

```text
Conv state:  (conv_dim, state_len)              e.g. (3072, 3)      -- bf16
SSM state:   (num_heads, head_dim, state_size)  e.g. (32, 64, 128)  -- fp32
```

These are a fixed-size summary of the whole sequence. `block_size` for SSM is effectively **1**: each block is a complete snapshot, not a group of per-token vectors. **A block is still the single unit of transfer.**

### The HMA shared-tensor layout

vLLM’s Hybrid Memory Allocator (HMA) groups layers by type (all FA, all SSM, …) then pools so **layers at the same position in each group share one physical tensor**. Blocks are interchangeable. The same tensor is FA K/V to one group and conv+SSM+pad to another. For a Nemotron-H-style model:

```text
                KV Cache Tensor (shared via HMA pooling)
                 /                        \
                /                          \
     Attention (FA) View              Mamba View
              |                            |
    +-----------------------+    +-----------------------+
    | Block 0               |    | Block 0               |
    |   Key     |  Value    |    |  Conv |    SSM  |[pad]|
    | Block 1               |    | Block 1               |
    |   Key     |  Value    |    |  Conv |    SSM  |[pad]|
    |  ...                  |    |  ...                  |
    +-----------------------+    +-----------------------+
```

Page sizes differ. FA pages are governed by `block_size * num_kv_heads * head_dim` (`*2` for K/V); SSM pages are `conv_state_bytes + ssm_state_bytes`. HMA **raises FA `block_size` until it is ≥ Mamba**, then pads Mamba rows (`+[pad]`) so both groups have equal page sizes in bytes.

**The NIXL problem:** a single descriptor list with uniform `(address, length)` cannot index both views. Heterogeneous setups (D TP ≠ P TP) also need K/V (and Conv/SSM) on **separate** descriptors so heads can be sliced.

An FA descriptor for block `b` is `base + b * page_size` with length `fa_block_len`. A Mamba descriptor for the same `b` uses the same base with length `conv_size` or `ssm_size`.

## Dual descriptor views

Register **two descriptor lists** over the same physical memory, concatenated under one NIXL transfer handle:

```text
+------------------------------------------------------+
|  FA descriptors (M regions x N_phys blocks)          |
|                                                      |
|  Region 0                                            |
|    FA_desc_K[0], FA_desc_K[1], ... FA_desc_K[N-1]    |
|    FA_desc_V[0], FA_desc_V[1], ... FA_desc_V[N-1]    |
|  Region 1                                            |
|    ...                                               |
|  Region M                                            |
|    ...                                               |
|                                                      |   ^
|  --------------------------------------------------- |   | num_descs
|                                                      |   v
|  Mamba descriptors (M regions x N_log blocks)        |
|                                                      |
|  Region 0                                            |
|    Mamba_desc_x[0]   ... Mamba_desc_x[N-1]           |
|    Mamba_desc_B[0]   ... Mamba_desc_B[N-1]           |
|    Mamba_desc_C[0]   ... Mamba_desc_C[N-1]           |
|    Mamba_desc_SSM[0] ... Mamba_desc_SSM[N-1]         |
|  Region 1                                            |
|    ...                                               |
|  Region M                                            |
|    ...                                               |
+------------------------------------------------------+
```

`N_phys` / `N_log` are physical vs logical blocks. You can assume `N_phys = N_log = N` until the next section.

The Mamba section already shows the conv-state split into x, B, C (see 3-descriptor transfer below). For homogeneous TP those collapse to two sub-regions (Conv, SSM).

FA occupies the first `num_descs = M * N_phys` slots. Mamba follows. Mapping:

```python
if is_fa_group:
    desc_id = region_id * N_phys + block_id
else:  # mamba group
    desc_id = mamba_region_id * N_log + block_id + num_descs
```

## Physical vs logical block sizes

FlashInfer-class backends want a fixed **physical** block (e.g. **16 tokens**) that may differ from the user- or HMA-computed **logical** block.

For standard models:

```text
physical_blocks = logical_blocks * ratio
ratio = logical_block_size / kernel_block_size
```

For hybrids the ratio applies **only to FA**. SSM has no token dimension to split, so it always uses `logical_blocks`. The two sections of the descriptor list therefore have different counts:

```text
FA section:    M regions * N_phys blocks    (N_phys = N_logical * ratio)
Mamba section: M regions * N_logical blocks
```

Tracked in `_physical_blocks_per_logical`, **per engine** (P and D can differ when TP differs). `_get_block_descs_ids` picks the stride by whether it is resolving an FA group or a Mamba group.

## The 3-descriptor conv transfer

Homogeneous TP (same `--tensor-parallel-size` on P and D): each D rank reads matching conv + SSM from the matching P rank.

Heterogeneous TP is harder. Example `P_TP=1, D_TP=4`: four D workers each need their shard from one P worker. SSM temporal state shards on **heads** (first axis) — easy. Conv is:

```text
Conv state = [x | B | C]     where x, B, C are sub-projections
              ^   ^   ^
              |   |   |
     intermediate_size / TP   groups_ss / TP   groups_ss / TP
```

With the standard **SD** layout `(state_len, dim)`, those sub-projections are interleaved. Gathering non-contiguous bytes is impractical for zero-copy RDMA.

### The DS layout

Require **DS** `(dim, state_len)` via `VLLM_SSM_CONV_STATE_LAYOUT=DS`. Each sub-projection is contiguous:

```text
DS layout within one page:

|--- x (x_bytes) ---|--- B (b_bytes) ---|--- C (b_bytes) ---|--- SSM ---|
```

Each D rank reads its slice of `x`, `B`, and `C` with three contiguous RDMA reads — hence “3-descriptor transfer” (still **one** NIXL READ).

For heterogeneous TP, `remote_conv_offsets` finds each D rank’s slice inside the P page from the TP ratio. That is **4** descriptor regions per Mamba layer (x, B, C, SSM) vs **2** (Conv, SSM) when homogeneous. Larger descriptor list; RDMA stays contiguous.

**No extra in-memory staging buffer** on either GPU.  
**No data reshuffling** on either side.

They had not measured noticeable kernel regressions from DS in colocated setups; they may make DS the default later.

### Zero overhead: no extra buffers, no permutation

The simpler alternative is “send the whole conv to every D rank, permute/slice locally.” They refuse that for Mamba:

- **No staging buffer.** Permuting on D needs a temp buffer the size of **P’s full conv** on every D worker. Nemotron-H: `3 * 3072 * 2` bytes bf16 per block, × thousands of blocks × all Mamba layers — KV space stolen.
- **No post-transfer reshuffling.** With DS, each D rank reads exactly the bytes it needs, directly into the final KV location. No permute kernel. The transfer completes and the state is usable.
- **Transfer only what you own.** Each D rank moves its `1/TP` share. `D_TP=4` → 4× less per rank than “transfer everything, slice locally.”
- **Skip HMA padding.** HMA pads SSM pages to match FA. Mamba descriptors are sized to actual `conv_bytes + ssm_bytes`, not the padded page. Padding never goes over the wire. When FA pages are much larger than raw SSM, that is a real per-block saving.

Figure 1 checks this on Nemotron Super **120B**, TP=4, FA `block_size=4224` (HMA). For each KV dtype (bf16 and fp8): **Naive** = full HMA-padded Mamba pages; **Optimal** = real conv+SSM only, no padding, no auxiliary buffers. Measured NIXL bytes match Optimal.

**fp8:** FA page is 1 byte/elem vs 2, so padding is negligible in that config.  
**bf16:** ~**50 MB** of unnecessary transfer **per request** removed.

Mamba state is a **fixed per-request** summary, so transfer size vs ISL tracks the **FA** block count.

*Figure 1 (original caption): P→D transfer volume vs input sequence length for Nemotron Super 120B (TP=4, FA block_size=4224). Naive and Optimal are computed analytically from page sizes and block counts. Measured is actual bytes transferred as reported by NIXL during disaggregated P/D. Optimal eliminates HMA padding; the measured line follows it.*

## Putting it together: Nemotron-H walk-through

Serve `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` with disaggregated P/D at **TP=2**.

**Model structure:** **52** layers, alternating Mamba and FA. HMA → **5** groups (**4** Mamba, **1** FA). After pooling: **6** shared KV tensors.

**KV cache layout:**

```text
FA layers:    [num_blocks, 2, block_size=400, 4, 128]   # K/V with HMA-inflated block_size
SSM layers:   [num_blocks, 3, 3072]  (conv)  +  [num_blocks, 48, 64, 128]  (ssm)
```

HMA pads so both views have the same page size in bytes. The kernel (FlashInfer / FlashAttention) may further subdivide FA blocks, creating a physical/logical ratio.

**Descriptor registration:**

1. The 6 shared tensors are registered as NIXL regions (same as dense models).
2. FA descriptors for all 6 regions × `N_phys` blocks, indexing K and V separately.
3. Mamba descriptors appended: 6 regions × `N_logical` blocks, 4 sub-regions each (x, B, C, SSM) for 3-descriptor transfer.

**Transfer flow:**

1. P finishes prefill. The scheduler assigns per-group block IDs: `[[fa_block_ids], [mamba_block_ids_g0], [mamba_block_ids_g1], ...]`.
2. D maps them: FA uses `region * N + block_id`; Mamba adds the `num_descs` offset and uses `N_logical` stride.
3. D issues a single `make_prepped_xfer` READ with both FA and Mamba descriptors, then polls.
4. On completion D notifies P so P can free the blocks.

From D’s perspective the whole transfer is **one** async operation. No intermediate buffers, no reshuffling.

## Performance

**8× H200**, NVLink. Model: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` (120B LatentMoE, interleaved Mamba2 and full attention).

- **Co-located baseline:** one instance, TP=8, all 8 GPUs.
- **Disaggregated P/D:** 1 prefill (TP=4, 4 GPUs) + 1 decode (TP=4, 4 GPUs), same total GPU count.

Concurrency **8–256**. Plot output throughput per GPU against per-user output token rate (*Interactivity*). Workload: ShareGPT.

Very high warmup so KV is “scrambled” — a fresh run’s contiguous-block allocation gives a boost that does not match long-running use. You can also check that descriptor count in metrics stays constant over a full dataset sweep. Prefix-caching **off**.

*Figure 2 (original caption): Disaggregated P/D vs co-located serving for a hybrid SSM model. Throughput-vs-latency Pareto across concurrency. Prefix-caching disabled.*

Same pattern as dense transformers: disagg Pareto-dominates colocated at **higher batch sizes**. Decode isolated from prefill can sustain larger batches without stalling → higher output tok/s per GPU at high concurrency.

## Getting started

```bash
# Prefill instance
VLLM_SSM_CONV_STATE_LAYOUT=DS vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --max-model-len 8192 \
    --block-size 128 \
    --no-disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

`VLLM_SSM_CONV_STATE_LAYOUT=DS` is **required for heterogeneous TP**, not otherwise.

## Limitations and future work (then-current)

- **Mamba1:** 3-descriptor conv is **Mamba2 only**. Mamba1 temporal shape `(intermediate_size // tp, state_size)` cannot reconstruct `intermediate_size` for the conv split. **GDN** (Qwen3.5+) is on the disagg [roadmap](https://github.com/vllm-project/vllm/issues/33702).
- **Speculative decoding:** interaction with SSM state transfer had not been extensively validated.
- **Mixed block sizes with HMA:** different P vs D block sizes (`block_size_ratio > 1`) **not yet** supported when HMA is on.

## Acknowledgments

Thomas Parnell (IBM Research), Roi Koren (NVIDIA).

Router-style P/D assumed one shape of memory. Hybrids need two dictionaries on the same NIXL pipe. Neighbors: [router.md](router.md), [mooncake.md](mooncake.md), [moriio.md](moriio.md), [large-scale.md](large-scale.md).
