---
source: https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg
lang: en
fetched: 2026-09-04
---

# Disaggregated Serving for Hybrid SSM Models in vLLM

Chinese: [zh/vllm/blog/serving/hybrid-ssm.md](../../../../zh/vllm/blog/serving/hybrid-ssm.md)

2026-04-21. `vllm>=0.20.0`. Nemotron-H-style: Mamba SSM interleaved with full attention (FA). Additive on NIXL — no SSM layers ⇒ old transformer path unchanged. Study note.

Key ideas in the post:

- **Dual descriptor views** over the same physical memory
- **Physical vs logical blocks** (FlashInfer-style kernel block vs HMA/user block)
- **3-descriptor conv transfer** for heterogeneous TP without sender-side reshuffle

Builds on the [HMA interface for NIXL](https://github.com/vllm-project/vllm/pull/35758):

- [#36687](https://github.com/vllm-project/vllm/pull/36687) — dual descriptors + homogeneous TP
- [#37416](https://github.com/vllm-project/vllm/pull/37416) — DS conv layout for Mamba kernels
- [#37635](https://github.com/vllm-project/vllm/pull/37635) — heterogeneous-TP 3-descriptor conv
- [#37310](https://github.com/vllm-project/vllm/pull/37310) — N-1 Prefill for Mamba P/D

Local figures (copyright remains with the original site; study copies):

![transfer volume vs isl](../../../../assets/vllm/blog/serving/hybrid-ssm/01-transfer-volume-vs-isl.png)

![disagg vs colocated](../../../../assets/vllm/blog/serving/hybrid-ssm/02-disagg-vs-colocated.png)

## Background: NIXL P/D for a dense transformer

Four phases:

1. **Register** KV tensors for RDMA
2. **Per-block descriptors** `(address, length, device_id)` — transfer unit is a block, not a whole region
3. **Handshake** once per P–D pair (agent handles, block counts, lengths)
4. **Transfer** — scheduler tells D which blocks; D maps `block_id → descriptor_id`, RDMA READ, poll

Uniform model: `M` regions × `N` blocks. Block `b` in region `r` → descriptor `r * N + b`. Hybrids break the uniform `(address, length)` list: FA and SSM want different descriptor sizes and block counts.

## Why FA and SSM are different animals

Dense KV: `[num_blocks, 2, block_size, num_kv_heads, head_dim]` (or a layout variant). Same block size, page size, block count per layer.

Mamba stores a **collapsed** conv + temporal SSM — no token axis:

```text
Conv:  (conv_dim, state_len)                 e.g. (3072, 3)      bf16
SSM:   (num_heads, head_dim, state_size)     e.g. (32, 64, 128)  fp32
```

Each block is a full snapshot. `block_size` for SSM is effectively **1**. The block is still the transfer unit.

### HMA shared tensor

Hybrid Memory Allocator groups by type (all FA, all SSM, …) then pools so **same-position layers share one physical tensor**. One page: FA sees K/V; Mamba sees conv+SSM+pad. FA page ~ `block_size * num_kv_heads * head_dim * 2`; SSM page `conv_state_bytes + ssm_state_bytes`. HMA **raises FA `block_size` until it is ≥ Mamba**, then pads Mamba rows so page sizes match in bytes.

A single NIXL descriptor list cannot index both views. Heterogeneous TP (D TP ≠ P TP) also needs K/V (and Conv/SSM) on **separate** descriptors so heads can be sliced.

FA descriptor for block `b`: `base + b * page_size`, length `fa_block_len`. Mamba for the same `b`: same base, length `conv_size` or `ssm_size`.

## Dual descriptors

Two lists over the same memory, concatenated under one NIXL transfer handle. FA first: `M` regions × `N_phys` blocks, K and V split. Then Mamba: `M` regions × `N_log` blocks. Homogeneous TP: Conv+SSM (2 sub-regions). Heterogeneous: x, B, C, SSM (see below).

```text
if is_fa_group:
    desc_id = region_id * N_phys + block_id
else:  # mamba
    desc_id = mamba_region_id * N_log + block_id + num_descs
```

`num_descs = M * N_phys` is the FA section length. When `N_phys = N_log = N` the two counts coincide; next section is when they do not.

## Physical vs logical

FlashInfer-class backends want a fixed **physical** block (e.g. **16 tokens**) that may differ from the user/HMA **logical** block.

```text
physical_blocks = logical_blocks * ratio
ratio = logical_block_size / kernel_block_size
```

That ratio applies to **FA only**. SSM has no token axis to split → always `logical_blocks`. FA section: `M * N_phys` with `N_phys = N_logical * ratio`. Mamba section: `M * N_logical`. Tracked in `_physical_blocks_per_logical`, **per engine** (P and D can differ when TP differs). `_get_block_descs_ids` picks the stride by group.

## 3-descriptor conv

Homogeneous TP: each D rank reads matching conv+SSM from the matching P rank.

Heterogeneous example `P_TP=1, D_TP=4`: four D workers each need their shard. SSM temporal state shards on **heads** (first axis) — easy. Conv is `[x | B | C]` with widths `intermediate_size / TP`, `groups_ss / TP`, `groups_ss / TP`. **SD layout** `(state_len, dim)` **interleaves** those sub-projections. Gathering non-contiguous bytes is impractical for zero-copy RDMA.

Require **DS** `(dim, state_len)` via `VLLM_SSM_CONV_STATE_LAYOUT=DS`:

```text
|--- x ---|--- B ---|--- C ---|--- SSM ---|
```

Each D rank issues three contiguous reads for x/B/C (still **one** NIXL READ). `remote_conv_offsets` finds the slice inside the P page using the TP ratio → **4** descriptor regions per Mamba layer (x, B, C, SSM) vs **2** (Conv, SSM) when homogeneous. Larger descriptor list; RDMA stays contiguous.

No extra staging buffer. No reshuffle. Why they refuse “send whole conv, slice on D”:

- No temp buffer the size of **P’s full conv** on every D rank. Nemotron-H example: `3 * 3072 * 2` bytes bf16 per block, × thousands of blocks × all Mamba layers — that is KV space stolen.
- Bytes land in final KV layout; no post-transfer permute kernel.
- Each D rank moves only its **1/TP** share (`D_TP=4` → 4× less per rank).
- Skip HMA padding: Mamba descriptors are `conv_bytes + ssm_bytes`, not the padded page.

They had not measured noticeable kernel regressions from DS in colocated setups; they might make DS the default later.

Figure 1: Nemotron Super **120B**, TP=4, FA `block_size=4224` (HMA). Naive = transfer full HMA-padded Mamba pages; Optimal = real conv+SSM only. Measured NIXL bytes match Optimal. **fp8:** FA page is 1 byte/elem vs 2, padding negligible in that config. **bf16:** ~**50 MB** unnecessary transfer **per request** removed. Mamba state is a **fixed per-request** summary, so transfer volume vs ISL tracks the **FA** block count.

## Nemotron-H walk-through

Serve `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`, disagg P/D, **TP=2**.

- **52** layers, alternating Mamba/FA. HMA → **5** groups (**4** Mamba, **1** FA) → **6** shared KV tensors after pooling.
- FA: `[num_blocks, 2, block_size=400, 4, 128]` (HMA-inflated block_size)
- SSM: `[num_blocks, 3, 3072]` conv + `[num_blocks, 48, 64, 128]` ssm
- Register 6 regions; FA descriptors all 6 × `N_phys` (K/V split); append Mamba 6 × `N_logical` with 4 sub-regions for 3-descriptor transfer
- Scheduler hands per-group block IDs `[[fa_block_ids], [mamba_block_ids_g0], …]`; D maps FA with `region * N + block_id`, Mamba with `num_descs` offset and `N_logical` stride; one `make_prepped_xfer` READ; on completion D notifies P to free. No intermediate buffers.

## Performance

**8× H200**, NVLink. Model: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` (120B LatentMoE, Mamba2 + FA). Co-located: TP=8, all 8 GPUs. Disagg: 1P TP=4 + 1D TP=4, same GPU count. Concurrency **8–256**. ShareGPT. High warmup so KV is “scrambled” (avoid the contiguous-block boost of a fresh run); descriptor count in metrics should stay constant over a full sweep. Prefix-caching **off**. Figure 2: disagg Pareto-dominates colocated at **higher batch sizes** — Decode isolated from Prefill, larger batches without stall, higher output tok/s/GPU at high concurrency. Same pattern as dense P/D.

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

## Limitations (then-current)

- **Mamba1:** 3-descriptor conv is **Mamba2 only**. Mamba1 temporal shape `(intermediate_size // tp, state_size)` cannot reconstruct `intermediate_size` for conv split. **GDN** (Qwen3.5+) on the disagg [roadmap](https://github.com/vllm-project/vllm/issues/33702)
- **Speculative decoding** + SSM transfer: not extensively validated
- **Mixed block sizes with HMA:** `block_size_ratio > 1` between P and D **not yet** supported when HMA is on

Acknowledgements: Thomas Parnell (IBM Research), Roi Koren (NVIDIA).

Router-style P/D assumed one shape of memory. Hybrids need two dictionaries on the same NIXL pipe.
