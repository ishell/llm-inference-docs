---
source: https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1
lang: en
fetched: 2026-09-04
---

# Driving vLLM WideEP and Large-Scale Serving Toward Maturity on Blackwell (Part I)

Chinese: [zh/vllm/blog/serving/gb200-wideep.md](../../../../zh/vllm/blog/serving/gb200-wideep.md)

2026-02-03. **Meta and NVIDIA Team**. Follows [large-scale.md](large-scale.md) — DeepSeek @ ~**2.2k tok/s/H200** with wide-EP. Study note; the page’s demo numbers, not a promise for your box.

Not [epd.md](epd.md): this is **text Prefill/Decode split + wide EP**, not visual-encoder disaggregation.

Local figures (copyright remains with the original site; study copies). Interleaved with the matching sections below.

## Introduction

After the H200 wide-EP work, the same team kept optimizing for NVIDIA **GB200**. The headline: **26.2K Prefill TPGS** (tokens per GPU second) and **10.1K Decode TPGS** on GB200, workload **2K input / 2K output**, DeepSeek-style MoE — DeepSeek R1 / V3 / V3.1. Collected on a deployment of **4 Prefill instances × 2 GB200** plus **1 Decode instance × 8 GB200**, all mixing data-parallelism (DP) and expert-parallelism (EP).

New knobs named on the page:

- Lower-precision ops ([NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) GEMM, FP8 GEMM, NVFP4 MoE Dispatch)
- Kernel fusion (RoPE+Quant+Q write, RoPE+Quant, Concat K)
- Scaling down Prefill via weight offloading
- Minimized chunking overheads

Already discussed in the H200 post, still in the mix: async scheduling; Prefill/Decode disaggregated serving.

GB200’s extra compute plus those targeted opts is the claimed step up from H200.

## Results

Same 2K/2K workload. GB200 vs H200 for DeepSeek-V3/R1. Setup:

![topline comparison](../../../../assets/vllm/blog/serving/gb200-wideep/01-topline_comparison.png)

| Deployment setup | H200 | GB200 |
| :---- | :---- | :---- |
| Prefill | 16 GPUs | 8 GPUs (4 instances × 2 GPUs) |
| Decode | 32 GPUs | 8 GPUs (1 instance × 8 GPUs) |

The page attributes the headroom to GB200’s memory bandwidth (**8 TB/s** vs **4.8 TB/s**), higher compute through FP4, and **NVLink-C2C** between CPU and GPU — then to the opts below. The topline figure is on the page; prose does not restate H200 TPGS from that chart (the earlier H200 line remains ~2.2k tok/s/H200 in [large-scale.md](large-scale.md)). Summary later calls the GB200 pair a **3–5×** improvement over H200.

They also swept DeepSeek-V3/R1 **Decode** throughput on GB200 across standard workloads, same parallelism, varying the Decode batch size that fully utilizes GPU memory. Reproduce-all pointer: [vllm#33583](https://github.com/vllm-project/vllm/issues/33583). Chart values are not tabulated in the text.

![decode throughput various](../../../../assets/vllm/blog/serving/gb200-wideep/02-decode_throughput_various.png)

## Key Optimizations

### Lower-Precision Operations

GB200’s FP4 / FP8 throughput is substantially higher than H200. vLLM uses that in three places.

#### NVFP4 GEMM (MoE GEMMs, O-proj)

DeepSeek-V3/R1 MoE expert weights and output-projection layers can be quantized to FP4. vLLM integrates FlashInfer’s **TRTLLM-Gen** GEMM kernels, scheduled for GB200 FP4 tensor cores.

The checkpoint stores packed 4-bit weights with per-group scales. At runtime the TRTLLM-Gen kernels dequantize on the fly inside the tensor cores — near-native FP4 throughput, quality claimed held.

Implementation notes on the page:

- FP4 weights with **FP8 or FP16** scales, packed
- FlashInfer TRTLLM-Gen kernels for GB200 tensor-core scheduling
- Applied to **MoE expert GEMMs** and attention **O-proj**

#### FP8 GEMM for MLA

MLA’s query up-projection (latent → full query) uses **FP8**, not FP4. The post’s tradeoff: MoE likes FP4 throughput; attention projections are more quantization-sensitive, so they keep FP8. Optimized FP8 GEMM: significant speedup over FP16, attention quality claimed held.

#### NVFP4 MoE Dispatch

Dispatch — routing tokens to experts — also drops precision. **NVFP4 dispatch** quantizes token activations to FP4 **before** all-to-all. Communication volume **4×** smaller than FP16 dispatch, which cuts inter-GPU latency in EP. Quantization overhead is amortized across that saving.

### Kernel Fusion

Fuse to cut HBM traffic and launch overhead.

#### RoPE + Quant + Q Write (Decode)

Decode query path:

1. RoPE
2. Quantization for the next GEMM
3. Write to the query buffer

One kernel; two intermediate round-trips gone.

![rope quant fusion timeline](../../../../assets/vllm/blog/serving/gb200-wideep/03-rope_quant_fusion_timeline.png)

RoPE+Quant+Q Write fusion in Decode.

#### RoPE + Quant (Prefill)

Prefill fuses RoPE and quantization. Larger token batches, so the bandwidth win is even more visible.

#### Concat K Optimization

MLA keys: FlashInfer `concat_mla_k`. The key is two parts — `k_nope` (per-head, no positional embedding) and `k_rope` (shared across heads). They must be concatenated.

Naive path: copy `k_nope` and broadcast `k_rope` across all **128** heads — a lot of bandwidth. `concat_mla_k`:

- **Warp-based processing:** each warp handles one `(token, head_chunk)` pair, **16** heads at a time
- **Vectorized access:** 8-byte vector loads for nope, 4-byte loads for rope
- **Software pipelining with L2 prefetch:** next row while the current row runs
- **Register reuse for rope:** rope is shared, loaded once into registers, written to all 16 heads in the chunk

### Scaling Down Prefill

#### Why Scaling Down Makes Sense

Throughput serving usually adds GPUs to fit the model or to shard memory (experts, context) so the batch can grow. Prefill that is already **compute-bound** can go the other way: fewer GPUs, less communication.

Microbenchmarks on the page: MLA backend throughput starts plateauing as batch size goes from **16K to 64K** tokens. Beyond **64K**, MoE throughput gains are also negligible. Compute saturates at a batch that fits a **2-GPU** serving setup.

![mla trtllm ragged prefill prefill](../../../../assets/vllm/blog/serving/gb200-wideep/04-mla_trtllm_ragged_prefill_prefill.png)

![moe flashinfer trtllm nvfp4 prefill](../../../../assets/vllm/blog/serving/gb200-wideep/05-moe_flashinfer_trtllm_nvfp4_prefill.png)

MLA and MoE throughput plateau at ~64K batch size.

Dropping GPU count **4 → 2** halves the NCCL collectives (`all_gather` and `reduce_scatter`) for EP.

![nccl all gather](../../../../assets/vllm/blog/serving/gb200-wideep/06-nccl_all_gather.png)

![nccl reduce scatter](../../../../assets/vllm/blog/serving/gb200-wideep/07-nccl_reduce_scatter.png)

Reducing EP degree halves communication overhead.

#### Weight Offloading v2

Shrink the GPU footprint without giving up throughput: weight offloading **v2** with asynchronous prefetch. Inspired by [SGLang Prefill offload](https://github.com/sgl-project/sglang/pull/8034), then adapted for `torch.compile` and CUDA graph inside vLLM.

**v1:** offloaded weights stayed on CPU and were touched via UVA — slow PCIe. Last resort when GPU memory was simply not enough.

**v2:** explicitly copy (onload) weights onto the GPU ahead of time. The next layer’s weights onload on a **separate CUDA stream**. Overlap onload with kernel execution and the delay can be fully hidden.

Group-based knobs:

![layer group](../../../../assets/vllm/blog/serving/gb200-wideep/08-layer_group.png)

- `group_size`: group every N layers
- `num_in_group`: offload this many layers per group (last N of each group)
- `prefetch_step`: how many layers to prefetch ahead

For **DeepSeek-R1 Prefill** serving they offload **one of every two** MoE GEMM weights — memory down, throughput claimed full.

![onloading trace](../../../../assets/vllm/blog/serving/gb200-wideep/09-onloading_trace.png)

Trace: weight onload overlapping layer execution.

GB200’s **NVLink-C2C** CPU–GPU link is why v2 bites harder here than on PCIe boxes.

### Minimize Chunking Overheads

Large MoE batches need chunks to fit GPU memory. Small chunks re-pay launch and sync, and leave GPU bubbles. vLLM exposes chunk sizes; this post’s **GB200** choices are not H200 defaults — interconnect and kernel shapes differ. Do not paste the H200 note’s knobs here.

#### MoE DP Chunk

Under DP+EP, tokens dispatch from each DP rank in coordinated chunks. `VLLM_ENABLE_MOE_DP_CHUNK` (on by default) turns that on.

Larger chunks amortize dispatch/combine. Size: `VLLM_MOE_DP_CHUNK_SIZE` (default **256** tokens).

**On GB200:** disable MoE DP chunking for Prefill (`VLLM_ENABLE_MOE_DP_CHUNK=0`); for Decode set `VLLM_MOE_DP_CHUNK_SIZE` to the **batch size**.

#### MoE Activation Chunk

Large Prefill batches: chunk activation tensors through MoE. `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING` (on by default). Size: `VLLM_FUSED_MOE_CHUNK_SIZE` (default **16K** tokens). Optimal is “as large as GPU memory allows.”

**On GB200:** disable activation chunking (`VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=0`) — the larger memory holds full batches.

#### Output Processing Chunk

V1 async serving path chunks output processing (logits, sampling, response). `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` (default **128**). Larger helps throughput; very large chunks can increase inter-message latency variance on streaming.

**On GB200**, throughput-optimized Decode: chunk size **2048**.

## Future Work

Then in flight for GB200:

1. **Load balancedness and scaling up EP** — expert load balancing at larger EP degrees and more dynamic traffic, better rebalancing.
2. **MoE dispatch latency** — cheaper all-to-all via kernel work and communication scheduling.
3. **Hiding communication behind compute** — more aggressive overlap when the path is communication-bound.
4. **WideEP / large-scale on GB300** — more HBM and compute, higher TPGS, smaller host footprint.

Living page: [roadmap.vllm.ai](http://roadmap.vllm.ai).

## Summary

- **26.2K Prefill TPGS** and **10.1K Decode TPGS** for DeepSeek-style MoE — **3–5×** over H200 on the page’s telling.
- Lower precision (NVFP4 GEMM, FP8 GEMM, NVFP4 dispatch) uses GB200 tensor cores.
- Kernel fusion cuts bandwidth and launch tax.
- Scaling Prefill down, plus weight offloading v2, cuts EP communication while keeping compute saturated.
- Chunking knobs (environment variables) are how they killed large-batch overhead **on this platform**.

## Team

- Meta: Ming Yang, Xiaozhu Meng, Pengchao Wang, Lucia (Lu) Fang, Bangsheng Tang, Yan Cui, Hongyi Jia, Jinghui Zhang, Zebing Lin, Jason Park, Yejin Lee, Jaewon Lee, Bradley Davis, Jingyi Yang, Adi Gangidi, Ayush Goel, Charlotte (Ye) Qi, Stephen Chen, Raj Ganapathy, Akshay Hegde, Lu Fang
- NVIDIA: Duncan Moss, Cyrus Chang, Andrew Briand, Siyuan Fu, Hanjie Qiu, Jason Li, Pavani Majety, Xin Li, Chirayu Garg, Abhinav Singh, Minseok Lee

## References

- [vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP](https://blog.vllm.ai/2025/12/17/large-scale-serving.html) — study note: [large-scale.md](large-scale.md)
- [FlashInfer: Kernel Library for LLM Serving](https://github.com/flashinfer-ai/flashinfer)
- [NVIDIA GB200 NVL72 Architecture](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
