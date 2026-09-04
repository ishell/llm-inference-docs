---
source: https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm
lang: en
fetched: 2026-09-04
---

# PTPC-FP8: Boosting vLLM Performance on AMD ROCm

Chinese: [zh/vllm/blog/performance/ptpc-fp8.md](../../../../zh/vllm/blog/performance/ptpc-fp8.md)

2025-02-24. **AMD and Embedded LLM**. PR: [vllm#12501](https://github.com/vllm-project/vllm/pull/12501). Needs **vLLM ≥ 0.7.3**. Numbers are **MI300X** demos at commit `4ea48fb35cf67d61a1c3f18e3981c362e1d8e26f`, not your card’s SLA.

This is **weight + activation** FP8, quantized on the fly from Hugging Face — no pre-quant checkpoint. Distinct from [FP8 KV-cache](fp8-kvcache.md), which is KV dtype and attention compute. Later ROCm attention orchestration: [rocm-attention.md](../architecture/rocm-attention.md). Pushing KV further to 3–4 bit: [turboquant.md](turboquant.md).

**TL;DR from the page**

- **What’s new:** `--quantization ptpc_fp8` (v0.7.3+, AMD ROCm).
- **Why:** Speeds similar to other FP8, accuracy closer to BF16. The post calls it the best FP8 option on ROCm then.
- **How:** Install ROCm; vLLM ≥ 0.7.3; add `--quantization ptpc_fp8` on a Hugging Face model. No pre-quantization.

Local figures (copyright remains with the original site; study copies):

![What is PTPC-FP8](../../../../assets/vllm/blog/performance/ptpc-fp8/01-PTPC121.png)

**PTPC-FP8** = Per-Token-Activation, Per-Channel-Weight FP8. Per-token scales on activations, per-channel scales on weights — tighter than classic per-tensor FP8.

## Introduction

LLMs are expensive to run. FP8 cuts memory and speeds matmuls; classic quantization hits outliers. PTPC’s claim: near-BF16 accuracy at FP8 speed, directly from Hugging Face weights.

### The outlier problem

Past a certain scale, activations grow outliers:

- Per-tensor quantization leaves most values with few effective bits
- Outliers sit **persistently in the same channels**, across tokens
- Weights are relatively uniform and easy to quantize; activations are not

### PTPC: granularity from three observations

1. Outliers consistently appear in the same channels
2. Channel magnitudes within a token vary widely
3. The **same channel’s** magnitude across tokens stays relatively stable

So:

- **Per-token activation quantization:** one scale per input token
- **Per-channel weight quantization:** one scale per weight column

![PTPC Diagram](../../../../assets/vllm/blog/performance/ptpc-fp8/02-PTPC-Diagram.png)

Two approaches in the figure. Tensors:

- $X$: input activations ($T \times C_i$)
- $W$: weights ($C_i \times C_o$)
- $T$: token sequence length; $C_i / C_o$: in / out channels; $*$: matmul

Scales:

- **Top (per-tensor):** scalars $\Delta_X[1]$ and $\Delta_W[1]$ for whole tensors
- **Bottom (PTPC):** vector $\Delta_X[T \times 1]$ (one per token) and $\Delta_W[1 \times C_o]$ (one per input channel)

That granularity is how they claim BF16-like accuracy while staying in 8-bit compute.

## Fused kernel: don’t let fine scales become two HBM trips

Fine-grained scales without fusion would be slow. ROCm’s answer: a **fused FP8 rowwise scaled GEMM**.

### Two-step vs fused

Naive path:

```python
# Naive 2-step approach:
output = torch._scaled_mm(input, weight)       # Step 1: FP8 GEMM
output = output * token_scales * channel_scales  # Step 2: Apply scaling factors
```

Write a large intermediate, read it back, waste bandwidth and cycles.

Fused path: matmul and scaling as one hardware op:

```python
# Optimized fused operation:
output = torch._scaled_mm(input, weight,
                         scale_a=token_scales,
                         scale_b=channel_scales)
```

![Fused GEMM](../../../../assets/vllm/blog/performance/ptpc-fp8/03-FusedGEMM.svg)

MI300X has native FP8. The page’s reasons this matters: scaling happens in on-chip memory before the writeback; fewer redundant ops; up to ~**2.5×** versus the two-step implementation. Without the fused kernel, PTPC’s accuracy edge would pay a memory tax first.

## Speed and accuracy on MI300X

vLLM on AMD MI300X, commit `4ea48fb35cf67d61a1c3f18e3981c362e1d8e26f`.

### Throughput: PTPC vs per-tensor FP8

- Model: Llama-3.1-70B-Instruct
- Dataset: SharedGPT
- GPU: **1× MI300X**
- Result: throughput **virtually identical** to per-tensor FP8 (slightly *better* — ~**1.01×**). The fused kernel absorbs the extra scaling complexity.

![Throughput reqs/s](../../../../assets/vllm/blog/performance/ptpc-fp8/04-PTPCReqs.svg)

![Speedup vs per-tensor FP8](../../../../assets/vllm/blog/performance/ptpc-fp8/05-PTPCSpeedup.svg)

### Accuracy: Wikitext perplexity (lower is better)

- Model: Llama-3.1-8B-Instruct
- Dataset: Wikitext
- Setup: **2× MI300X**, tensor parallelism

Perplexity is how “confused” the model is at next-token prediction. Lower = more confident; higher = more often surprised. The page notes that even a **0.1** bump can be a real quality drop on a heavily optimized LLM.

![bits and byte perplexity](../../../../assets/vllm/blog/performance/ptpc-fp8/06-PerplexityBits.png)

![Word perplexity](../../../../assets/vllm/blog/performance/ptpc-fp8/07-Perplexitywords.png)

| Precision | Word Perplexity | % Degradation |
| --- | ---: | ---: |
| BF16 (baseline) | 9.4281 | — |
| PTPC-FP8 | 9.5093 | 0.86% |
| Standard FP8 | 9.5124 | 0.89% |

PTPC beats standard FP8 (9.5093 vs 9.5124); the gap to BF16 is **0.86%**. `bits_per_byte` / `byte_perplexity` follow the same pattern. Small quality gaps compound on reasoning and generation — that is why they spend this much ink on perplexity.

### Accuracy: GSM8K (math reasoning)

GSM8K: grade-school word problems. Multi-step reasoning, numerical accuracy, logical consistency. Reasoning is often the first thing quantization hurts.

Two scorers:

- **Flexible-extract:** credit if the correct number appears anywhere
- **Strict-match:** the exact answer in the expected format

![GSM8K 8B](../../../../assets/vllm/blog/performance/ptpc-fp8/08-GSM8K8B.png)

**8B, strict-match:**

| Method | Strict-match | vs BF16 |
| --- | ---: | ---: |
| BF16 | 73.2% | 100% |
| PTPC-FP8 | 70.8% | 96.7% |
| Standard FP8 | 69.2% | 94.5% |

![GSM8K 70B](../../../../assets/vllm/blog/performance/ptpc-fp8/09-GSM8K70B.png)

**70B:** PTPC strict-match **87.3%**, slightly above BF16’s **86.3%**. Both beat standard FP8 on strict-match. Treat PTPC “winning” BF16 at 70B as **noise, not free accuracy**.

The page’s own reading: reasoning is preserved; PTPC beats standard FP8 at both sizes; near-BF16 quality at 8-bit memory and speed; the gap between quant methods narrows as models grow, so PTPC is especially worth it on large models.

## Getting started

1. Install a recent ROCm.
2. Path on the page: clone vLLM, build `Dockerfile.rocm`.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
DOCKER_BUILDKIT=1 docker build -f Dockerfile.rocm -t vllm-rocm .
docker run -it \
   --network=host \
   --group-add=video \
   --ipc=host \
   --cap-add=SYS_PTRACE \
   --security-opt seccomp=unconfined \
   --device /dev/kfd \
   --device /dev/dri \
   -v <path/to/model>:/app/model \
   vllm-rocm \
   bash
```

3. Turn on `--quantization ptpc_fp8`. Weights quantize **on the fly**. Replace `<your-model>` with any Hugging Face model:

```bash
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve <your-model> --max-seq-len-to-capture 16384 --enable-chunked-prefill=False --num-scheduler-steps 15 --max-num-seqs 1024 --quantization ptpc_fp8
```

**Caveat from their example:** chunked prefill **off** (`--enable-chunked-prefill=False`), multi-step scheduler **on** (`--num-scheduler-steps 15`), and `VLLM_USE_TRITON_FLASH_ATTN=0`. Trust your version’s docs; do not freeze this line as an eternal default.

## Conclusion

The post frames PTPC as the accuracy–speed sweet spot: near-BF16 quality at FP8 speed, so more people can run large models on AMD hardware. Invitation to try it, report back, and contribute to vLLM. The numbers remain that one MI300X snapshot.

## Appendix: lm-evaluation-harness

Wikitext (8B, `HIP_VISIBLE_DEVICES=0,1`, TP2, `max_model_len=2048`, `gpu_memory_utilization=0.6`, `--batch_size 16`):

```bash
# Unquantized (Bfloat16)
MODEL=meta-llama/Llama-3.1-8B-Instruct
HIP_VISIBLE_DEVICES=0,1 lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True,tensor_parallel_size=2,kv_cache_dtype=auto,max_model_len=2048,gpu_memory_utilization=0.6 \
  --tasks wikitext --batch_size 16

# Per-Tensor FP8 Quantization
MODEL=meta-llama/Llama-3.1-8B-Instruct
HIP_VISIBLE_DEVICES=0,1 lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True,tensor_parallel_size=2,quantization=fp8,kv_cache_dtype=fp8_e4m3,max_model_len=2048,gpu_memory_utilization=0.6 \
  --tasks wikitext --batch_size 16

# Per-Token-Activation Per-Channel-Weight FP8 Quantization
MODEL=meta-llama/Llama-3.1-8B-Instruct
HIP_VISIBLE_DEVICES=0,1 lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True,tensor_parallel_size=2,quantization=ptpc_fp8,kv_cache_dtype=fp8_e4m3,max_model_len=2048,gpu_memory_utilization=0.6 \
  --tasks wikitext --batch_size 16
```

GSM8K (8B on the page; comment says adjust the path for 70B; `--num_fewshot 5 --batch_size auto --limit 250`). Appendix model paths are the in-container `/app/model/...`:

```bash
# FP8 (Per-Tensor)
MODEL=/app/model/Llama-3.1-8B-Instruct/  # Or Llama-3.1-70B-Instruct
lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True,quantization=fp8,kv_cache_dtype=fp8_e4m3 \
  --tasks gsm8k  --num_fewshot 5 --batch_size auto --limit 250

# PTPC FP8
MODEL=/app/model/Llama-3.1-8B-Instruct/  # Or Llama-3.1-70B-Instruct
lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True,quantization=ptpc_fp8,kv_cache_dtype=fp8_e4m3 \
  --tasks gsm8k  --num_fewshot 5 --batch_size auto --limit 250

# BF16
MODEL=/app/model/Llama-3.1-8B-Instruct/  # Or Llama-3.1-70B-Instruct
lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True,kv_cache_dtype=auto \
  --tasks gsm8k  --num_fewshot 5 --batch_size auto --limit 250
```

The PTPC / standard-FP8 eval lines also set `kv_cache_dtype=fp8_e4m3`. That is the **KV** dtype, not PTPC itself. See [fp8-kvcache.md](fp8-kvcache.md).
