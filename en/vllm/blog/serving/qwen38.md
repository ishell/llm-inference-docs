---
source: https://vllm.ai/blog/2026-08-12-qwen3.8
lang: en
fetched: 2026-09-04
---

# Qwen3.8-2.4T day-0: Max-class open weights, same Qwen 3.5 skeleton

Chinese: [zh/vllm/blog/serving/qwen38.md](../../../../zh/vllm/blog/serving/qwen38.md)

2026-08-12. **vLLM Team and Inferact**. Demo numbers. First Qwen-Max-class open weights: `Qwen3.8-2.4T-A95B`. Same Qwen 3.5 skeleton — 512-expert sparse MoE, full attention every 4th of 92 layers, 69 linear-attention layers. **No new engine architecture.** Predecessor hybrid: [qwen3-next.md](qwen3-next.md). GDN + P/D sequel: [qwen35-25k-tps.md](qwen35-25k-tps.md). GSM8K / AIME25 are the page’s checks, not your SLA.

**Figure (social preview; not scraped; caption from the page).** `/assets/figures/2026-08-12-qwen3.8/social-preview.png`.

**TL;DR from the page:**

- **Day-0:** reuses Qwen 3.5 architecture; runs on vLLM from day one; no architecture changes.
- **Precision:** official FP8 / BF16; Inferact NVFP4 / MXFP4.
- **Hardware:** at least two NVIDIA B300 / AMD MI355X nodes; FP4 can fit one node.
- NVIDIA + AMD kernel work on the existing Qwen 3.5 path.

## Why this model

Qwen3.8-2.4T-A95B is billed as the first Qwen-family model to bring Qwen-Max-class weights into the open. Built on Qwen 3.5; runs on vLLM out of the box. Inferact MXFP4 / NVFP4: claimed full-precision quality at lower memory and bandwidth.

2.4-trillion-parameter sparse MoE, **512** experts. 92-layer hybrid backbone: full attention every 4th layer; remaining **69** layers linear attention. Inference: ≥2× B300 / MI355X, or a single node for FP4.

## Quick start

NVFP4 (`--linear-backend flashinfer_cutedsl`, `--tensor-parallel-size 8`, MTP 3):

```bash
# See recipes for the exact docker run command
vllm serve Inferact/Qwen3.8-2.4T-A95B-NVFP4 \
  --linear-backend flashinfer_cutedsl \
  --tensor-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

MXFP4 (same parsers / MTP; **no** `flashinfer_cutedsl`):

```bash
vllm serve Inferact/Qwen3.8-2.4T-A95B-MXFP4 \
  --tensor-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Full serving guide: [recipes.vllm.ai](https://recipes.vllm.ai/).

## FP4 quantization: quality at lower cost

Inferact quantized selected layers — including routed experts — to FP4 via Round-to-Nearest (RTN) plus activation calibration (4-bit activations). Initial checks: quantization accuracy intact. **Increasing the reasoning budget is required** to reproduce the evals.

| Benchmark | FP8 | NVFP4 |
| :--- | :--- | :--- |
| GSM8K (strict / flexible) | 89.61% / 90.52% | 90.37% / 91.05% |
| AIME25 @3 (avg / pass) | 87.78% / 93.33% | 92.22% / 96.67% |

Quantization is not the accuracy story here; the budget is.

## Optimizations

Kernels built on existing Qwen 3.5 support.

**NVIDIA.** NVIDIA and Inferact: Linear Attention (Gated Delta Rule), Attention (GQA), Dense GEMMs, MoE routing. New fused kernels to cut communication. Work decomposition: DP+TP for Attention, Expert Parallelism for MoE.

**AMD Instinct.** AITER-fused Gated DeltaNet decode, attention, and MoE — less kernel-launch and data-movement overhead. Shared-expert path: hipBLASLt GEMM. Routed experts: AITER FusedMoE. AMD Quark: MXFP4, less memory, claimed strong accuracy.

## Deployment tips

Model-card sampling:

```
temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0
```

Reasoning model: give agentic workflows a large `max_tokens`. Client from the page (`timeout=3600`, `max_tokens=128_000`; `model="Qwen/Qwen3.8-2.4T-A95B"` while serve uses Inferact FP4 slugs):

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=3600)

resp = client.chat.completions.create(
    model="Qwen/Qwen3.8-2.4T-A95B",
    messages=[{"role": "user", "content": "Give me three primes above 100."}],
    temperature=1.0, top_p=0.95, max_tokens=128_000,
)
print(resp.choices[0].message.content)
```

## Acknowledgements

Qwen team (weights + ongoing collaboration). NVIDIA and AMD (joint kernels). Inferact (quantized checkpoints + e2e vLLM). Broader vLLM community. Inference partners named for early testing: DigitalOcean, Together AI.
