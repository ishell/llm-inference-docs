---
source: https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive
lang: en
fetched: 2026-09-01
---

# Triton attention backend

2026-03-04. IBM Research / Red Hat / AMD office hours. Kernel: `triton_unified_attention.py` (~800 LOC vs FA3 ~70k). Paper: *The Anatomy of a Triton Attention Kernel*. Study note.

Always-on fallback: PyTorch+Triton only. Default on ROCm; Intel XPU fp32; ALiBi sqrt, sink tokens, GPT-OSS, small heads, enc/dec, multimodal prefix, batch invariance; pre-Hopper NVIDIA.

Tiles + autotune instead of a kernel zoo. Paged attention: **Q blocks** fatten `tl.dot` (GQA heads + several query tokens). Decode: **3D kernel** splits KV traversal, second kernel reduces (no Triton global barrier). CUDA graphs hate variable grids → **persistent kernels** (fixed launch, work from GPU metadata).

Late-2025: Llama 3.1 8B, bs=1, 500-in. H100 long decode **100.7%** of FA3; MI300 ~**5.8×** vs earlier. Same source. Helion preview. `optimization.md` auto-picks backends; Triton is the portable default, not the slow spare.
