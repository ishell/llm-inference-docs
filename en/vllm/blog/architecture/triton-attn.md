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

Local figures (copyright remains with the original site; study copies):

![image1](../../../../assets/vllm/blog/architecture/triton-attn/01-image1.png)

![image2](../../../../assets/vllm/blog/architecture/triton-attn/02-image2.png)

![image3](../../../../assets/vllm/blog/architecture/triton-attn/03-image3.png)

![image4](../../../../assets/vllm/blog/architecture/triton-attn/04-image4.png)

![image5](../../../../assets/vllm/blog/architecture/triton-attn/05-image5.png)

![image6](../../../../assets/vllm/blog/architecture/triton-attn/06-image6.png)

![image7](../../../../assets/vllm/blog/architecture/triton-attn/07-image7.png)

![image8](../../../../assets/vllm/blog/architecture/triton-attn/08-image8.png)

![image9](../../../../assets/vllm/blog/architecture/triton-attn/09-image9.png)

![image10](../../../../assets/vllm/blog/architecture/triton-attn/10-image10.png)
