---
source: https://vllm.ai/blog/2026-02-26-multi-lora
lang: en
fetched: 2026-09-01
---

# Multi-LoRA on MoE: many adapters, one GPU

Chinese: [zh/vllm/blog/serving/multi-lora.md](../../../../zh/vllm/blog/serving/multi-lora.md)  
vLLM ≥0.15.0. GPT-OSS 20B as the running example. SageMaker / Bedrock have extra tuned configs.

Freeze the base; swap LoRA per request. Five customers at 10% each need one GPU, not five. Each MoE expert’s `gate_up` / `down` gets shrink+expand — **four skinny GEMMs per expert per adapter** (r typically 16–64, 100–300× smaller than hidden). Dense Multi-LoRA kernels do not know expert routing, so `fused_moe_lora` adds an extra grid dim for active adapters.

First cut: TTFT ~**10×** worse than the base — Triton specialized on input-length vars and recompiled every context length. `do_not_specialize` reuses the binary. Then Split-K, CTA swizzle, `EVEN_K` to skip masks, fuse base add into expand. Open-source path: 144 OTPS / 135 ms TTFT; AWS-tuned: 171 OTPS / 124 ms TTFT (1600/600, rank 32, 8 adapters). Vs 0.11.1rc3: OTPS +454%, TTFT −87%. Dense Qwen3-32B OTPS ~+99%. Custom Triton configs: vLLM LoRA Tuning docs. Their load, not your SLA.

Local figures (copyright remains with the original site; study copies):

![moe schematic](../../../../assets/vllm/blog/serving/multi-lora/01-moe_schematic.png)

![exec opt](../../../../assets/vllm/blog/serving/multi-lora/02-exec_opt.png)

![otps](../../../../assets/vllm/blog/serving/multi-lora/03-otps.png)
