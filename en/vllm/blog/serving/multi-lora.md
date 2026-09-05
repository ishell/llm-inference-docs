---
source: https://vllm.ai/blog/2026-02-26-multi-lora
lang: en
fetched: 2026-09-04
---

# Multi-LoRA on MoE: many adapters, one GPU

Chinese: [zh/vllm/blog/serving/multi-lora.md](../../../../zh/vllm/blog/serving/multi-lora.md)

2026-02-26. **AWS AI Team** (Danielle Maddix Robinson, Florian Saupe, George Novack, Haipeng Li, Mani Kumar Adari, Xiang Song, Yu Gong). vLLM ≥**0.15.0**. Running example: [gpt-oss](gpt-oss.md) 20B. SageMaker / Bedrock have extra tuned configs. Also on AWS Blogs. 1600/600, rank 32, 8 adapters — **their** load, not your SLA.

**TL;DR from the page:**

- Freeze the base; swap LoRA per request. Five customers at 10% each → one GPU, not five.
- MoE families: GPT-OSS, Qwen3-MoE, DeepSeek, Llama MoE. Dense also improved (Llama3.3 70B, Qwen3 32B).
- First cut: TTFT ~**10×** worse than the base. `do_not_specialize` reused the Triton binary.
- Open-source path: **144 OTPS** / **135 ms TTFT**. AWS-tuned: **171 OTPS** / **124 ms TTFT**. vs 0.11.1rc3: OTPS **+454%**, TTFT **−87%**. Dense Qwen3-32B OTPS ~**+99%**.

## Why Multi-LoRA

Idle GPUs when each custom model cannot saturate its own endpoint. Multi-LoRA: keep original weights frozen, inject small trainable adapters, swap adapters per request on a shared GPU.

Amazon-specific extras over vLLM 0.15.0 for GPT-OSS 20B: **19%** higher OTPS, **8%** lower TTFT — on [SageMaker AI](https://aws.amazon.com/sagemaker/ai/) or [Bedrock](https://aws.amazon.com/bedrock/).

## Implementing multi-LoRA inference for MoE models in vLLM

MoE: specialized experts; a router sends each token to the relevant ones; sparse — only a fraction of parameters fire. Each expert is a small FFN in two stages:

- `gate_up` expands hidden (e.g. **4096**) into intermediate (e.g. **11008**) — room to disentangle and gate.
- `down` compresses back. Bottleneck: keep only useful features.

vLLM `fused_moe`: those projections as Group GEMM — one GEMM per expert assigned to a token.

LoRA: freeze `W` (e.g. `W_gate_up`); train `A` (`h_in × r`) and `B` (`r × h_out`); `y = xW + xAB`. Rank `r` typically **16–64**. Shrink: `z = xA` (`h_in → r`). Expand: `z B` (`r → h_out`).

Local figures (copyright remains with the original site; study copies):

![moe schematic](../../../../assets/vllm/blog/serving/multi-lora/01-moe_schematic.png)

**Figure 1.** MoE-LoRA with hidden 4096, intermediate 11008, LoRA rank `r = 32`.

Each expert has `gate_up` and `down`. Each adapter adds shrink+expand to **each** projection → **four** LoRA kernel ops per expert per adapter per request. One dimension (`r`) is **100–300×** smaller than hidden / intermediate. Square GEMM kernels hate skinny matrices.

Two extra problems: (1) dense Multi-LoRA kernels do not know expert routing; (2) compound sparsity — expert routing **and** adapter selection. Fix: `fused_moe_lora` inside `fused_moe`. Same logic as `fused_moe`, plus an extra grid dimension for the active adapters.

## Improving multi-LoRA inference performance in vLLM

Nsight Systems: `fused_moe_lora` was the highest-latency piece. Nsight Compute: profile `gate_up_shrink`, `gate_up_expand`, `down_shrink`, `down_expand`. Then execution opts, kernel opts, tuned configs.

### Execution optimizations

Initial Multi-LoRA TTFT ~**10×** worse than the public GPT-OSS 20B base. Triton treated input-length-dependent variables as compile-time constants → recompile `fused_moe_lora` for every new context length.

![exec opt](../../../../assets/vllm/blog/serving/multi-lora/02-exec_opt.png)

**Figure 2.** Before the fix: `cuModuleLoadData` before each `fused_moe_lora` (new binary, not cache); large gaps = GPU idle during recompile. That idle drove the 10× TTFT. Fix: `do_not_specialize` — compile once, reuse across context lengths.

### Kernel optimizations

**Split-K.** LoRA shrink is `xA` with `x` `1×h_in`, `A` `h_in×r`. Each of `r` outputs sums `h_in` multiplies. Standard GEMM parallelizes across outputs; each thread group still walks `h_in` sequentially. Split-K splits K (`h_in`) across thread groups; partial sums combine with atomic add. Pure add, no extra logic → `sem="relaxed"`.

**CTA swizzling** on `lora_shrink`. Neighboring columns of `A` share rows / cache lines. Reorder so nearby columns run together → more L2 reuse.

**EVEN_K.** Triton loads fixed-size blocks; leftover K needs masks on every load. `EVEN_K` is true when K divides `BLOCK_SIZE_K` — skip masks and extra dots.

**Fuse** LoRA + base add into the expand kernel — one less launch.

After these: **144 OTPS** / **135 ms TTFT** for GPT-OSS 20B (open-source path).

### Tuning kernel configurations for Amazon SageMaker AI and Amazon Bedrock

Triton knobs: `BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M` (cache locality), `SPLIT_K`. Defaults from standard fused MoE ignored the extra LoRA-index grid dim and adapter sparsity. Users can load a custom folder; see vLLM LoRA Tuning docs. Four ops tuned together (they share `BLOCK_SIZE_M`). SageMaker / Bedrock load these automatically → **171 OTPS** / **124 ms TTFT** for GPT-OSS 20B.

## Results & conclusion

Open-sourced Multi-LoRA for GPT-OSS, Qwen3 MoE, DeepSeek, Llama MoE. vs vLLM 0.11.1rc3 → 0.15.0: OTPS **+454%**, TTFT **−87%** on GPT-OSS 20B. Kernel tuning + CTA swizzle also helped dense: Qwen3 32B OTPS **+99%**. Local: ≥0.15.0. Amazon extras: **19%** OTPS / **8%** TTFT vs 0.15.0 on that same model. Hosting docs: [SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html), [Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/fine-tuning-openai-apis.html).

![otps](../../../../assets/vllm/blog/serving/multi-lora/03-otps.png)

**Figure 3.** OTPS and TTFT for GPT-OSS 20B Multi-LoRA: (1) initial 0.11.1rc3; (2) 0.15.0; (3) 0.15.0 + AWS custom kernel tuning. **1600** input / **600** output, rank **32**, **8** adapters in parallel.

Also published on [AWS Blogs](https://aws.amazon.com/blogs/machine-learning/efficiently-serve-dozens-of-fine-tuned-models-with-vllm-on-amazon-sagemaker-ai-and-amazon-bedrock/).

## Acknowledgments

vLLM community: Jie Li, Chen Wu, Varun Sundar Rabindranath, Simon Mo, Robert Shaw. AWS: Xin Yang, Sadaf Fardeen, Ashish Khetan, George Karypis.
