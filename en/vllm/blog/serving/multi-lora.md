---
source: https://vllm.ai/blog/2026-02-26-multi-lora
lang: en
fetched: 2026-09-05
---

# Efficiently serve dozens of fine-tuned models with vLLM on Amazon SageMaker AI and Amazon Bedrock

Chinese: [zh/vllm/blog/serving/multi-lora.md](../../../../zh/vllm/blog/serving/multi-lora.md)  
Source: https://vllm.ai/blog/2026-02-26-multi-lora

2026-02-26. **AWS AI Team** (Danielle Maddix Robinson, Florian Saupe, George Novack, Haipeng Li, Mani Kumar Adari, Xiang Song, Yu Gong). Study rewrite, not an official reprint. vLLM ≥**0.15.0**. Running example: [gpt-oss](gpt-oss.md) 20B. SageMaker AI / Bedrock ship extra tuned configs. Also on AWS Blogs. 1600/600, rank 32, 8 adapters — **their** load, not your SLA.

**TL;DR from the page:**

- Freeze the base; swap LoRA per request. Five customers at 10% each → one GPU, not five.
- MoE families: GPT-OSS, Qwen3-MoE, DeepSeek, Llama MoE. Dense models also improved (Llama3.3 70B, Qwen3 32B).
- First cut: TTFT ~**10×** worse than the public base. `do_not_specialize` reused the Triton binary across context lengths.
- Open-source path: **144 OTPS** / **135 ms TTFT**. AWS-tuned: **171 OTPS** / **124 ms TTFT**. vs 0.11.1rc3: OTPS **+454%**, TTFT **−87%**. Dense Qwen3-32B OTPS ~**+99%**.

Original sections: Implementing multi-LoRA inference for MoE models in vLLM → Improving multi-LoRA inference performance in vLLM (Execution optimizations / Kernel optimizations / Tuning kernel configurations for Amazon SageMaker AI and Amazon Bedrock) → Results & Conclusion → Acknowledgments.

Teams that host many custom models — especially recent Mixture of Experts (MoE) families — pay for idle GPUs when no single model saturates its own endpoint. AWS and the vLLM community built Multi-Low-Rank Adaptation (Multi-LoRA) serving for open-source MoE models such as GPT-OSS and Qwen.

Multi-LoRA fine-tunes without retraining the full weights: the original weights stay frozen, and small trainable adapters are injected into the layers. At inference, many custom models share one GPU and only the adapters swap per request. Five customers each using 10% of a dedicated GPU can share one card.

The post walks through the vLLM MoE implementation, then the kernel work, then how to use it. GPT-OSS 20B is the running example.

Use **0.15.0** or later locally. Multi-LoRA serving covers GPT-OSS, Qwen3-MoE, DeepSeek, and Llama MoE. The same work also helps dense models (Llama3.3 70B, Qwen3 32B). Amazon-specific extras over vLLM 0.15.0: **19%** higher OTPS (Output Tokens Per Second) and **8%** lower TTFT (Time To First Token) on GPT-OSS 20B, if the LoRA-customized models run on [Amazon SageMaker AI](https://aws.amazon.com/sagemaker/ai/) or [Amazon Bedrock](https://aws.amazon.com/bedrock/).

## Implementing multi-LoRA inference for MoE models in vLLM

MoE and LoRA background first, because the kernel choices follow from it.

An MoE has specialized networks called experts. A router sends each input token to the relevant experts and aggregates their outputs. Sparse: only a fraction of parameters fire per token. Figure 1.

Each expert is a small feed-forward network in two stages. `gate_up` expands a compact hidden state (e.g. **4096**) into a larger intermediate space (e.g. **11008**). Features in the compact space are entangled; the larger space gives room to disentangle, transform, and gate. `down` compresses back so the output matches the rest of the model, and acts as a bottleneck that keeps only useful features. Expand-then-compress: rich transforms, consistent output size.

vLLM’s `fused_moe` kernel runs those projections as Group GEMM — one GEMM per expert assigned to a token.

Multi-LoRA freezes base `W` (e.g. `W_gate_up`) and trains `A` and `B`. For a projection `h_in × h_out`, LoRA trains `A` (`h_in × r`) and `B` (`r × h_out`), rank `r` typically **16–64**. The fine-tuned output is `y = xW + xAB`. Each adapter adds two ops: shrink `z = xA` (`h_in → r`), expand `z B` (`r → h_out`). Right side of Figure 1.

Local figures (copyright remains with the original site; study copies):

![moe schematic](../../../../assets/vllm/blog/serving/multi-lora/01-moe_schematic.png)

**Figure 1.** MoE-LoRA with hidden 4096, intermediate 11008, LoRA rank `r = 32`.

Each expert has `gate_up` and `down`. Each adapter adds shrink+expand to **each** projection → **four** LoRA kernel ops per expert (`gate_up` shrink/expand, `down` shrink/expand). In multi-LoRA serving those four ops run per expert, per adapter, per request — the bottleneck.

One dimension (`r`) is **100–300×** smaller than hidden / intermediate. Square GEMM kernels hate skinny matrices, which is why the later kernel work exists.

Two further problems: (1) there was no MoE-layer LoRA kernel — dense Multi-LoRA kernels do not handle expert routing; (2) compound sparsity — expert routing **and** adapter selection. Fix: `fused_moe_lora` inside `fused_moe`. Same logic as `fused_moe`, plus an extra grid dimension for the active adapters. It runs LoRA shrink and expand GEMMs for `gate_up` and `down`.

## Improving multi-LoRA inference performance in vLLM

After the first implementation, NVIDIA Nsight Systems (Nsys) found `fused_moe_lora` as the highest-latency piece. NVIDIA Nsight Compute (NCU) then profiled compute and memory throughput on `gate_up_shrink`, `gate_up_expand`, `down_shrink`, and `down_expand`. That led to execution optimizations, kernel optimizations, and tuned configs for those four kernels.

### Execution optimizations

Initial Multi-LoRA TTFT was ~**10×** worse than the public GPT-OSS 20B base. Triton treated input-length-dependent variables as compile-time constants, so `fused_moe_lora` recompiled from scratch for every new context length. Figure 2: `cuModuleLoadData` before each `fused_moe_lora` means a newly compiled binary, not a cache hit; large gaps between kernel starts are GPU idle during recompile. That idle drove the 10× TTFT. Fix: `do_not_specialize` on those variables — compile once, reuse across context lengths.

![exec opt](../../../../assets/vllm/blog/serving/multi-lora/02-exec_opt.png)

**Figure 2.** Profiling of `fused_moe_lora` before the execution optimizations.

### Kernel optimizations

**Split-K** is a work-decomposition for skinny matrices. LoRA shrink is `xA` with `x` `1×h_in` and `A` `h_in×r`. Each of `r` outputs sums `h_in` multiplies. Standard GEMM assigns thread groups — batches of GPU threads that share fast on-chip memory — to different outputs, but each group still walks `h_in` sequentially. With `r` in the tens and `h_in` in the thousands, there are few outputs to parallelize and a long sequential sum each. Split-K splits the inner dimension `K` (`K = h_in`) across thread groups; they compute partials in parallel and combine. Partials need an atomic add. Pure add, no extra logic → `sem="relaxed"` on the atomic, giving Triton room to optimize.

The scheduler can assign several thread groups to the same output and run groups for different outputs together. For `lora_shrink`, each output reads one column of `A` spanning `h_in` rows. With `h_in` in the thousands, each column touches cache lines over a large region. Nearby columns share rows and overlap in cache, so groups on neighboring columns can reuse loaded data. Cooperative Thread Array (CTA) swizzling reorders the schedule so those neighboring columns run together → more L2 reuse. Applied to `lora_shrink`.

They also dropped unnecessary masks and dots from shrink and expand. Triton loads fixed-size blocks; matrix dims may not divide the block. If `BLOCK_SIZE_K` is 64 and K is 100, the second block would read 28 invalid locations. Masking checks bounds on every load — overhead even when the element is valid. `EVEN_K` is true when K divides `BLOCK_SIZE_K`; then every load is in bounds, masking can be skipped, and extra dots go away.

Last: fuse LoRA+base addition into the expand kernel — one less launch. After these: **144 OTPS** / **135 ms TTFT** for GPT-OSS 20B.

### Tuning kernel configurations for Amazon SageMaker AI and Amazon Bedrock

Triton knobs: `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K` (how the GEMM is tiled across thread groups). Advanced: `GROUP_SIZE_M` (thread-group order / cache locality) and `SPLIT_K` (parallelize the inner-dimension sum).

Defaults from standard fused MoE performed poorly for multi-LoRA: they ignored the extra LoRA-index grid dimension and adapter sparsity. Users can load a custom folder; see the vLLM LoRA Tuning docs. The four `fused_moe_lora` ops (`gate_up_shrink`, `gate_up_expand`, `down_shrink`, `down_expand`) were tuned together because they share `BLOCK_SIZE_M`. SageMaker AI and Bedrock load these automatically → **171 OTPS** / **124 ms TTFT** for GPT-OSS 20B.

## Results & Conclusion

Open-sourced Multi-LoRA for GPT-OSS, Qwen3 MoE, DeepSeek, Llama MoE. vs vLLM 0.11.1rc3 → 0.15.0: OTPS **+454%**, TTFT **−87%** on GPT-OSS 20B. Kernel tuning and CTA swizzle also helped dense models: Qwen3 32B OTPS **+99%**. Local: ≥0.15.0. Amazon extras on Bedrock and SageMaker AI: **19%** OTPS / **8%** TTFT vs 0.15.0 on that same model. Hosting docs: [SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html), [Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/fine-tuning-openai-apis.html).

![otps](../../../../assets/vllm/blog/serving/multi-lora/03-otps.png)

**Figure 3.** OTPS and TTFT for GPT-OSS 20B Multi-LoRA: (1) initial implementation in 0.11.1rc3; (2) vLLM 0.15.0; (3) 0.15.0 + AWS custom kernel tuning. **1600** input / **600** output tokens, LoRA rank **32**, **8** adapters loaded in parallel.

Also published on [AWS Blogs](https://aws.amazon.com/blogs/machine-learning/efficiently-serve-dozens-of-fine-tuned-models-with-vllm-on-amazon-sagemaker-ai-and-amazon-bedrock/).

## Acknowledgments

vLLM community: Jie Li, Chen Wu, Varun Sundar Rabindranath, Simon Mo, Robert Shaw. AWS: Xin Yang, Sadaf Fardeen, Ashish Khetan, George Karypis.
