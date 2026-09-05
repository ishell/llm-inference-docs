---
source: https://vllm.ai/blog/2026-02-26-multi-lora
lang: en
fetched: 2026-09-04
---

# Efficiently serve dozens of fine-tuned models with vLLM on Amazon SageMaker AI and Amazon Bedrock

Chinese: [zh/vllm/blog/serving/multi-lora.md](../../../../zh/vllm/blog/serving/multi-lora.md)

2026-02-26. **Danielle Maddix Robinson, Florian Saupe, George Novack, Haipeng Li, Mani Kumar Adari, Xiang Song, Yu Gong (AWS AI Team)**. Also on [AWS Blogs](https://aws.amazon.com/blogs/machine-learning/efficiently-serve-dozens-of-fine-tuned-models-with-vllm-on-amazon-sagemaker-ai-and-amazon-bedrock/). **vLLM ≥ 0.15.0** for the open-source path. Running example: **GPT-OSS 20B**. Cousin: [gpt-oss.md](gpt-oss.md). Numbers are **their** load (1600 in / 600 out, LoRA rank **32**, **8** adapters in parallel) — not your SLA.

Idle GPUs when many custom models each see little traffic. Multi-LoRA: freeze base weights, swap small adapters per request. Pitch: five customers at 10% of a GPU each share **one** GPU.

Works for MoE families: GPT-OSS, Qwen3-MoE, DeepSeek, Llama MoE. Dense also benefits (Llama 3.3 70B, Qwen3 32B). SageMaker AI / Bedrock host extra Amazon-tuned configs: they quote **+19% OTPS** and **−8% TTFT** vs stock vLLM 0.15.0 on GPT-OSS 20B.

Local figures (copyright remains with the original site; study copies):

![moe schematic](../../../../assets/vllm/blog/serving/multi-lora/01-moe_schematic.png)

![exec opt](../../../../assets/vllm/blog/serving/multi-lora/02-exec_opt.png)

![otps](../../../../assets/vllm/blog/serving/multi-lora/03-otps.png)

**Fig 1:** MoE-LoRA with example hidden **4096**, intermediate **11008**, rank **r = 32**.  
**Fig 2:** Nsys before `do_not_specialize` — `cuModuleLoadData` before every `fused_moe_lora`, GPU idle while Triton recompiles.  
**Fig 3:** OTPS / TTFT: (1) first impl on vLLM **0.11.1rc3**; (2) vLLM **0.15.0**; (3) 0.15.0 + AWS kernel tuning.

## Why MoE LoRA is four skinny GEMMs

MoE: router sends each token to a few experts. Each expert is FFN: `gate_up` expands (e.g. 4096 → 11008), `down` compresses back. vLLM `fused_moe` runs those as Group GEMM — one GEMM per expert on that token.

LoRA: freeze `W`, train `A` (`h_in × r`) and `B` (`r × h_out`), `y = xW + xAB`. **Shrink** `z = xA`; **expand** `zB`. Rank typically **16–64**.

Each expert has two projections × shrink+expand = **four LoRA ops per expert per adapter per request**. One dimension (`r`) is **100–300×** smaller than hidden/intermediate. Square GEMM kernels hate that.

Two problems they name:

1. Dense multi-LoRA kernels **do not know expert routing**
2. Compound sparsity: expert routing **and** which adapter the request uses

Fix: `fused_moe_lora` inside `fused_moe`. Same logic, **extra grid dimension = active LoRA adapters**. Shrink/expand GEMMs for `gate_up` and `down`.

## Execution: the 10× TTFT bug

First Multi-LoRA TTFT was **10× worse** than base GPT-OSS 20B. Triton treated **input-length-dependent** variables as compile-time constants → **recompile `fused_moe_lora` for every new context length**. Fig 2: `cuModuleLoadData` + idle gaps.

Fix: Triton `do_not_specialize` on those variables — compile once, reuse across context lengths.

## Kernel work

- **Split-K** on shrink (`xA` is `1×h_in` × `h_in×r`): split the long K reduction across thread groups, atomic add with `sem="relaxed"`
- **CTA swizzling** on `lora_shrink` so neighboring columns of `A` run together (L2 reuse)
- **`EVEN_K`**: skip masking/dot work when `K` divides `BLOCK_SIZE_K`
- Fuse LoRA+base add into the **expand** kernel (fewer launches)

After these: **144 OTPS**, **135 ms TTFT** on GPT-OSS 20B (their load).

## Amazon tuned configs

Default fused-MoE Triton blocks were wrong for multi-LoRA (extra LoRA-index grid + compound sparsity). Users can load a folder of custom configs (vLLM LoRA Tuning docs). They tuned `gate_up_shrink` / `gate_up_expand` / `down_shrink` / `down_expand` together (`BLOCK_SIZE_M` shared). SageMaker / Bedrock load these automatically: **171 OTPS**, **124 ms TTFT**.

## Headline deltas they print

Open-source path, GPT-OSS 20B, 0.15.0 vs 0.11.1rc3: **+454% OTPS**, **−87% TTFT**. Dense Qwen3 32B OTPS **+99%** from some of the same tricks (tuning + CTA swizzle). Amazon extras vs 0.15.0: **+19% OTPS**, **−8% TTFT**.

## Acknowledgements

vLLM: Jie Li, Chen Wu, Varun Sundar Rabindranath, Simon Mo, Robert Shaw. AWS: Xin Yang, Sadaf Fardeen, Ashish Khetan, George Karypis.
