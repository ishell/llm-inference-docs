---
source: https://vllm.ai/blog/2026-08-10-nemotron-3-5-lightning-vllm
lang: en
fetched: 2026-09-04
---

# Nemotron 3.5 Lightning day-0: 30B/3B hybrid MoE for always-on agents

Chinese: [zh/vllm/blog/serving/nemotron-35-lightning.md](../../../../zh/vllm/blog/serving/nemotron-35-lightning.md)

2026-08-10. **NVIDIA Nemotron Team and vLLM Team**. Image in the post: `vllm/vllm-openai:v0.27.1`. Distilled from [Nemotron 3 Ultra](nemotron-3-ultra.md); same hybrid family as [Nemotron 3 Nano](nemotron-3-nano.md) / [Super](nemotron-3-super.md). Earlier 9B hybrid: [Nano 2](nemotron-nano2.md). Multimodal cousins: [Nano 2 VL](nemotron-nano-vl.md), [Nano Omni](nemotron-omni.md). Spark box-level caveats: [dgx-spark.md](dgx-spark.md). Study note; Pareto / PinchBench numbers are the page’s demos, not your SLA.

**TL;DR from the page:**

- Hybrid MoE **30B total / 3B active**, text in / text out, context up to **1 million** tokens.
- Distilled from Nemotron 3 Ultra; trained for popular agent harnesses; post-training allowed.
- Day-0 checkpoints: **BF16** and **NVFP4**. Speculative decoding: **MTP**, **DFlash**, **DSpark**.
- Claimed up to **4×** throughput vs similarly sized open models; PinchBench: complete 10,000 agentic tasks up to **30%** faster at comparable accuracies.
- Architecturally identical to Nemotron 3 except weights and the speculative stack. **Not a new engine.**
- Low-latency serving: **DSpark** on H100, H200, DGX Spark. Max throughput *then*: run **without** speculative decoding.

## Role: the small model in a two-model agent split

Nemotron 3.5 Lightning is billed as a customizable open model for always-on agents: local personal assistants through high-volume datacenter / cloud steps. Strengths named: coding, tool use, instruction following, multi-turn intelligence.

Modern agent platforms split work: a frontier model plans and orchestrates; a smaller model runs the frequent, well-scoped steps. Lightning is that second role without dropping the skills real agent workflows need.

Two practical requirements on the page:

- **Fast execution at scale.** Agents spend most of their time on many small steps. Hybrid MoE (3B of 30B active per token) plus multi-token prediction; claimed up to **4×** throughput vs similarly sized open models.
- **Adaptable agent intelligence.** Org terminology, policies, tools, multi-turn context. Trained for popular agent harnesses; can be post-trained. Named domains: financial and risk automation, cybersecurity investigation, telecommunications operations, retail, local personal assistants.

vLLM exposes an OpenAI-compatible API so existing agent frameworks, local apps, and enterprise automation can attach.

## TL;DR: About Nemotron 3.5 Lightning

- **Architecture:** Hybrid mixture-of-experts
- **Model size:** 30B total, 3B active
- **Context length:** up to 1 million tokens
- **Modalities:** text in, text out
- **Speculative decoding:** Multi-token prediction, DFlash, and DSpark
- **Reasoning:** enable or disable per request; configurable reasoning-token budget
- **Training:** distilled from NVIDIA Nemotron 3 Ultra; trained for popular agent harnesses
- **Customization:** open model, open datasets, post-training on specialized workflows
- **Availability at launch:** BF16 and NVFP4
- **Deployment targets:** NVIDIA DGX Spark, DGX Station, RTX PRO, RTX, NVIDIA Jetson, H100, H200, A100, L40S, B200/GB200, B300/GB300
- **Get started:**
  - Weights: [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16), [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
  - vLLM cookbook: [Nemotron-3.5-Lightning/vllm_cookbook.ipynb](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/vllm_cookbook.ipynb)

## Run high-throughput inference with vLLM

Intended to run across a wide NVIDIA platform range. vLLM is the serving layer: continuous batching, prefix caching, speculative decoding, OpenAI-compatible API.

BF16 is the straightforward baseline. NVFP4 is also at launch for lower-precision inference.

### Install vLLM

```bash
docker pull vllm/vllm-openai:v0.27.1

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --network=host \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.27.1
```

### Serve the model

This command assumes a **1 × H100** setup.

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --async-scheduling \
  --mamba-backend flashinfer \
  --moe-backend humming \
  --linear-backend humming \
  --mamba-ssu-algorithm horizontal \
  --mamba-cache-mode align \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --reasoning-parser nemotron_v3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --host 0.0.0.0 \
  --port 8000
```

OpenAI-compatible client from the page (`temperature=1.0`, `top_p=0.95`, `max_tokens=1024`; reads `choice.message.reasoning` and `choice.message.content`):

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="null",
)

response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Briefly explain: what is vLLM?"},
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=1024,
)

choice = response.choices[0]
print("Reasoning:", choice.message.reasoning)
print("Content:", choice.message.content)
```

## Accelerate long-running agentic workflows with speculative decoding

Three speculators: **MTP**, **DFlash**, **DSpark**. Draft-and-verify; claimed to preserve the target model’s output quality.

- **MTP:** lightweight, model-integrated prediction heads propose several future tokens.
- **DFlash:** a diffusion-based drafter generates an entire candidate block in parallel. Needs a compatible draft checkpoint; configured separately from MTP.
- **DSpark:** confidence-aware, semi-autoregressive drafting; hybrid of autoregressive and diffusion-style. Sits between MTP (fully autoregressive) and DFlash (fully diffusion). Best of the three **on DGX Spark** in the post.

Because the architecture matches Nemotron 3 except weights and the spec stack, most performance work landed in the runtimes. Upstream contributions named:

- **DSpark integration** into vLLM and the Nemotron model definition — three speculators alongside MTP and DFlash.
- **Quantized DSpark draft head:** W4A16 cuts memory and per-step latency without hurting acceptance rate; matters most on memory-constrained parts like DGX Spark.
- **Removal of syncs and async scheduling:** host-device syncs removed from the draft-and-verify loop; async scheduling prepares the next batch while the current one still executes.
- **MoE and linear backend for W4A16:** default Marlin replaced with a Hopper-optimized **Humming** backend; W4A16 GEMM for Nemotron’s non-gated ReLU² MoE, worth roughly **20%** throughput; same recipe extended to dense linear layers.
- **ReplaySSM** for Mamba2 state-space layers; reduces per-step overhead in the recurrent path of the hybrid architecture.

Guidance on the page: low-latency → DSpark on H100, H200, DGX Spark. Maximum throughput *today* → **no** speculative decoding.

### Multi-token prediction

Built-in MTP heads propose future tokens; the target verifies them, cutting sequential generation steps on longer responses.

```bash
vllm serve --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 \
  --speculative_config.method mtp \
  --speculative_config.num_speculative_tokens 3 \
  --speculative_config.moe_backend flashinfer_cutlass \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```

### DFlash

Dedicated diffusion draft model proposes a linear block; the target verifies in parallel.

```bash
vllm serve --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --speculative_config.num_speculative_tokens 3 \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 \
  --speculative_config.model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```

DFlash draft: [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash).

### DSpark

Hybrid speculator; best of the three on DGX Spark per the post.

```bash
vllm serve --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --speculative_config.num_speculative_tokens 3 \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 \
  --speculative_config.model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```

DSpark draft: [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark).

## Local deployment on NVIDIA DGX Spark

Starting configuration for single-user local development (NVFP4 + DSpark; long `cudagraph_capture_sizes` list as published):

```bash
vllm serve --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --compilation_config.cudagraph_capture_sizes '[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 1024, 2048, 4096, 8192]' \
  --speculative_config.num_speculative_tokens 3 \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 \
  --speculative_config.method dspark \
  --speculative_config.model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark
```

**Figure 1 (not scraped; caption from the page).** Pareto chart comparing inference performance of Nemotron 3.5 Lightning using various speculative decoding techniques on NVIDIA DGX Spark. Config: Prefix **32K**, then **10** rounds of **2k** input and **1k** output. Axis numbers are in the chart, not in the prose.

## Deploy on NVIDIA H100

Starting configuration for single-user local development. This recipe has **no** speculative decoding (matches the “max throughput today” guidance) and uses **Humming** for MoE and linear:

```bash
vllm serve --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --moe-backend humming \
  --linear-backend humming \
  --max-num-seqs 256 \
  --trust-remote-code \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --async-scheduling \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --mamba-cache-mode align \
  --mamba-ssu-algorithm horizontal \
  --reasoning-parser nemotron_v3
```

**Figure 2 (not scraped; caption from the page).** Same Pareto comparison on NVIDIA H100 GPUs. Same config: Prefix **32K**, then **10** rounds of **2k** input / **1k** output. No numeric TPS / TTFT table in the text.

## Local deployment on NVIDIA Jetson

Starting configuration for single-user local development. No speculative decoding in this snippet:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --reasoning-parser nemotron_v3 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --mamba-cache-mode align
```

## Leading accuracy and efficiency for specialized agent tasks

Hybrid MoE activates 3B of 30B per token; MTP reduces sequential work. Together: up to **4×** throughput vs similarly sized open models.

Accuracy claim: distilling from Ultra plus agent-harness training → strong agent productivity, coding, tool use, instruction following, and long-context reasoning benchmarks. No per-benchmark score table in the post.

**Figure 3 (not scraped; caption from the page).** Line chart: PinchBench accuracy vs time to complete **10,000** tasks. Claim: leads the efficiency frontier by completing agentic tasks up to **30%** faster at comparable accuracies.

## Summary

Customizable agent intelligence for local, edge, datacenter, and cloud. 30B hybrid MoE, 3B active, up to 1M context, controllable reasoning, speculative generation through MTP or DFlash (summary sentence on the page does **not** name DSpark; the body does).

Day-0 vLLM: OpenAI-compatible stack for local assistants, agent harnesses, specialized enterprise workflows.

Same get-started links as the TL;DR (BF16 / NVFP4 weights + cookbook).

Stay-up-to-date links on the page: [NVIDIA Nemotron](https://developer.nvidia.com/nemotron), NVIDIA AI on [LinkedIn](https://www.linkedin.com/showcase/nvidia-ai/posts/?feedView=all), [X](https://x.com/NVIDIAAIDev), [YouTube](https://www.youtube.com/@NVIDIADeveloper), [Nemotron Discord channel](https://discord.com/channels/1019361803752456192/1407781691698708682) / [invite](https://discord.com/invite/nvidiadeveloper).

## Acknowledgement

NVIDIA: Nirmal Kumar Juluru, Anusha Pant, Amir Klein, Faradawn Yang, Nave Assaf, Ryan Stewart, Alex Steiner, Bita Rouhani.

## FAQs

### What is new compared with the Nemotron 3 Nano?

Nano established the efficient hybrid Mamba-Transformer MoE: 30B total, 3B active, 1M-token context, controllable reasoning. Lightning builds on that. The page says **four** important ways, then lists **three**:

- **Frontier-model distillation:** from Nemotron 3 Ultra into a much smaller deployment footprint.
- **Agent-harness optimization:** popular harnesses and multi-turn workflows; coding, tool use, instruction following, specialized task completion.
- **Speculative decoding:** MTP, DFlash, and DSpark; draft-and-verify multiple tokens in parallel.

The fourth item is not in the fetched post. Result claimed: more agent tasks, more accurately, in less time.
