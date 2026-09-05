---
source: https://vllm.ai/blog/2026-06-04-nemotron-3-ultra-vllm
lang: en
fetched: 2026-09-04
---

# Nemotron 3 Ultra: 550B/55B hybrid MoE; training rollouts also vLLM

Chinese: [zh/vllm/blog/serving/nemotron-3-ultra.md](../../../../zh/vllm/blog/serving/nemotron-3-ultra.md)

2026-06-04. **NVIDIA Nemotron Team**. Image in the post: `vllm/vllm-openai:v0.22.0`. **8 × B200** example. Cookbook is the real recipe; this page is the day-0 skeleton. Distilled-down sibling: [Lightning](nemotron-35-lightning.md). Same family: [Super](nemotron-3-super.md), [Nano](nemotron-3-nano.md), [Nano 2](nemotron-nano2.md), [Nano 2 VL](nemotron-nano-vl.md), [Nano Omni](nemotron-omni.md). Rollout backend sits next to [native-rl.md](native-rl.md). Hybrid Mamba serving internals: [hybrid-ssm.md](hybrid-ssm.md). Marketing **30%** cost / “leading TPS” live on the figures — not a signed SLA.

**Hero (cover; not scraped; caption from the page).** Hero image at `/assets/figures/2026-nemotron-3-ultra/hero.png` on the original post. No numeric claim in the alt/caption.

**TL;DR from the page:**

- Hybrid Transformer-Mamba MoE: **550B** total, **55B** active; context up to **1M** tokens; text in / text out.
- BF16 and NVFP4. NVFP4 checkpoint works on Blackwell; the same NVFP4 checkpoint is also said to run on Hopper **and** Blackwell via specialized NVFP4 quantization kernels.
- vLLM was used in **training**: multi-node inference for rollouts and evaluation; inside NeMo RL as a generation backend, with NeMo Gym for multi-step / multi-turn environments.

## Why this model

[Nemotron 3 Ultra](https://blogs.nvidia.com/blog/nvidia-gtc-taipei-computex-2026-news/#nemotron-3-ultra) is billed as frontier-class reasoning for long-running autonomous agents: complex orchestration, coding, deep research, enterprise automation — plan, call tools, recover from errors, reason across extended context.

Persistent agents do not just answer one prompt. They search, write code, run tests, inspect failures, coordinate tools, evaluate evidence, keep working across long horizons. Need reasoning depth **and** inference fast enough to deploy.

Two requirements:

**Fast Task Completion.** Throughput so the agent finishes more reasoning steps in the same wall-clock budget. Hybrid Transformer-Mamba MoE, multi-token prediction, NVIDIA-optimized inference precision.

**Advanced Agentic Reasoning.** Architectural planning, multi-step debugging, source evaluation, regulatory review, design verification. Post-trained for reasoning, tool use, instruction following across agentic environments.

vLLM in the training loop: high-throughput multi-node inference for rollouts and model evaluation. Within [NeMo RL](https://github.com/nvidia-nemo/rl), vLLM is the generation backend — sampling, scalable inference, integration with [NeMo Gym](https://github.com/NVIDIA-NeMo/gym). The Nemotron team also used vLLM in the evaluation loop that tracked whether each training stage moved the model the right way.

## TL;DR: About Nemotron 3 Ultra

- **Architecture:** Mixture of Experts with Hybrid Transformer-Mamba
  - Model size: 550B total, 55B active
  - Context length: up to 1M tokens
  - Modalities: text in, text out
- **Efficiency:** high-throughput inference with NVFP4 and BF16. NVFP4 checkpoint works on Blackwell GPUs.
- **Reasoning:** long-running autonomous agents, tool calling, coding, deep research, orchestration
- **Training:** post-trained with multi-environment reinforcement learning
- **Deployment:** open weights, open data, open recipes
- **Supported GPUs:**
  - BF16: **8×** GB200/B200/GB300/B300, **16×** H100, **8×** H200
  - NVFP4: **4×** GB200/B200/GB300/B300, **8×** H100
- **Get started:**
  - Weights: [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16), [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4)
  - Cookbook: [Nemotron-3-Ultra/vllm_cookbook.ipynb](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Ultra/vllm_cookbook.ipynb)
  - Technical report: [NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf)

## Run optimized agentic inference with vLLM

BF16 and NVFP4. OpenAI-compatible API for agent frameworks, coding systems, research pipelines, enterprise automation.

Easier setup: the cookbook, or the NVIDIA Brev [launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-3EPQRUP8Sl27sxp1fMvXt3Lor8T) for NVFP4.

### Install vLLM

```bash
docker pull vllm/vllm-openai:v0.22.0

docker run --rm -it --gpus all --ipc=host --network=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.22.0
```

### Serve the model

Configured for **8× B200**. If hardware differs, adjust parallelism and related flags. Cookbook for NVFP4 guidance.

```bash
export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

vllm serve nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \
  --served-model-name nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 16 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 32768 \
  --enable-flashinfer-autotune \
  --async-scheduling \
  --speculative_config.method mtp \
  --speculative_config.num_speculative_tokens 5 \
  --mamba-backend triton \
  --mamba-ssm-cache-dtype float32 \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

Caveats in this snippet vs the TL;DR: `--max-model-len 262144` is **not** the 1M context in the spec sheet; `--mamba-backend triton` here, Lightning’s H100 recipe uses `flashinfer`. MTP `num_speculative_tokens` is **5** (Lightning’s MTP example uses **3**).

OpenAI-compatible client (`api_key="EMPTY"`; reads `getattr(msg, "reasoning", None)`):

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

resp = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Give me 3 bullet points about vLLM"},
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=1024,
)

msg = resp.choices[0].message
print("Reasoning:", getattr(msg, "reasoning", None))
print("Content:", msg.content)
```

NVFP4 deployment: [cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Ultra/vllm_cookbook.ipynb).

## High-throughput reasoning for long-running agents

Figures 1–3: leads on accuracy for agent productivity, instruction following, and long-context tasks; leading throughput; saves **30%** on costs vs other leading open models. No numeric TPS / TTFT table in the prose.

Local figures (copyright remains with the original site; study copies):

![figure1](../../../../assets/vllm/blog/serving/nemotron-3-ultra/02-figure1.svg)

![figure2](../../../../assets/vllm/blog/serving/nemotron-3-ultra/03-figure2.svg)

![figure3](../../../../assets/vllm/blog/serving/nemotron-3-ultra/04-figure3.svg)

**Figure 1.** Nemotron 3 Ultra leads among open models on agentic benchmarks for agent productivity, coding, and instruction following.

**Figure 2.** Most attractive quadrant: leading accuracy and leading throughput among open models. Config: vLLM with **10k/2k** ISL/OSL, **BS 1**.

**Figure 3.** Saves up to **30%** in costs and leads on the cost-efficiency frontier.

Architectural mitigations of the usual efficiency–accuracy tradeoff:

- **Post-Trained for Agent Harness.** [NeMo RL](https://github.com/nvidia-nemo/rl) and [Gym](https://github.com/NVIDIA-NeMo/gym) across many harnesses. Optimized for leading open agent harnesses, not just single-turn chat: plan, call tools, read observations, delegate to sub-agents, validate outputs, recover from errors across many turns.
- **Hybrid Mamba-Transformer.** Mamba for sequence efficiency on long context; Transformer for precise recall when agents must pull specific facts from large windows.
- **Latent MoE.** More efficient expert routing across reasoning, code, tool calls, domain logic.
- **Multi-Token Prediction (MTP).** Several future tokens in one forward pass; throughput on long outputs and multi-turn workflows.
- **NVFP4 precision.** Same NVFP4 checkpoint on Hopper and Blackwell via specialized NVFP4 quantization kernels.

## Summary

Open frontier reasoning model for long-running autonomous agents: high-throughput inference, long-context reasoning, tool use, open deployment.

Same get-started trio: BF16 / NVFP4 weights, cookbook, technical report.

Stay-up-to-date: [NVIDIA Nemotron](https://developer.nvidia.com/nemotron), NVIDIA AI on [LinkedIn](https://www.linkedin.com/showcase/nvidia-ai/posts/?feedView=all), [X](https://x.com/NVIDIAAIDev), [YouTube](https://www.youtube.com/@NVIDIADeveloper), [Nemotron Discord channel](https://discord.com/channels/1019361803752456192/1407781691698708682) / [invite](https://discord.com/invite/nvidiadeveloper).

## Acknowledgement

Thanks to everyone who contributed to bringing NVIDIA Nemotron 3 Ultra to vLLM.

NVIDIA: Nirmal Kumar Juluru, Anusha Pant, Alex Steiner, Tomer Asida, Daniel Afrimi, Shaun Kotek, Roi Koren, Daniel Serebrenik, Amir Klein, Omer Ullman Argov, Netanel Haber, Amit Zuker, Shahar Mor, Tomer Bar Natan.

vLLM team and community: Michael Goin, Kaichao You, Yongye Zhu, Roger Wang, Simon Mo, Woosuk Kwon, Yasong Wang, Nick Hill, Zachary Xi.
