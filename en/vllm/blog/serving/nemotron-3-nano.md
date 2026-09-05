---
source: https://vllm.ai/blog/2025-12-15-run-nvidia-nemotron-3-nano
lang: en
fetched: 2026-09-04
---

# Nemotron 3 Nano: 30B/3B hybrid MoE, Thinking Budget, NVFP4 later

Chinese: [zh/vllm/blog/serving/nemotron-3-nano.md](../../../../zh/vllm/blog/serving/nemotron-3-nano.md)

2025-12-15. **NVIDIA Nemotron Team**. Day-0 how-to, not kernel depth. **1M** context. Install then: `git+https://github.com/vllm-project/vllm.git@main` with `VLLM_USE_PRECOMPILED=1`. Larger: [Super](nemotron-3-super.md) / [Ultra](nemotron-3-ultra.md). Distilled later: [Lightning](nemotron-35-lightning.md). Predecessor 9B: [Nano 2](nemotron-nano2.md). Multimodal cousins: [Nano 2 VL](nemotron-nano-vl.md), [Nano Omni](nemotron-omni.md). Spark: [dgx-spark.md](dgx-spark.md). Hybrid Mamba: [hybrid-ssm.md](hybrid-ssm.md). **4×** charts are the page’s demos, not your SLA.

**Jan 28th Update.** NVFP4 checkpoint, supported out of the box. Quantization-Aware Distillation (QAD) to keep NVFP4 accuracy; claimed **4×** throughput on B200 vs FP8-H100. Weights: [NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4). Brev: [launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-386KFyCvmg3y22JIf0q8BUh6jia).

**TL;DR from the page:**

- Hybrid Mamba-Transformer MoE: **30B** total, **3B** active; context **1M**; text in / text out.
- vs [Nano 2](nemotron-nano2.md): FFN → sparse MoE; most attention → Mamba-2. Claimed up to ~**4×** token throughput.
- Day-0: BF16 and FP8. NVFP4 + QAD in the January addendum.
- Then `--reasoning-parser deepseek_r1` (later Nemotron 3 posts often `nemotron_v3`). `VLLM_ATTENTION_BACKEND=FLASHINFER`.
- `--tool-call-parser qwen3_coder`. Thinking Budget.

## Why this model

Nemotron 3 Nano is billed as the small efficient open model in the then-new Nemotron 3 family, for agentic AI. Hybrid Mamba-Transformer MoE and 1M context: reliable, high-throughput agents across multi-document and long-duration work.

Fully open: weights, datasets, recipes — customize and deploy on your own infra.

Local figures (copyright remains with the original site; study copies):

![figure 1](../../../../assets/vllm/blog/serving/nemotron-3-nano/01-figure_1.png)

**Figure 1.** “NVIDIA Nemotron 3 Sets a New Standard for Open Source AI.” Chart: most attractive quadrant on Artificial Analysis Openness vs Intelligence Index.

Named strengths: coding, reasoning, agentic tasks. Named benches: SWE Bench Verified, GPQA Diamond, AIME 2025, Arena Hard v2, IFBench. No numeric score table in the prose.

## About Nemotron 3 Nano

- **Architecture:**
  - Mixture of Experts (MoE) with Hybrid Transformer-Mamba
  - Thinking Budget: claimed optimal accuracy with minimum reasoning-token generation
- **Accuracy:** leading on coding, scientific reasoning, problem solving, math, instruction following, chat
- **Model size:** 30B with 3B active
- **Context length:** 1M
- **Model input:** text
- **Model output:** text
- **Supported GPUs:** NVIDIA RTX Pro 6000, DGX Spark, H100, B200
- **Get started:**
  - Weights: [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16), [FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)
  - [Run with vLLM](https://brev.nvidia.com/launchable/deploy?launchableID=env-36ikINrMffBCbrtTVLr6MFcllcs)
  - Technical report: [NVIDIA-Nemotron-3-Nano-Technical-Report.pdf](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)

## Run optimized inference with vLLM

BF16 and FP8. OpenAI-compatible API.

### Install vLLM

Then: install from `main`, precompiled:

```shell
VLLM_USE_PRECOMPILED=1 pip install git+https://github.com/vllm-project/vllm.git@main
```

### Serve the model

`VLLM_ATTENTION_BACKEND=FLASHINFER`. Two equivalent BF16 launches (`vllm serve` or `python -m vllm.entrypoints.openai.api_server`). Port **5000**.

Serve path on the page is `nvidia/NVIDIA-Nemotron-Nano-3-30B-A3B-BF16` — **Nano-3**, not the Hugging Face slug `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`. Keep both as published.

```bash
export VLLM_ATTENTION_BACKEND=FLASHINFER

# BF16
vllm serve --model "nvidia/NVIDIA-Nemotron-Nano-3-30B-A3B-BF16" \
    --dtype auto \
    --trust-remote-code \
    --served-model-name nemotron \
    --host 0.0.0.0 \
    --port 5000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser deepseek_r1
```

Or:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model "nvidia/NVIDIA-Nemotron-Nano-3-30B-A3B-BF16" \
    --dtype auto \
    --trust-remote-code \
    --served-model-name nemotron \
    --host 0.0.0.0 \
    --port 5000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser deepseek_r1
```

FP8 swap (page uses `--reasoning_parser` with an underscore):

```bash
vllm serve --model "nvidia/NVIDIA-Nemotron-Nano-3-30B-A3B-FP8" \
    --dtype auto \
    --trust-remote-code \
    --served-model-name nemotron \
    --host 0.0.0.0 \
    --port 5000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning_parser deepseek_r1
```

Client (`api_key="null"`; prints `reasoning_content` then `content`):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:5000/v1", api_key="null")

# Simple chat completion
resp = client.chat.completions.create(
    model="nemotron",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about GPUs."}
    ],
    temperature=0.7,
    max_tokens=256,
)
print(resp.choices[0].message.reasoning_content, resp.choices[0].message.content)
```

Easier setup: [Brev cookbook launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-36ikINrMffBCbrtTVLr6MFcllcs).

## Highly efficient with leading accuracy for agentic tasks

Builds on [Nano 2](nemotron-nano2.md)’s hybrid Mamba-Transformer: FFN layers become sparse MoE; most attention layers become Mamba-2. MoE: better accuracy at a fraction of the active-parameter count; lower compute for real-world latency.

Hybrid architecture: claimed up to **4×** higher token throughput — think faster and stay accurate. Thinking Budget: stop overthinking; lower, more predictable inference cost.

![figure 2](../../../../assets/vllm/blog/serving/nemotron-3-nano/02-figure_2.png)

**Figure 2.** Higher throughput and leading accuracy among open reasoning models. No numeric TPS table in the prose.

NVIDIA-curated data. Same benches as the intro: SWE Bench Verified, GPQA Diamond, AIME 2025, Arena Hard v2, IFBench — coding, reasoning, math, instruction following. Named enterprise cases: finance, cybersecurity, software development, retail.

![figure 3](../../../../assets/vllm/blog/serving/nemotron-3-nano/03-figure_3.png)

**Figure 3.** Leading accuracy on popular academic benchmarks among open small reasoning models. Scores live in the chart, not in a table.

## Get started

Open weights, training datasets, recipes: fine-tune and deploy on-prem or cloud.

- Weights: [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16), [FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)
- Cookbook / Brev: [launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-36ikINrMffBCbrtTVLr6MFcllcs)

Ideas board: [nemotron.ideas.nvidia.com](http://nemotron.ideas.nvidia.com/?ncid=so-othe-692335). Stay-up-to-date: [NVIDIA Nemotron](https://developer.nvidia.com/nemotron), NVIDIA AI on [LinkedIn](https://www.linkedin.com/showcase/nvidia-ai/posts/?feedView=all), [X](https://x.com/NVIDIAAIDev), [YouTube](https://www.youtube.com/@NVIDIADeveloper), [Nemotron Discord channel](https://discord.com/channels/1019361803752456192/1407781691698708682) / [invite](https://discord.com/invite/nvidiadeveloper).
