---
source: https://vllm.ai/blog/2026-03-11-nemotron-3-super
lang: en
fetched: 2026-09-04
---

# Nemotron 3 Super: 120B/12B hybrid MoE, 1M context, Thinking Budget

Chinese: [zh/vllm/blog/serving/nemotron-3-super.md](../../../../zh/vllm/blog/serving/nemotron-3-super.md)

2026-03-11. **NVIDIA Nemotron Team**. Day-0 image: `vllm==0.17.1`. **4 × H100** BF16 example. Cookbook is the real recipe; this page is the skeleton. Same hybrid family as [Nano](nemotron-3-nano.md) / [Ultra](nemotron-3-ultra.md). Distilled-down cousin: [Lightning](nemotron-35-lightning.md). Earlier 9B: [Nano 2](nemotron-nano2.md). Multimodal cousins: [Nano 2 VL](nemotron-nano-vl.md), [Nano Omni](nemotron-omni.md). Spark box-level caveats: [dgx-spark.md](dgx-spark.md). Hybrid Mamba serving: [hybrid-ssm.md](hybrid-ssm.md). Artificial Analysis / **4×** / **5×** charts are the page’s demos, not your SLA.

**TL;DR from the page:**

- Hybrid Transformer-Mamba MoE: **120B** total, **12B** active; context up to **1M**; text in / text out.
- BF16, FP8, and NVFP4. NVFP4 on Blackwell claimed **4×** throughput vs FP8 on H100 at matched accuracy.
- vs previous Nemotron Super: up to ~**5×** throughput, ~**2×** accuracy in their charts.
- MTP; Latent MoE (4 experts at ~1 expert cost); Thinking Budget.
- `--kv-cache-dtype fp8` `--reasoning-parser nemotron_v3` `--tool-call-parser qwen3_coder`.

## Why this model

Nemotron 3 Super is billed as the mid-size Nemotron 3 open model for complex multi-agent work: plan, reason, execute multi-step tasks. Needs both depth for hard technical problems and efficiency for continuous operation at scale.

Two problems named:

- **The "Context Explosion" Problem.** Multi-agent systems re-send history, tool outputs, and reasoning steps until the window fills. Super’s answer: a **1 million** token context window — long-term memory, less goal drift.
- **The "Thinking Tax".** Reasoning-heavy agents on conventional massive models are costly and slow. Hybrid MoE is claimed to give up to **4×** higher throughput so sub-tasks do not pay full-model latency and cost every time.

vLLM is the serving layer: OpenAI-compatible API, high-efficiency / high-accuracy multi-agent inference at scale.

Local figures (copyright remains with the original site; study copies):

![figure1](../../../../assets/vllm/blog/serving/nemotron-3-super/01-figure1.png)

**Figure 1.** Artificial Analysis chart: Nemotron 3 Super leading on intelligence vs. openness among popular open models. Fully open: weights, datasets, recipes — customize and deploy on your own infra.

## About Nemotron 3 Super

- **Architecture:** Mixture of Experts (MoE) with Hybrid Transformer-Mamba
- Highest throughput efficiency in its size category; up to **5×** higher throughput vs previous Nemotron Super
- **Multi-Token Prediction (MTP):** several future tokens in one forward pass; long-form generation
- **Thinking Budget:** claimed optimal accuracy with minimum reasoning-token generation

**Key specs:**

- **Accuracy:** leading on Artificial Analysis Intelligence Index in its size category; up to **2×** vs previous Nemotron Super
- **Latent MoE:** 4 experts for the inference cost of one
- **Model size:** 120B total, 12B active
- **Context length:** up to 1M
- **Model I/O:** text in, text out
- **Supported GPUs:** B200, H100, DGX Spark, RTX 6000

**Get started:**

- Weights: [Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) — BF16, FP8, and NVFP4
- Run with vLLM
- Technical report: [NVIDIA-Nemotron-3-Super-Technical-Report.pdf](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)

## Run optimized inference with vLLM

BF16, FP8, and NVFP4. NVFP4 on Blackwell: claimed **4×** vs FP8 on H100 while maintaining accuracy. Cookbook for FP8 / NVFP4 detail.

### Install vLLM

```bash
pip install vllm==0.17.1
```

### Serve the model

OpenAI-compatible API. Command below is **4 × H100**. If hardware differs, adjust parallelism. Cookbook for FP8 and NVFP4.

```bash
# BF16
vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --served-model-name nemotron \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3
```

Client from the page (`base_url` port **5000**, `api_key="null"`, reads `reasoning_content`):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:5000/v1", api_key="null")

# Simple chat completion
resp = client.chat.completions.create(
    model="nemotron",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Give me 3 bullet points about vLLM"}
    ],
    temperature=0.7,
    max_tokens=256,
)
print("Reasoning:", resp.choices[0].message.reasoning_content,
      "\nContent:", resp.choices[0].message.content)
```

Caveats vs later Nemotron 3 posts: client hits **:5000** (vLLM default is 8000 unless you pass `--port`); reasoning field is `reasoning_content` here, Lightning/Ultra often read `reasoning`. Easier setup: [cookbook](https://github.com/anushapant/Nemotron/blob/main/usage-cookbook/Nemotron-3-Super/vllm_cookbook.ipynb) or [NVIDIA Brev](https://brev.dev).

## Highest efficiency with leading accuracy for multi-agent applications

![figure2](../../../../assets/vllm/blog/serving/nemotron-3-super/02-figure2.png)

**Figure 2.** Artificial Analysis: intelligence vs. efficiency among popular open models of similar size. Claim: leading accuracy at higher efficiency. No numeric TPS / TTFT table in the prose.

## Get started

Open weights, datasets, and recipes: fine-tune and deploy from workstation to cloud.

- Weights: [Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) — BF16, FP8, NVFP4
- Cookbook: [Nemotron-3-Super/vllm_cookbook.ipynb](https://github.com/anushapant/Nemotron/blob/main/usage-cookbook/Nemotron-3-Super/vllm_cookbook.ipynb) and [Brev](https://brev.dev)
- Technical report: [NVIDIA-Nemotron-3-Super-Technical-Report.pdf](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)

Stay-up-to-date on the page: [NVIDIA news](https://www.nvidia.com/en-us/preferences/email-signup/), NVIDIA AI on [LinkedIn](https://www.linkedin.com/company/nvidia/), [X](https://x.com/NVIDIAAI), [YouTube](https://www.youtube.com/nvidia), Nemotron on [Discord](https://discord.gg/nvidia).

## Acknowledgement

Thanks to everyone who contributed to bringing Nemotron 3 Super to vLLM.

NVIDIA: Nirmal Kumar Juluru, Anusha Pant.

vLLM team and community: Roger Wang, Michael Goin, Thomas Parnell, Kevin Luu, Robert Shaw, Tyler Michael Smith.
