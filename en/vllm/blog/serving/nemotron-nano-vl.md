---
source: https://vllm.ai/blog/2025-10-31-run-multimodal-reasoning-agents-nvidia-nemotron
lang: en
fetched: 2026-09-04
---

# Nemotron Nano 2 VL: 12B video/docs, EVS drops redundant frames

Chinese: [zh/vllm/blog/serving/nemotron-nano-vl.md](../../../../zh/vllm/blog/serving/nemotron-nano-vl.md)

2025-10-31. **NVIDIA Nemotron Team**. Then **nightly**. Text LLM backbone: [Nano 2](nemotron-nano2.md). Nemotron 3 text: [Nano](nemotron-3-nano.md) / [Super](nemotron-3-super.md) / [Ultra](nemotron-3-ultra.md) / [Lightning](nemotron-35-lightning.md). Successor omni: [Nano Omni](nemotron-omni.md). That Omni is a perception sub-agent, not the [vLLM-Omni](vllm-omni.md) diffusion/TTS stack. VLM-suite average **74** vs then-top VL **64.2** — marketing; re-measure from the cookbook.

**TL;DR from the page:**

- 12B VLM for video understanding and document intelligence. Context **128K**. Text out.
- Stack: [CRADIOH-V2](https://huggingface.co/nvidia/C-RADIOv2-H) encoder + Efficient Video Sampling (EVS) + [Nano 2 LLM](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2).
- `--video-pruning-rate 0` means no prune. FP8/FP4: `--quantization modelopt` / `modelopt_fp4`.
- System `/no_think` disables thinking.
- Nightly: `uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly --prerelease=allow`.

## Why this model

[NVIDIA Nemotron Nano 2 VL](https://huggingface.co/nvidia/Nemotron-Nano-12B-v2-VL-BF16): open VLM for video understanding and document intelligence. Hybrid Transformer–Mamba; claimed higher throughput at SOTA multimodal reasoning accuracy. [EVS](https://arxiv.org/abs/2510.14624) cuts redundant video tokens so more video fits in the same compute.

## Leading multimodal model for efficient video understanding and document intelligence

One model for video and documents. Hybrid Transformer–Mamba: Transformer reasoning + Mamba compute efficiency; multi-image inputs faster.

NVIDIA-curated multimodal data. Named benches: MMMU, MathVista, AI2D, OCRBench, OCRBench-v2, OCR-Reasoning, ChartQA, DocVQA, Video-MME — multimodal reasoning, character recognition, chart reasoning, VQA. Enterprise pitch: extract and comprehend across videos, documents, forms, charts.

Local figures (copyright remains with the original site; study copies):

![figure1](../../../../assets/vllm/blog/serving/nemotron-nano-vl/01-figure1.png)

**Figure 1.** Leading accuracy on video-understanding and document-intelligence benches. Scores in the chart, not a table.

### Improving efficiency with EVS

EVS: higher throughput, faster responses, claimed no accuracy sacrifice. Prunes redundant frames; keeps semantic richness; longer video at the same budget. Pitch: hours of meetings / training / calls in minutes.

![figure2](../../../../assets/vllm/blog/serving/nemotron-nano-vl/02-figure2.png)

**Figure 2.** Accuracy vs token-drop thresholds (EVS) on Video-MME and LongVideo. Axis numbers in the chart.

## About Nemotron Nano 2 VL

- **Architecture:**
  - [CRADIOH-V2](https://huggingface.co/nvidia/C-RADIOv2-H) vision encoder
  - EVS as token-compression module
  - Hybrid Transformer-Mamba — [Nemotron Nano 2 LLM](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) backbone with reasoning
- **Accuracy:**
  - Leading on OCRBench v2
  - **74** average (vs **64.2** then-top VL) on MMMU, MathVista, AI2D, OCRBench, OCRBench-v2, OCR-Reasoning, ChartQA, DocVQA, Video-MME
- **Model size:** 12B
- **Context length:** 128k
- **Model input:** multi-image documents, videos, text
- **Model output:** text
- **Get started:**
  - Weights: [BF16](https://huggingface.co/nvidia/Nemotron-Nano-12B-v2-VL-BF16), [FP8](https://huggingface.co/nvidia/Nemotron-Nano-12B-v2-VL-FP8), [FP4-QAD](https://huggingface.co/nvidia/Nemotron-Nano-12B-v2-VL-FP4-QAD)
  - Run with vLLM
  - Technical report: [NVIDIA-Nemotron-Nano-V2-VL-report.pdf](https://research.nvidia.com/labs/adlr/files/NVIDIA-Nemotron-Nano-V2-VL-report.pdf)

Dataset write-up named in the body: [nemotron-vlm-dataset-v2](https://huggingface.co/blog/nvidia/nemotron-vlm-dataset-v2).

## Run optimized inference with vLLM

BF16, FP8, FP4. OpenAI-compatible server. Concurrent requests.

### Install vLLM

Nightly then:

```bash
uv venv
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly --prerelease=allow
```

### Deploy and query the inference server

`--video-pruning-rate 0` = no prune. FP8/FP4 Hugging Face slugs on the serve lines swap to `Nemotron-Nano-VL-12B-V2-*` (VL before 12B) vs the BF16 / get-started `Nemotron-Nano-12B-v2-VL-*`. Keep both as published.

```bash
vllm serve nvidia/Nemotron-Nano-12B-v2-VL-BF16 --trust-remote-code --dtype bfloat16 --video-pruning-rate 0

# FP8
vllm serve nvidia/Nemotron-Nano-VL-12B-V2-FP8 --trust-remote-code --quantization modelopt --video-pruning-rate 0

# FP4
vllm serve nvidia/Nemotron-Nano-VL-12B-V2-FP4-QAD --trust-remote-code --quantization modelopt_fp4 --video-pruning-rate 0
```

Client (`temperature=0.0`, `max_tokens=1024`; system `/no_think`; one `image_url`):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="null")
# Simple chat completion
resp = client.chat.completions.create(
    model="nvidia/Nemotron-Nano-12B-v2-VL-BF16",
    messages=[
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": [
            {"type": "text", "text": "Give me 3 interesting facts about this image."},
            {"type": "image_url", "image_url": {"url": "https://blogs.nvidia.com/wp-content/uploads/2025/08/gamescom-g-assist-nv-blog-1280x680-1.jpg"}
            }
            ]},
    ],
    temperature=0.0,
    max_tokens=1024,
)
print(resp.choices[0].message.content)
```

More examples: [vLLM cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-Nano2-VL/vllm_cookbook.ipynb), [recipe](https://docs.vllm.ai/projects/recipes/en/latest/NVIDIA/Nemotron-Nano-12B-v2-VL.html).

Ideas board: [nemotron.ideas.nvidia.com](http://nemotron.ideas.nvidia.com/?ncid=so-othe-692335). Stay-up-to-date: [NVIDIA Nemotron](https://developer.nvidia.com/nemotron), NVIDIA AI on [LinkedIn](https://www.linkedin.com/showcase/nvidia-ai/posts/?feedView=all), [X](https://x.com/NVIDIAAIDev), [YouTube](https://www.youtube.com/@NVIDIADeveloper), [Nemotron Discord channel](https://discord.com/channels/1019361803752456192/1407781691698708682) / [invite](https://discord.com/invite/nvidiadeveloper).
