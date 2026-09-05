---
source: https://vllm.ai/blog/2026-04-28-nemotron-omni
lang: en
fetched: 2026-09-04
---

# Nemotron 3 Nano Omni: one 30B/3B for image/audio/video; TPS compared at fixed per-user rate

Chinese: [zh/vllm/blog/serving/nemotron-omni.md](../../../../zh/vllm/blog/serving/nemotron-omni.md)

2026-04-28. **NVIDIA Nemotron Team**. Image: `vllm[audio]==0.20.0`. **256K** spec; the serve snippet uses `--max-model-len 131072`. Same Nemotron 3 text family: [Nano](nemotron-3-nano.md) / [Super](nemotron-3-super.md) / [Ultra](nemotron-3-ultra.md) / [Lightning](nemotron-35-lightning.md). Predecessor VLM: [Nano 2 VL](nemotron-nano-vl.md). Text 9B: [Nano 2](nemotron-nano2.md). This is a **perception sub-agent**, not the [vLLM-Omni](vllm-omni.md) diffusion/TTS stack. Spark: [dgx-spark.md](dgx-spark.md). Training rollout cousins: [native-rl.md](native-rl.md). **7.4×** / **9.2×** / “9x” live on figures — not your SLA.

**TL;DR from the page:**

- Hybrid Transformer-Mamba MoE: **30B** total, **3B** active; context **256K**.
- In: text / image / video / audio. Out: text. Conv3D + Efficient Video Sampling.
- BF16, FP8, NVFP4.
- Throughput compared at a **fixed per-user token rate**: multi-doc ~**7.4×**, video ~**9.2×** vs another open omni. Intro also says **9×** at the same interactivity.
- Accuracy: **20%** higher multimodal intelligence vs the best open alternative (prose); six leaderboards named on Figure 3.
- `--media-io-kwargs '{"video":{"num_frames":512,"fps":1}}'` `--video-pruning-rate 0.5` `--reasoning-parser nemotron_v3` `--tool-call-parser qwen3_coder`.

## Why this model

[Nemotron 3 Nano Omni](https://developer.nvidia.com/blog/nvidia-nemotron-3-nano-omni-powers-multimodal-agent-reasoning-in-a-single-efficient-open-model): highest-efficiency open multimodal model with leading accuracy, for sub-agents that perceive and reason across vision, audio, and language in one loop.

Enterprise agents are multimodal: screens, documents, audio, video, text, often in the same pass. Most stacks bolt separate vision / speech / language models together — extra hops, extra orchestration, fragmented context.

Two problems:

- **Fragmented Models.** Sequential vision, audio, language passes raise latency, cost, and failure modes. Omni collapses that into one multimodal reasoning loop.
- **Efficiency.** Always-on perception (screens, docs, video) needs sustained scale. Hybrid MoE activates **3B of 30B** per forward; EVS and temporal-aware perception cut video compute.

Claim: **9×** higher throughput than other open omni models at the same interactivity.

## TL;DR: About Nemotron 3 Nano Omni

- **Architecture:** Mixture of Experts (MoE) with Hybrid Transformer-Mamba
- **Model size:** 30B total, 3B active
- **Context length:** 256K
- **Unified vision and audio encoders** — one model replaces fragmented stacks. Conv3D for temporal-spatial video.
- **Modalities:** input text, image, video, audio; output text
- **Efficiency:** 9× throughput vs other open omni at the same interactivity. EVS for longer video at the same compute. FP8 and NVFP4.
- **Accuracy:** 20% higher multimodal intelligence vs the best open alternative
- **Post-training:** multi-environment RL via [NeMo RL](https://github.com/nvidia-nemo/rl) and [NeMo Gym](https://github.com/NVIDIA-NeMo/gym) across text, image, audio, video
- **Supported GPUs:** B200, H100, H200, A100, L40S, DGX Spark, RTX 6000

**Get started:**

- Weights: [BF16](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16), [FP8](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8), [NVFP4](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4)
- Cookbook: [Nemotron-3-Nano-Omni/vllm_cookbook.ipynb](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano-Omni/vllm_cookbook.ipynb) and [Brev](https://brev.nvidia.com/launchable/deploy?launchableID=env-3Cm2gB9j5ROkCbiNKH5SQhERqBV)
- Technical report: [NVIDIA-Nemotron-3-Omni-report.pdf](https://research.nvidia.com/labs/adlr/files/NVIDIA-Nemotron-3-Omni-report.pdf)

## Run optimized multimodal inference with vLLM

BF16, FP8, NVFP4. Cookbook for FP8 / NVFP4 flags.

### Install vLLM

```bash
pip install vllm[audio]==0.20.0
```

### Serve the model

OpenAI-compatible API. Set attention backend / env as needed. Snippet `--max-model-len 131072` is **not** the 256K spec sheet.

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16" \
    --served-model-name nemotron \
    --trust-remote-code \
    --dtype auto \
    --host 0.0.0.0 \
    --port 5000 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --media-io-kwargs '{"video":{"num_frames":512,"fps":1}}' \
    --video-pruning-rate 0.5 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3
```

Client (port **5000**, `api_key="null"`, reads `message.reasoning` — not `reasoning_content`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5000/v1", api_key="null")
resp = client.chat.completions.create(
    model="nemotron",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about GPUs."}
    ],
    temperature=1,
    max_tokens=1024,
)
print("Reasoning:", resp.choices[0].message.reasoning,
      "\nContent:", resp.choices[0].message.content)
```

The snippet is **text-only**. Multimodal payloads live in the cookbook / Brev. Easier setup: [cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano-Omni/vllm_cookbook.ipynb) or [Brev](https://brev.nvidia.com/launchable/deploy?launchableID=env-3Cm2gB9j5ROkCbiNKH5SQhERqBV).

## Highest efficiency with leading accuracy for multimodal agentic applications

Hardware-efficient inference: FP8, NVFP4, NVIDIA kernels, EVS. Conv3D temporal-spatial processing: workstation to cloud.

Figure 1 holds **per-user token rate** fixed and measures total system throughput that still keeps real-time interactivity — multi-document and video. Not peak concurrency.

Local figures (copyright remains with the original site; study copies):

![figure1](../../../../assets/vllm/blog/serving/nemotron-omni/01-figure1.png)

**Figure 1.** Total system throughput at a fixed per-user interactivity threshold (tokens/sec/user): **7.4×** multi-document, **9.2×** video vs an alternative open omni.

### Multimodal accuracy

![figure2](../../../../assets/vllm/blog/serving/nemotron-omni/02-figure2.png)

**Figure 2.** vs previous [Nemotron Nano VL V2](nemotron-nano-vl.md): higher multimodal accuracy on document intelligence, video, and audio reasoning. Combined with efficiency → six leaderboard placements.

![figure3](../../../../assets/vllm/blog/serving/nemotron-omni/03-figure3.png)

**Figure 3.** Six leaderboards: MMlongbench-Doc, OCRBenchV2, WorldSense, DailyOmni, VoiceBench, MediaPerf. MediaPerf: highest throughput on every task, lowest inference cost for video-level tagging.

Role in an agent system: multimodal perception and context sub-agent — eyes and ears on screens, documents, audio, video; structured understanding fed to orchestration / execution agents. Lightweight enough to sit beside other models without duplicating separate perception pipelines. Pitch: computer-use agents, document intelligence, audio-video pipelines — without a fragmented multimodal stack.

## Get started

Open weights, datasets, recipes: workstation to cloud.

- Weights: [BF16](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16), [FP8](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8), [NVFP4](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4)
- Cookbook: [vllm_cookbook.ipynb](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano-Omni/vllm_cookbook.ipynb) and [Brev](https://brev.nvidia.com/launchable/deploy?launchableID=env-3Cm2gB9j5ROkCbiNKH5SQhERqBV)
- Technical report: [NVIDIA-Nemotron-3-Omni-report.pdf](https://research.nvidia.com/labs/adlr/files/NVIDIA-Nemotron-3-Omni-report.pdf)

Stay-up-to-date: [NVIDIA news](https://www.nvidia.com/en-us/preferences/email-signup/), NVIDIA AI on [LinkedIn](https://www.linkedin.com/company/nvidia/), [X](https://x.com/NVIDIAAI), [YouTube](https://www.youtube.com/nvidia), Nemotron on [Discord](https://discord.gg/nvidia).

## Acknowledgement

Thanks to everyone who contributed to bringing Nemotron 3 Nano Omni to vLLM.

NVIDIA: Nirmal Kumar Juluru, Anusha Pant.

vLLM team and community: Roger Wang, Michael Goin, Thomas Parnell, Kevin Luu, Robert Shaw, Tyler Michael Smith.
