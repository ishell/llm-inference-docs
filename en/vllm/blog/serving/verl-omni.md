---
source: https://vllm.ai/blog/2026-05-14-verl-omni
lang: en
fetched: 2026-09-04
---

# Announcing VeRL-Omni: Easy, Fast, and Stable RL Training for Diffusion and Omni-Modality Models

Chinese: [zh/vllm/blog/serving/verl-omni.md](../../../../zh/vllm/blog/serving/verl-omni.md)

2026-05-14. **VeRL-Omni Team**. Pre-release of [VeRL-Omni](https://github.com/verl-project/verl-omni) on [`verl`](https://github.com/verl-project/verl) + [`vllm-omni`](https://github.com/vllm-project/vllm-omni). Training stays in verl; diffusion / omni rollout uses Omni. v0.2: [verl-omni-v020.md](verl-omni-v020.md). Same Omni line: [vllm-omni.md](vllm-omni.md). Study note; H800 / H200 recipe numbers on the page, not your SLA. The post inlines **no** `vllm serve` catalog and **no** trainer CLI — launch scripts live under `examples`.

**TL;DR from the page:** LoRA FlowGRPO on NVIDIA H800: colocated 4 GPU **0.305** images/GPU/s at **420 s**/step; async reward on 5 GPUs **0.280** images/GPU/s at **360 s**/step (~**14%** wall-clock). Non-CFG full-model Qwen-Image OCR on 4× H200: **0.510** images/GPU/s at ~**250 s**/step. Text rendering visibly better by step **120**. NVIDIA GPU and Ascend NPU.

Figures stay on the original page (no local copies). Captions:

**Architecture.** VeRL-Omni architecture (hero).

**FlowGRPO.** Algorithm diagram: rollout → reward → policy update → weight sync.

**Quality table.** Prompt “Hidden Trail” / “Make A Wish”: training step **0** vs **120**.

**Curves.** Validation reward ~**0.7 → 0.95**; rollout reward mean ~**0.15 → 0.9** (low start expected for non-CFG); zero-std ratio climbs only after reward saturates; actor `pg_clipfrac` stays in a healthy range (figure caption says “clip ratio”).

## Why VeRL-Omni?

LLM RL moved fast; **multimodal generative RL** (diffusion and omni, image/video/audio understand + generate) still needs:

- **Diffusion and omni-modality extension.** verl’s flexibility onto DiT (Qwen-Image), mixed AR-DiT (Qwen-Omni), unified understand+gen (BAGEL, HunyuanImage3.0).
- **Heterogeneous rollout pipelines.** Rollouts are *denoising trajectories* in continuous latent space, not token sequences. One rollout may call text encoder → DiT → VAE.
- **Complex workload scheduling.** Rewards are themselves multimodal models (VLM judges, OCR scorers). Generation rollouts peak higher in memory than text.

## Key features

- **Efficient multimodal rollout.** vLLM-Omni async serving; accuracy they call on par with diffusers. Step-wise continuous batching, embedding caching, etc. No throughput table for the serving path itself.
- **Flexible reward engine.** Rule-based and model-based (VLM-as-judge for OCR). vLLM for VLM/LLM reward inference. Reward overlapped with rollout and training.
- **Modular trainers.** DiffusersFSDP / Megatron / VeOmni; FSDP / USP / TP.
- **Hardware.** NVIDIA GPUs and Ascend NPUs.
- **E2E recipes and benchmarks.** Reference throughput on the page (H800 LoRA table + H200 full-model line).

## Algorithm and model support

| Model | Architecture | Modality | Algorithm | Status |
|---|---|---|---|---|
| Qwen-Image | DiT | Text → Image | [FlowGRPO](https://arxiv.org/abs/2505.05470), [MixGRPO](https://arxiv.org/abs/2507.21802), [GRPO-Guard](https://arxiv.org/abs/2510.22319) | Released |
| BAGEL | Unified understand + gen | Text + Image | [FlowGRPO](https://arxiv.org/abs/2505.05470) | PR ready |
| Qwen3-Omni-Thinker | AR | Text / Image / Video / Audio | [GSPO](https://arxiv.org/abs/2507.18071) | PR ready |
| Wan2.2 | DiT | Text → Video | DanceGRPO | WIP |
| SD3.5 | DiT | Text → Image | DPO | WIP |
| HunyuanImage-3.0 | Unified understand + gen | Text + Image | MixGRPO, SRPO | Planned |

Statuses are as of the 2026-05-14 pre-release. v0.2 is a later note.

## Getting started

### Installation

Install: [Installation Doc](https://verl-omni.readthedocs.io/en/latest/start/install.html). The post does not inline `pip` / Docker.

### Training diffusion models

Launch scripts: [examples](https://github.com/verl-project/verl-omni/tree/main/examples). Image / audio / video understand+gen trainers. Track on wandb.

### Demo: Qwen-Image FlowGRPO post-training

[flowgrpo example](https://github.com/verl-project/verl-omni/tree/main/examples/flowgrpo_trainer): Qwen-Image, OCR reward. Reward model `Qwen3-VL-8B-Instruct` reads rendered text vs dataset ground truth.

#### Algorithm review

FlowGRPO: online policy for flow-matching. Multi-step SDE sampling with a diffusion policy for exploration; model-based rewards. Four stages:

1. **Rollout generation** — trajectories of log probabilities and images.
2. **Reward model scoring** — trajectory advantages.
3. **Policy optimization** — FlowGRPO CLIP-style loss.
4. **Weight synchronization** — trainer weights → rollout workers, **periodically**, so samples follow the latest policy.

#### LoRA fine-tuning (NVIDIA H800)

| Mode | # GPUs | Actor | Rollout | Async Reward | Throughput (images/GPU/s) | Time per Step (s) |
|---|---:|---:|---:|---|---:|---:|
| FlowGRPO colocated training | 4 | 4 | 4 | 0 (sync) | 0.305 | 420 |
| FlowGRPO w/ async reward | 5 | 4 | 4 | 1 (async) | 0.280 | 360 |

Dedicated GPU for the reward model: wall-clock per step **~14%** (420 → 360) by overlapping reward with policy training. Throughput per GPU is slightly lower (0.280 vs 0.305) because the fifth GPU is the scorer, not another actor/rollout. Batch size, resolution, denoising steps, LoRA rank, and whether actor/rollout share the same four cards’ memory are **not** in the table.

#### Full-model fine-tuning

**Non-CFG** full-model Qwen-Image OCR on **4 × NVIDIA H200**: **0.510** images/GPU/s at ~**250 s**/step. Text rendering “largely enhanced” in **120** steps. Prompts on the page (images stay on the original site):

- **Hidden Trail:** `A wooden trail marker in a dense forest with "Hidden Trail" carved into the wood, surrounded by moss and foliage.`
- **Make A Wish:** `A birthday card interior with "Make A Wish" in cursive handwriting, surrounded by sparkling candles and colorful confetti.`

Reference curves: critic and validation rewards converge. Rollout mean starts low (expected for non-CFG). Figure alt/captions:

- validation reward ~**0.7 → 0.95**
- rollout reward mean ~**0.15 → 0.9**
- `critic/rewards/zero_std_ratio` climbs only after reward saturates
- `actor/pg_clipfrac` (“clip ratio”) stays in a healthy range — the post does not give the numeric band

Metrics docs: [Training Metrics](https://verl-omni.readthedocs.io/en/latest/start/metrics.html).

## Future roadmap

Pre-release; core diffusion RL stack they call stable. Named next:

- More open-source diffusion / omni models (image/video/audio; unified understand+gen).
- Algorithms as they land (e.g. DiffusionNFT).
- **Fully** async RL across actor, rollout, and reward — beyond current async-reward.
- Co-optimize with vLLM-Omni: parallelism, quantization, batching, request scheduling (rollout is a large fraction of step time).
- More trainer engines on Megatron-core and VeOmni, besides DiffusersFSDPTrainer.
- Harden Ascend NPU; hardware plugin system for more backends.

## Join the community

- **Code:** [github.com/verl-project/verl-omni](https://github.com/verl-project/verl-omni)
- **Docs:** [verl-omni.readthedocs.io](https://verl-omni.readthedocs.io/en/latest/index.html)
- **Contributing:** [`CONTRIBUTING.md`](https://github.com/verl-project/verl-omni/blob/main/CONTRIBUTING.md)
- **Weekly meeting:** Tuesday **11:00AM (GMT+8)** — [meet.google.com/rho-aode-kmg](https://meet.google.com/rho-aode-kmg)
