---
source: https://vllm.ai/blog/2026-08-20-verl-omni-v0-2-0
lang: en
fetched: 2026-09-04
---

# VeRL-Omni v0.2.0: Faster Diffusion RL and Stable Omni Training

Chinese: [zh/vllm/blog/serving/verl-omni-v020.md](../../../../zh/vllm/blog/serving/verl-omni-v020.md)

2026-08-20. **VeRL-Omni Team**. Follows the [May announcement](https://vllm.ai/blog/2026-05-14-verl-omni) ([verl-omni.md](verl-omni.md)). Repo: [verl-project/verl-omni](https://github.com/verl-project/verl-omni). Two headlines: faster diffusion RL (Qwen-Image FlowGRPO via vLLM-Omni + verl V1 trainer); stable omni training (omni V1 trainer, reusable adapters, FSDP2, vLLM-Omni rollout). Study note; wandb recipe numbers on the page, not your SLA.

**TL;DR from the page:** v0.1 rollout was serial `B≈1` DiT forwards (10 denoising steps, True-CFG doubles each step); GPU util ~**80%**. v0.2 request-level packing: GPU util ~**100%**, isolated generation **226 s → 108 s** (**52%**). MMK12 (Qwen3-Omni Thinker × GSPO, 4× H800 80GB): val reward **0.833**, actor-rollout Pearson **0.998**, ~**59 GB**. Recipe step times still cite the v0.1 LoRA table: ~**420 s** on 4× H800; async reward ~**360 s** on 5 GPUs.

Cover figure stays on the original page (no local copy): **VeRL-Omni v0.2.0 release overview**.

Local figures (copyright remains with the original site; study copies). Charts: **blue = v0.1**, **green = v0.2**.

![qwen image gpu utilization](../../../../assets/vllm/blog/serving/verl-omni-v020/02-qwen-image-gpu-utilization.svg)

![qwen image timing gen](../../../../assets/vllm/blog/serving/verl-omni-v020/03-qwen-image-timing-gen.svg)

![qwen image timing step](../../../../assets/vllm/blog/serving/verl-omni-v020/04-qwen-image-timing-step.svg)

![omni ppo adapter flow](../../../../assets/vllm/blog/serving/verl-omni-v020/05-omni-ppo-adapter-flow.svg)

![mmk12 training rewards](../../../../assets/vllm/blog/serving/verl-omni-v020/06-mmk12_training_rewards.svg)

![mmk12 val rewards](../../../../assets/vllm/blog/serving/verl-omni-v020/07-mmk12_val_rewards.svg)

## 1. Faster diffusion RL

Not the same expense as autoregressive LLM RL. One rollout: many denoising steps, large latents, prompt embeddings, optional CFG, reward scoring, old-log-prob recompute, policy-weight sync. For Qwen-Image FlowGRPO there is **no single villain** — step time is all of those together.

### Key features

- **Request-level batching** is the default vLLM-Omni rollout path for supported diffusion adapters. Compatible requests pack into larger transformer forwards; explicit concurrency knobs. Guide: [rollout batching](https://verl-omni.readthedocs.io/en/latest/start/rollout_batching.html). Runtime design: [diffusion continuous batching](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/diffusion_continuous_batching).
- **V1 trainer** for diffusion — closer to the modern trainer used elsewhere; groundwork for decoupled rollout and training.

Correctness-sensitive fixes named: request-batched diffusion log-probs, async rollout semantics, rank-local LoRA weight-update routes, hooks for optional rollout-correction recipes. Faster rollout only counts if trajectories and log-probs still describe the **same** policy.

### New support

| Model × Algorithm | Acceleration / support | Script | W&B |
|---|---|---|---|
| Qwen-Image × FlowGRPO LoRA | **request-level batching** | [script](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh) | [run](https://wandb.ai/mikecheung/flow_grpo/runs/1vsrnhbd) |
| Qwen-Image × FlowGRPO full model | step-wise continuous batching | [script](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr.sh) | [run](https://wandb.ai/andyzhou/VeRL-Omni-demo/runs/8p8y9olb) |
| SD3.5 Medium × FlowGRPO LoRA, **V1 trainer** | **request-level batching**, sync | [script](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh) | [run](https://wandb.ai/mikecheung/flow_grpo/runs/h04p15jr) |
| SD3.5 Medium × FlowGRPO LoRA, **V1 trainer** | **request-level batching**, `separate_async` | [script](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1_separate_async.sh) | [run](https://api.wandb.ai/links/didan/kk5uxbmh) |

Full matrix: [README.md](https://github.com/verl-project/verl-omni#model-and-algorithm-support-).

### Recipe and benchmark

Qwen-Image LoRA OCR is the exhibit. v0.1: serial `B≈1`, 10 denoising steps, True-CFG → two forwards per step; GPU util hovered ~**80%**. v0.2 packs complete requests into one transformer forward; util ~**100%**; isolated generation **226 s → 108 s** (**52%**). Per-image generation latency follows. Reference: [v0.1](https://wandb.ai/mikecheung/flow_grpo/runs/o7x44yrr), [v0.2](https://wandb.ai/mikecheung/flow_grpo/runs/1vsrnhbd).

**Figure (GPU util).** Blue v0.1 → green v0.2 after request-level packing.

**Figure (generation time).** Isolated gen drops on the v0.2 path.

**Figure (step time).** Same trend.

Default-on for the production-style Qwen-Image FlowGRPO LoRA recipe. Entry: `run_qwen_image_ocr_lora.sh`. Switch off step-wise execution; let Omni schedule up to `max_num_seqs`:

```bash
actor_rollout_ref.rollout.step_execution=false
++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=32
```

Qwen-Image LoRA + True-CFG at 512 px: practical `max_num_seqs=8` to `32`; larger hits HBM pressure. SD3.5 is lighter: `max_num_seqs=256`.

Recipe-level step times (same story as [verl-omni.md](verl-omni.md)): baseline LoRA ~**420 s**/step on 4× H800; async reward ~**360 s**/step on 5 GPUs.

## 2. Stable omni training

Omni models are small systems: processors, modality towers, trainable stages, rollout that must stay aligned with the actor. v0.2 moves from one-off integrations to a reusable stack.

### Key features

- **verl V1 trainer** for omni: worker orchestration, standard config overrides, alignment with vLLM-Omni rollout.
- **Reusable omni adapter:** shared interface for model setup, processor setup, trainable-stage selection, FSDP prep, rollout alignment.

**Figure (adapter flow).** `main_omni.py` only decides an online omni job enters the verl PPO V1 path. PPO trainer owns the generic RL loop (rollout, advantage, policy update). FSDP omni engine loads the Hugging Face model and asks `OmniModelBase` for the adapter. For Qwen3-Omni thinker training, `Qwen3OmniThinkerAdapter` strips inactive modules (Talker, codec), redirects `forward` to the thinker, prepares processor and rollout hooks, then returns to PPO.

Thinker-only training: FSDP / FSDP2 wrapping.

### New support

| Model × Algorithm | Modality / dataset | Support | Script | W&B |
|---|---|---|---|---|
| Qwen3-Omni Thinker × GSPO | text → text / GSM8K | **V1 trainer**, reusable adapter, FSDP2, vLLM-Omni rollout | [script](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh) | [run](https://wandb.ai/mikecheung/gspo/runs/j5mro1tn) |
| Qwen3-Omni Thinker × GSPO | image → text / MMK12 | **V1 trainer**, multimodal data, actor-rollout consistency | [script](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh) | [run](https://wandb.ai/mikecheung/gspo/runs/2j8hxr36) |
| Qwen3-Omni Thinker × GSPO | text + image + audio → text / AVQA-R1-6K | **V1 trainer**, NPU recipe, multimodal inputs | [script](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh) | — |
| Qwen3-Omni Thinker × DPO | multimodal → preference / Omni-Preference | `OmniDPOLoss`, modality-grouped batches | [script](https://github.com/verl-project/verl-omni/blob/main/examples/dpo_trainer/qwen3_omni/qwen3_omni/run_qwen3_omni_omni_preference_lora.sh) | [report](https://api.wandb.ai/links/didan/iumxl2zr) |

### Recipe and benchmark: MMK12

Anchor: `run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh`. K12 visual math (`image → text`), GSPO, LoRA rank **32**, colocated actor-rollout on **4 × H800 80GB**. Rollout shape: **128** prompts × **16** responses = **2048** samples. After training: val reward **0.833**, actor-rollout Pearson **0.998**, ~**59 GB**. [wandb](https://wandb.ai/mikecheung/gspo/runs/2j8hxr36).

**Figure (MMK12 train rewards).** Mean training scores.

**Figure (MMK12 val rewards).** Mean validation scores.

Data pipeline: raw MMK12 parquet → verl RL parquet. Image bytes inline; prompt asks for a structured answer. Reward: `math_verify` accuracy + progressive format reward on `<answer>...\boxed{}...</answer>`.

```bash
python examples/gspo_trainer/data_process/mmk12.py \
    --local_dataset_path /path/to/mmk12/ \
    --local_save_dir ~/data/mmk12

TRAIN_FILE=$HOME/data/mmk12/train.parquet \
VAL_FILE=$HOME/data/mmk12/test.parquet \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh
```

This is the stability story: not a one-off launch path — V1 trainer, reusable adapter, multimodal data, consistency metrics, documented image-to-text benchmark. Thinker-side serving context: [qwen3-omni.md](qwen3-omni.md).

## Model and algorithm extensions

| Model / family | Category | Modality | Algorithm / recipe | Update |
|---|---|---|---|---|
| [LTX2.3](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/ltx2/README.md) | Diffusion generator | Text → Video + Audio | FlowGRPO | T2V+audio; CLAP and ImageBind rewards |
| [Qwen-Image-Edit](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image_edit/README.md) | Diffusion image editor | Text + Image → Image | FlowGRPO | Edit-training interface + data prep |
| [BAGEL](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/bagel/README.md) | Unified understand + gen | Text + Image | FlowGRPO | Full-param and LoRA; OCR and PickScore |
| [SD3.5 + DiNa-LRM](https://verl-omni.readthedocs.io/en/latest/examples/flowgrpo_trainer_sd35_drm.html) | Diffusion generator | Text → Image | FlowGRPO + latent reward | Scores clean latents; skips VAE decode at reward time |
| [Flow-DPPO](https://verl-omni.readthedocs.io/en/latest/algo/flowdppo.html) | Diffusion algorithm | Text/Image → Image | Flow-DPPO | Alternative policy-opt for Qwen-Image-style RL |
| [Wan2.2](https://github.com/verl-project/verl-omni/blob/main/examples/dancegrpo_trainer/README.md) | Diffusion video | Text → Video | DanceGRPO | Video-generation RL recipe |

Also: Ascend NPU Dockerfiles and install guidance.

## Future plan

Fully async omni training; MiniMax-H3, MiniCPM-o, OPD/M-OPD trainers; video diffusion efficiency via batching, TQ, V1 trainer; harden diffusion/omni rollout for async; agentic RL (multi-stage, multi-turn).

## Join the community

- **Code:** [github.com/verl-project/verl-omni](https://github.com/verl-project/verl-omni)
- **Docs:** [verl-omni.readthedocs.io](https://verl-omni.readthedocs.io/en/latest/index.html)
- **Contributing:** [`CONTRIBUTING.md`](https://github.com/verl-project/verl-omni/blob/main/CONTRIBUTING.md)
