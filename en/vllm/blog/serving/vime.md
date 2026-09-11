---
source: https://vllm.ai/blog/2026-06-09-announcing-vime
lang: en
fetched: 2026-09-04
---

# Announcing vime: A Simple, Stable, and Efficient RL Framework for LLMs

Chinese: [zh/vllm/blog/serving/vime.md](../../../../zh/vllm/blog/serving/vime.md)

2026-06-09. **vime Contributors and the vLLM Team**. Repo: [vllm-project/vime](https://github.com/vllm-project/vime). slime’s training stack + vLLM rollouts under one architecture. Native RL APIs in vLLM itself: [native-rl.md](native-rl.md). ROCm follow-on: [vime-rocm.md](vime-rocm.md). Study note; GB200 / H200 / A100 numbers on the page, not your SLA.

**TL;DR from the page:** `--vllm-` prefix pins inference knobs on one CLI; default rollout entry `vime.rollout.vllm_rollout`. Qwen3-30B-A3B, 8-GPU colocate, dapo-math-17k, GRPO: GB200 mean step ~**147 s**, H200 ~**252 s** → ~**1.72×**. Qwen3-4B A100: `train_rollout_logprob_abs_diff` ~**0.011** vs baseline drift to ~**0.77**. MoE R3: ~**0.019 → 0.013**. GLM-4.5-Air GB200: `raw_reward` mean ~**0.56** / 100 steps; logprob diff **0.02–0.03**, mean ~**0.028**.

Local figures (copyright remains with the original site; study copies):

![arch v1](../../../../assets/vllm/blog/serving/vime/01-arch_v1.png)

![Qwen3 30B A3B GB200 vs H200 step bar](../../../../assets/vllm/blog/serving/vime/02-Qwen3-30B-A3B_GB200_vs_H200_step_bar.png)

![Qwen3 4B Training raw reward compare](../../../../assets/vllm/blog/serving/vime/03-Qwen3-4B_Training_raw_reward_compare.png)

![Qwen3 30B A3B MoE R3 Comparison](../../../../assets/vllm/blog/serving/vime/04-Qwen3-30B-A3B_MoE_R3_Comparison.png)

![Qwen3 30B A3B GB200 vime baseline compare](../../../../assets/vllm/blog/serving/vime/05-Qwen3-30B-A3B_GB200_vime_baseline_compare.png)

![GLM 4.5 Air GB200 precision](../../../../assets/vllm/blog/serving/vime/06-GLM-4.5-Air_GB200_precision.png)

**Figure (architecture).** Megatron training with vLLM rollout through a decoupled data buffer.

**Figure (step speed).** Qwen3-30B-A3B on GB200 vs H200.

**Figure (Qwen3-4B).** vime versus baseline (`raw_reward` / alignment).

**Figure (MoE R3).** Routing replay vs train-inference mismatch.

**Figure (GB200 MoE).** Colocated train/rollout alignment vs baseline.

**Figure (GLM-4.5-Air).** Reward up; logprob alignment holds.

## Vision

[slime](https://github.com/THUDM/slime) (validated on GLM-class models): open, lightweight, efficient — no native vLLM backend. vLLM: active inference engine, multi-platform, fast iteration. vime’s job is to join those without forcing a choice among hardware stack, training stability, and inference performance.

## Positioning

vLLM already sits under several post-training frameworks (alphabetical): [NeMo RL](https://github.com/NVIDIA-NeMo/RL), [OpenRLHF](https://github.com/openrlhf/openrlhf), [verl](https://github.com/verl-project/verl), and others. vime is the slime-shaped bridge, aligned to both projects’ release clocks. The community keeps supporting the other integrations.

## Architecture

slime’s three-stage, decoupled train-inference design; rollout backend is vLLM:

- **Training (Megatron)** — parameter updates; sync weights to rollout.
- **Rollout (vLLM + Router)** — sampling with reward / verifier signals.
- **Data Buffer** — prompt injection and custom rollout logic between the two sides.

## Key capabilities

- **Easy to use.** slime / Megatron parameter conventions; vLLM knobs via `--vllm-`. Default rollout: `vime.rollout.vllm_rollout`.
- **Stable train-inference alignment.** Dense and MoE: `train_rollout_logprob_abs_diff` stays in a controllable range over long runs. MoE **R3** (routing replay) further cuts mismatch.
- **Algorithms and models.** GRPO, PPO; Qwen3 Dense/MoE, GLM-4.5 — E2E examples and CI-verified paths.
- **Multi-hardware.** Training resources, rollout resources, cluster topology abstracted so the same RL pipeline can follow vLLM’s hardware plugins as they land.

## Validation and benchmarks

Qwen3-30B-A3B, 8-GPU colocate, dapo-math-17k, GRPO: GB200 mean step ~**147 s**, H200 ~**252 s**. Same framework: GB200 E2E step ~**1.72×** H200.

### Qwen3-4B on A100

GRPO, 4 train + 4 inference **non-colocate**, gsm8k. vime `train_rollout_logprob_abs_diff` ~**0.011** throughout. Baseline drifts to ~**0.77**.

### Qwen3-30B-A3B MoE with R3

A100, 4 train + 4 inference, dapo-math-17k, **EP=4**. R3 routing replay: logprob diff ~**0.019 → 0.013**.

### Qwen3-30B-A3B MoE on GB200

8-GPU colocate, dapo-math-17k. vime and baseline `raw_reward` curves stay close. Both keep `train_rollout_logprob_abs_diff` ~**0.018**; no sustained baseline-side drift in this setup.

### GLM-4.5-Air on GB200

GRPO, 8-GPU colocate, dapo-math-17k. `raw_reward` trends up over **100** steps, mean ~**0.56**. `train_rollout_logprob_abs_diff` **0.02–0.03**, mean ~**0.028**.

## Roadmap

- Deeper vLLM: Router, PD disaggregation, FP8, multi-model serving.
- More hardware via vLLM’s plugin system.
- Training efficiency and algorithms: fully async pipelines, train-inference mismatch correction, Agentic RL (multi-turn tool calling, multi-agent), fast follow on MoE and VLM.

## Quick start

Like slime: configure Megatron training resources and vLLM rollout resources, prepare checkpoints and data, launch `train.py` or `train_async.py`.

- **Docs:** [Quick Start](https://github.com/vllm-project/vime/tree/main/docs/en/get_started)
- **Examples:** `scripts/` and `examples/` — Qwen3-4B, Qwen3-30B-A3B MoE, GLM-4.5-Air.

## Join the community

Apache 2.0. Built on slime, Megatron-LM, vLLM.

- **Code and docs:** [github.com/vllm-project/vime](https://github.com/vllm-project/vime)
- **Contributing:** issues and PRs; pre-commit for style.
- **Feedback:** experience, perf data, feature requests on GitHub.

## Acknowledgments

**Contributors:** Ao Shen, kaiyuan, princepride, Dakai An, knlnguyen1802, gcanlin, SamitHuang, Meihan-chen.

Thanks to maintainers of [slime](https://github.com/THUDM/slime), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [vLLM](https://github.com/vllm-project/vllm). Organizing support: Kaichao You, Roger Wang, Hongsheng Liu, Xiyuan Wang.
