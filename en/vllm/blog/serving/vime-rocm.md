---
source: https://vllm.ai/blog/2026-07-10-vime-rocm
lang: en
fetched: 2026-09-04
---

# vime on ROCm: `vllm/vime-rocm`, Qwen3-8B ~4100 tok/gpu/s on MI355X

Chinese: [zh/vllm/blog/serving/vime-rocm.md](../../../../zh/vllm/blog/serving/vime-rocm.md)

2026-07-10. **AMD contributors & vime community**. Mainline launch: [vime.md](vime.md) (2026-06-09). Image: `vllm/vime-rocm`. Tutorial: [amd_tutorial.md](https://github.com/vllm-project/vime/blob/main/docs/en/platform_support/amd_tutorial.md). slime: [THUDM/slime](https://github.com/THUDM/slime). Hardware-plugin cousin: [../../architecture/hardware-plugin.md](../../architecture/hardware-plugin.md). Engine-side pause / weight APIs (different layer): [native-rl.md](native-rl.md). Bitwise on-policy is **not** this post: [bitwise-rl.md](bitwise-rl.md). Study note; MI355X Qwen3-8B numbers on the page, not your SLA.

Since the vime launch, AMD worked with the vime team on ROCm: end-to-end on Instinct, upstream ROCm-specific fixes, prebuilt container so AMD users skip a from-source build. Same knob names as CUDA ≠ same kernels. Compare their `train_rollout_logprob_abs_diff`; do not assume bitwise.

Local figures (copyright remains with the original site; study copies):

![data buffer](../../../../assets/vllm/blog/serving/vime-rocm/01-data-buffer.png)

![image](../../../../assets/vllm/blog/serving/vime-rocm/02-image.png)

![image 1](../../../../assets/vllm/blog/serving/vime-rocm/03-image-1.png)

![image 2](../../../../assets/vllm/blog/serving/vime-rocm/04-image-2.png)

## TL;DR

- Prebuilt **`vllm/vime-rocm`**. Code at `/root/vime`. vLLM + Megatron-LM preinstalled. W&B online mode: valid `WANDB_API_KEY` required.
- Qwen3-8B on **MI355X**, 100 steps: ~**4100** `tokens_per_gpu_per_second`, slight upward trend. `train_rollout_logprob_abs_diff` ~**0.012**, slight downward trend. `raw_reward` ~**0 → 0.5–0.6**.
- Launcher: `NUM_ROLLOUT=100 VISIBLE_GPUS=0,1 bash scripts/run-qwen3-8B-amd.sh`. TP=2, one vLLM engine, colocate, DP=1. ~**230 GB** across the two GPUs.
- On ROCm, select GPUs with **`HIP_VISIBLE_DEVICES`**, not only `CUDA_VISIBLE_DEVICES`. Conversion uses `--no-gradient-accumulation-fusion --attention-backend flash`.
- **Caveats:** ROCm launcher sets `EVAL_ARGS=()` — training-set `raw_reward` is not held-out eval. logprob ~0.012 is on par with their NVIDIA reports, **not** bit-exact. R3 for AMD MoE, full Router / PD, FP8 pipeline: roadmap.
- Also validated (named, not charted here): Qwen3-4B, Qwen3-8B dense, Qwen3-30B-A3B MoE.

## vime, restated for ROCm

vime was [announced](vime.md) in June 2026. With ROCm, the same RL post-training workflows run natively on AMD Instinct.

**Figure.** vime architecture (data buffer between train and rollout).

slime’s three-stage, decoupled train-inference design. Difference vs slime-native: rollout backend is **vLLM**, not SGLang.

- **Training (Megatron):** parameter updates; sync weights to rollout.
- **Rollout (vLLM + Router):** sampling; reward / verifier signals.
- **Data Buffer:** prompt injection and custom rollout logic.

This whole pipeline is what they say is validated end-to-end on ROCm.

## Why AMD Instinct

RL post-training is memory-heavy. Each step holds training-side weights (Megatron format) **and** inference-side KV cache (vLLM rollouts). In colocated mode they compete for one device memory pool. The page’s reasons MI300X / MI355X fit:

- **Large unified HBM.** MI300X **192 GB** HBM3/GPU; MI355X **288 GB**. Large models can train without aggressive tensor parallelism just to spread memory — simpler topology, better cluster utilization.
- **Bandwidth.** HBM3 **>5 TB/s** aggregate on MI300X; HBM3E **8 TB/s** on MI355X. RL rollout (autoregressive decode at scale) is memory-bandwidth-bound: each decode step loads KV cache and weights from HBM. Higher bandwidth shortens the rollout phase that dominates most RL step times.
- **Open stack.** ROCm (HIP, LLVM, MIOpen). vLLM and PyTorch support ROCm natively, so vime inherits the vLLM rollout stack **without a separate code path**. Same names, not a promise of identical kernels.

## Training details (what they wired)

- **Megatron-LM backend.** ROCm-compatible fork plus a small patch that **guards CUDA fused-kernel initialization on non-CUDA builds**. Training loop: ROCm-compatible Megatron patches and ROCm-specific launch flags. Gradient accumulation uses the **native PyTorch** path (supported on ROCm). HuggingFace → Megatron `torch_dist` conversion runs on **one GPU** and produces a layout Megatron loads at job start.
- **Colocated weight sync.** Megatron and vLLM share the GPU pool. After each optimizer step, Megatron syncs updated weights to vLLM via **IPC** — no network round-trip. On ROCm, `torch.cuda.get_device_properties(i).uuid` returns stable, process-consistent device UUIDs, so vime’s UUID-keyed IPC routing works **without modification**.
- **GPU visibility and Ray.** ROCm uses `HIP_VISIBLE_DEVICES`. The vime launch script sets it **alongside** `CUDA_VISIBLE_DEVICES` so the Megatron actor and the vLLM subprocess see the same ordinals. Ray’s AMD GPU manager is configured **not** to override those masks. Container starts with `--ulimit nofile=1048576:1048576` because Ray needs the FD limit when spawning the full actor set.

## Getting started

Prebuilt container; skip a from-source ROCm stack if the image matches your host.

### Launch the container

```bash
# Pull the ROCm image
docker pull vllm/vime-rocm
# Start the container
docker run -d --name vime --ulimit nofile=1048576:1048576 \
  --ipc=host --network=host --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video --privileged \
  -e WANDB_API_KEY=$wandb_key vllm/vime-rocm
# The launch script enables W&B online mode, so a valid WANDB_API_KEY is required.

# Enter the container
docker exec -it vime bash
```

Image contents: vLLM, Megatron-LM, vime at `/root/vime`.

### Model and dataset

```bash
# Download model weights (Qwen3-8B)
hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B
# Download training dataset (dapo-math-17k)
hf download zhuzilin/dapo-math-17k --repo-type dataset --local-dir /root/dapo-math-17k
```

### Convert to Megatron `torch_dist`

Load Qwen3-8B model config, then convert. **On ROCm, `HIP_VISIBLE_DEVICES` selects GPUs** (not only `CUDA_VISIBLE_DEVICES`).

```bash
cd /root/vime && source scripts/models/qwen3-8B.sh
HIP_VISIBLE_DEVICES=0 PYTHONPATH=/root/vime:/root/Megatron-LM \
  torchrun --nproc-per-node=1 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
  --no-gradient-accumulation-fusion --attention-backend flash \
  --hf-checkpoint /root/Qwen3-8B --save /root/Qwen3-8B_torch_dist
```

Flags to keep: `--no-gradient-accumulation-fusion`, `--attention-backend flash`.

### Launch RL training

```bash
NUM_ROLLOUT=100 VISIBLE_GPUS=0,1 bash scripts/run-qwen3-8B-amd.sh
```

Full colocated pipeline: vLLM rollout workers, GRPO loop, on-policy rollout → train → weight-sync.

**Configuration notes:**

- `VISIBLE_GPUS` — two free GPU indices; the script masks to these. **TP=2**, single vLLM engine, **colocate**, **DP=1**.
- `NUM_ROLLOUT` — training steps. Default **3** is a smoke test; the charts use **100**.
- ~**230 GB** across the two selected GPUs. Launch only where that memory is free.

Rerun with a different `NUM_ROLLOUT`: clear the save dir or checkpoints mismatch:

```bash
rm -rf /root/Qwen3-8B_vime/
```

## Performance results

Named models they ran with this runbook: Qwen3-4B, Qwen3-8B (dense), Qwen3-30B-A3B (MoE). Charts below are the **Qwen3-8B** example.

**Figure.** Throughput on MI355X, Qwen3-8B.

Throughput sustains about **4,100** `tokens_per_gpu_per_second` across **100** training steps, slight upward trend. Their reading: the policy learns more predictable outputs; shorter or more uniform generations cut decode variance so vLLM batches better. That is a training-dynamics story, not a kernel-only speedup claim.

**Figure.** `train_rollout_logprob_abs_diff` on the same run.

The metric (training-side logprobs vs rollout-side) holds around **0.012** and trends slightly down. Weight sync Megatron → vLLM is what they credit for not letting logprob drift corrupt the policy gradient. Stable low diff is a prerequisite for GRPO; they call this magnitude **on par with reported NVIDIA numbers**. It is **not** `kl_div == 0.0` bitwise (that bar is [bitwise-rl.md](bitwise-rl.md)).

**Figure.** `raw_reward` on sampled training prompts.

Starts near **0** at step 0, climbs to about **0.5–0.6** by step 100. Fresh policy on dapo-math-17k competition problems solves almost none. Rising training reward = optimization progress **on the sampled training prompts**. The ROCm launcher disables evaluation (`EVAL_ARGS=()`); held-out eval is a separate job if you care about generalization.

## Feature support roadmap on AMD

**Today (named):**

- GRPO
- Colocated training and rollout
- Asynchronous (non-colocated) training with disjoint actor and rollout GPU pools
- Megatron-LM training backend
- vLLM rollout backend
- Qwen3 Dense and MoE

**Ahead (named, not claimed live):**

- Full vLLM Router and PD disaggregation
- FP8 pipeline optimization
- **R3 (Rollout Routing Replay) for AMD MoE** — on the CUDA launch ([vime.md](vime.md)) R3 is a measured ~**0.019 → ~0.013** cut; here it is still roadmap
- Async-pipeline performance (logprob divergence, memory leaks)
- Agentic RL: multi-turn tool calling and multi-agent

Goal on the page: keep pace with the vime and vLLM roadmaps.

## Acknowledgments

AMD contributors & vime community — collaboration and support as named on the page (no individual list in this post; the June launch lists vime contributors).

## References

- vime repo: [github.com/vllm-project/vime](https://github.com/vllm-project/vime)
- vime announcement: [vime.md](vime.md) / https://vllm.ai/blog/2026-06-09-announcing-vime
- AMD tutorial: [docs/en/platform_support/amd_tutorial.md](https://github.com/vllm-project/vime/blob/main/docs/en/platform_support/amd_tutorial.md)
- slime: [github.com/THUDM/slime](https://github.com/THUDM/slime)
