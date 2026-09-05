---
source: https://vllm.ai/blog/2026-09-01-minimax-h3-production-serving
lang: en
fetched: 2026-09-04
---

# MiniMax H3 on vLLM-Omni: From System-Wide Optimization to Real-Time Serving with FastVideo’s FastH3

Chinese: [zh/vllm/blog/serving/minimax-h3.md](../../../../zh/vllm/blog/serving/minimax-h3.md)

2026-09-01. **vLLM-Omni Team**. Two-stage story: first shrink overhead across the **complete** MiniMax H3 serving stack, then fuse FastVideo’s four-step **FastH3** so a complete MP4 is ready **faster than playback**. **Real-time** here means that complete-response criterion — **not** streaming delivery and **not** time to first frame. Same Omni family: [vllm-omni.md](vllm-omni.md), [omni-layerwise-offload.md](omni-layerwise-offload.md), [omni-diffusion-cache.md](omni-diffusion-cache.md). Text sibling: [minimax-m3.md](minimax-m3.md). Study note; do not scrape the post’s MP4s or SVG figures. No `assets/vllm/blog/serving/minimax-h3/` in-tree.

**TL;DR from the page:**

- Base H3 lane (dense BF16, 50 sigma / 49 DiT forwards, 8× B300, complete-MP4 wall): Diffusers **82.239 s** / **151.699 GiB** HBM vs vLLM-Omni **56.917 s** / **128.232 GiB** → **30.8%** lower latency, **1.445×**, labeled **lossless** (no quant / sparse attn / fewer steps; **not** bitwise-identical).
- FastH3 lane is a **different** SHA / prompt / seed / artifact. **Do not** divide one lane by the other. Eight-B300 FastH3: complete **10.125 s** MP4 in **8.678–8.710 s**.
- `RTF_client = T_client / T_media` with `T_media = max(T_video, T_audio)`. Criterion: `RTF_client <= 1.0`.
- FastH3 + DLO is **unsupported**. FastH3 + disaggregated encoder is **not yet qualified**.

## 1. Why MiniMax H3 serving is a system-wide problem

One request crosses a large Qwen3-VL encoder, a long-sequence audio-video DiT, separate video and audio VAEs, device/process boundaries, then H.264/AAC muxing. Optimizing only the DiT leaves latency elsewhere.

```text
request -> encoder -> joint audio/video DiT -> video + audio VAEs
        -> GPU output preparation -> D2H/IPC -> H.264/AAC MP4
```

**Figure 1** (not copied): multimodal inputs → shared encoders → joint audio-video diffusion → VAE decode → MP4. Text uses the H3/Qwen3-VL encoder; visual/audio conditions also use corresponding VAEs. Conditioning and noisy target latents pack into one sequence for joint denoising.

Released checkpoints cover three tasks:

| Task | Inputs | Typical use |
|---|---|---|
| T2VA | Text | Creative generation and synthetic media |
| FL2VA | Text plus first/last images | Controlled transitions and image animation |
| Ref2VA | Mixed image, video, and audio references | Consistent editing and reference-guided generation |

DiT dominates the base schedule, but encoder residency, VAE decode (once denoising shortens), and frame transport/mux remain. Sources named on the page: [MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3), [vLLM-Omni recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md), [Diffusers pipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3).

## 2. Benchmark contract and evidence boundaries

Two **separate** evidence lanes. Do **not** derive a base-to-FastH3 speedup.

| Evidence lane | Purpose |
|---|---|
| Base H3: Diffusers versus vLLM-Omni | System-wide runtime under a **50-point dense BF16** schedule |
| FastH3 duration sweep | Absolute low-latency / complete-response real-time with **four** DiT forwards |

### 2.1 Frozen controls

| Control | Base H3 system lane | FastH3 lane |
|---|---|---|
| Hardware | 8× NVIDIA B300 | 8× NVIDIA B300 |
| Task | T2VA through FL2VA partition | Dense/Data-Free T2VA only |
| Resolution / FPS | 1344×768 / 24 FPS | 1344×768 / 24 FPS |
| Source | vLLM-Omni [`b81aeb7`](https://github.com/vllm-project/vllm-omni/commit/b81aeb7b86837f6fe8956f3aef83798ad26c5a26) | vLLM-Omni [`86b85c07`](https://github.com/vllm-project/vllm-omni/commit/86b85c078bc041e04aee4c4d9167fb10fb1994c7) |
| Model | MiniMax H3 [`42ed227e`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe) | Same base model plus pinned FastH3 artifact |
| Prompt / seed | Official `case-T2VA` expanded prompt, SHA-256 `98f36b...f06`; seed **0** | Fixed FastH3 prompt; seed **1101** |
| Schedule | 50 sigma points / 49 DiT forwards | 5 sigma points / 4 DiT forwards |
| Topology | Encoder TP8; DiT USP8, Ring1; VAE PP8 tile | One replica; encoder TP8; DiT USP8, Ring1; VAE PP8 tile |
| Attention | Dense BF16 `TRTLLM_ATTN`, Fast Ulysses | Dense `TRTLLM_ATTN`, Fast Ulysses |
| Repetitions | One excluded full-shape warmup, then measured requests | One excluded feasibility request per shape, then two interleaved runs per duration |

Both lanes time from **synchronous request submit** through **complete MP4 receipt**. Downloads, startup, compilation, and excluded warmup are outside that interval. Accepted output must decode as H.264 + stereo 32 kHz AAC, expected frame count/FPS, nonzero video variance and audio RMS, and pass prompt-adherence review.

Failed media check, missing audio, OOM, accelerator error, or unexpected fallback **stops** that profile before repeated measurement.

Other hardware is **recipe coverage**, not another result matrix: H200/datacenter CUDA, RTX PRO 5000, RTX 4090, RTX 5090, GB10, ROCm (`gfx942` / `gfx950`) — links in the Omni MiniMax-H3 recipes.

## 3. System-wide optimization with vLLM-Omni

Base H3 lane keeps released BF16 weights, 50 sigma points, and dense attention. Optimizations follow the execution path.

### 3.1 Long-sequence attention and communication

Canonical packed sequence: **58,758** valid tokens in a **58,816**-token aligned buffer.

- [`TRTLLM_ATTN`](https://github.com/vllm-project/vllm-omni/pull/5283) gets valid sequence lengths; [packed-sequence refinement](https://github.com/vllm-project/vllm-omni/pull/5779) drops structural suffix padding.
- [Rank-local boundaries](https://github.com/vllm-project/vllm-omni/pull/6173) build only local embedding/RoPE rows and gather the compact **128-channel** projection rather than the **5,376-channel** hidden state.
- [Fast Ulysses](https://github.com/vllm-project/vllm-omni/pull/6340) uses NCCL SymmetricMemory so shards arrive in the layout attention wants — no separate relayout around all-to-all.

### 3.2 Fused DiT operators

The 49-forward loop is full of small ops around GEMMs. Fusions: Q/K RMSNorm + RoPE ([#5990](https://github.com/vllm-project/vllm-omni/pull/5990)); FP32 modulation / norm / residual ([#6281](https://github.com/vllm-project/vllm-omni/pull/6281), [#6878](https://github.com/vllm-project/vllm-omni/pull/6878)); fused SwiGLU instead of SiLU + multiply ([#6283](https://github.com/vllm-project/vllm-omni/pull/6283)).

### 3.3 Parallel and fused VAE decoding

Video and audio decode independently after denoising. VAE patch parallelism tiles the video decoder across eight GPUs. [Exact VAE operator path](https://github.com/vllm-project/vllm-omni/pull/6607): decoder-block materialization, fused Q/K norm + RoPE, fused SwiGLU, scaled residual updates; eager fallbacks for unsupported layouts.

### 3.4 GPU output preparation, transport, and MP4

A request is not done until hundreds of frames leave the GPU. Convert once:

1. [GPU output preparation](https://github.com/vllm-project/vllm-omni/pull/6824): decoded FP32 BCTHW → contiguous uint8 BTHWC (**75%** smaller video payload before transfer).
2. Pinned D2H and worker-to-engine IPC for the compact payload.
3. [Direct-planar encoding](https://github.com/vllm-project/vllm-omni/pull/6288), [persistent parallel converter](https://github.com/vllm-project/vllm-omni/pull/6499), [transported strided RGB planes](https://github.com/vllm-project/vllm-omni/pull/6776) feed H.264 without another full interleaved RGB buffer.

`FP32 BCTHW -> uint8 BTHWC -> pinned D2H/IPC -> planar frames -> H.264/AAC MP4`

### 3.5 Measured base H3 result

Same prompt/seed, 50 sigma points, complete-MP4 boundary. Diffusers: replicated weights + native context parallelism. vLLM-Omni: encoder TP8, DiT USP8/Ring1 + Fast Ulysses, VAE PP8 tile, `TRTLLM_ATTN`.

| Runtime | Model execution (s) | Prompt (s) | DiT total / per-forward (s) | Video / audio VAE (s) | MP4 (s) | Client E2E (s) | Peak HBM/rank (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Diffusers | - | - | - | - | - | **82.239** | 151.699 |
| vLLM-Omni | **54.246** | 0.057 | 51.800 / 1.057 | 0.952 / 0.055 | 1.528 | **56.917** | 128.232 |

Page embeds a MiniMax-H3 model-card sample MP4 and a vLLM-Omni baseline MP4; not copied here.

Lossless in this sentence means the speedup does **not** rely on quantization, sparse attention, cache reuse, or fewer denoising steps. Different kernels / FP reduction order can still perturb the diffusion trajectory.

These cuts shrink overhead **around** denoising. FastH3 then cuts the loop itself from 49 forwards to four.

## 4. Scaling the general H3 serving architecture

DLO and disaggregated encoding change **capacity and placement**. Quantized weights and approximate attention trade **numerical fidelity**. These paths did **not** produce the FastH3 numbers in §6.

### 4.1 Distributed Layerwise Offload

[DLO](https://vllm.ai/blog/2026-08-17-distributed-layerwise-offload) (local: [omni-layerwise-offload.md](omni-layerwise-offload.md)) keeps a bounded window of DiT layers in HBM and streams the rest from host. AllGather reconstructs active layers from host shards; rank-local streams tensors from each rank’s normal loader.

**Figure 2** (DLO article still; not recopied): next layer prepared while current layer computes.

#### 8× B300 BF16 DLO Pareto

Official BF16 MiniMax-H3 FL2VA checkpoint (5.175 s, 1344×768, SP8/Ulysses8/Ring1/DP1/TP1, AllGather, CUDNN attention). First request excluded (lazy CUDA/cuDNN/JIT); remaining two averaged. Video/audio have expected shapes.

**Figure 3** (not copied): latency–memory Pareto. *r* = resident DiT blocks. Non-dominated: no offload, then 50, 35, 30, and 0 leading blocks resident; 40, 20, and 10 are dominated. At **r = 35**, DLO lowers reported HBM by **37.5%** for a **5.1%** latency cost; **r = 0** is the min-memory endpoint.

### 4.2 Disaggregated encoding

H3 keeps ~**51.5 GB** of Qwen3-VL encoder weights in BF16. [Disaggregated encoder](https://github.com/vllm-project/vllm-omni/pull/5885) moves that one-shot encoder into its own vLLM stage (placement, TP, replicas, queue, kernels, prefix cache). Orchestrator merges layer-50 hidden states + token-role tags with original media before DiT/VAE.

**Figure 4** (not copied): encoder and diffusion scale independently. Merged single-node recipe returns conditioning through the orchestrator and keeps diffusion inline; it does **not** configure OmniConnector. SHM/RDMA is future cross-node work in [RFC #5707](https://github.com/vllm-project/vllm-omni/issues/5707).

### 4.3 Optional quantization and attention acceleration

§3 uses dense BF16 attention and released checkpoint precision. Extra paths are **separate quality-performance profiles**, not lossless runtime gains.

#### Weight and activation quantization

- **Online FP8.** Merged [global FP8 path](https://github.com/vllm-project/vllm-omni/pull/5910): from BF16 checkpoint, quantize eligible DiT and Qwen3-VL text-decoder linears at load. Embeddings, norms, RoPE, vision tower, both VAEs, and precision-sensitive projections keep declared precision.
- **SVDQuant NVFP4 W4A4.** Merged [offline loader](https://github.com/vllm-project/vllm-omni/pull/6162): NVFP4 W4A4 base GEMM + BF16 low-rank correction. Evidence so far: checkpoint/correctness compatibility; native fused residual-GEMM performance path is **future work**.

**Figure 5** (not copied): online FP8 vs offline SVDQuant.

A quantized profile must report peak HBM, startup host RAM, checkpoint storage, latency, **and** same-seed video/audio quality. Capacity win ≠ latency win. Loader correctness ≠ fused-kernel gain.

#### B300 Online FP8 capacity and latency

Dense, resident; isolates Online FP8 from released BF16. 8× B300, Ulysses8/Ring1 + Fast Ulysses, encoder TP8, VAE PP8 tile, CUDNN attention, 10-second 1344×768 / 24 FPS, 50 requested sigma points (49 DiT forwards). One warmup excluded; mean of three measured requests. “Stage generation” = native diffusion-stage timer; E2E = offline client wall through returned video/audio tensors, **excluding MP4 muxing**.

| Weights | Stage generation (mean, n=3) | E2E (mean, n=3) | Peak HBM / rank | Result |
|---|---:|---:|---:|---|
| BF16 | 52.572 s | 53.118 s | 87.16 GiB | Lossless baseline |
| Online FP8 | **49.769 s** | **50.331 s** | **53.27 GiB** | 5.3% lower stage time; 38.9% lower peak HBM |

Every measured request returned **243** RGB frames at 1344×768 and 32 kHz stereo audio. Distinct seeds → shape + successful generation, **not** pixelwise equivalence to BF16.

#### Quantized and Sparse Attention in `TRTLLM_ATTN`

- **SAGE**: quantize QK and PV paths to FP8.
- **Skip-Softmax**: use QK to skip unimportant Softmax and P×V ([BLASST](https://arxiv.org/abs/2512.12087)).

**Figure 6** (not copied): SAGE around Skip-Softmax main loop.

| Attention policy | SAGE configuration | Skip-Softmax configuration | Model execution | Speedup | LPIPS vs. baseline |
|---|---|---|---:|---:|---:|
| TRTLLM Baseline | Off | Off | 54.246 s | 1.000x | 0 |
| SAGE FP8 | `dtype_qk=fp8_e4m3`, `q_block_size=1`, `k_block_size=16` | Off | 44.787 s | **1.211x** | 0.3697 |
| Skip-Softmax | Off | threshold 0.05; disabled until 0.97 | 50.029 s | **1.084x** | 0.0917 |
| SAGE + Skip-Softmax | same SAGE | same Skip-Softmax | 43.867 s | **1.237x** | 0.3750 |

Measured Skip-Softmax is **conservative** for video quality. Higher threshold or more denoising steps can trade quality for speed. Controls: [TRTLLM attention guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/attention_backends/trtllm.md).

#### Cache-DiT

[Cache-DiT](https://github.com/vllm-project/vllm-omni/pull/5853) is a **request-level cache policy**, not an attention backend. For H3, `quality=high` enables dynamic per-step reuse; `quality=lossless` restores the reference path. Hit/miss is deployment-dependent; **not** in the attention A/B above.

### 4.4 Compatibility boundaries

| Combination | Status for this article |
|---|---|
| Base H3 + DLO | Supported through maintained H3 recipes; qualify topology locally |
| Base H3 + DLO + online FP8 | Supported, including AllGather via [#6279](https://github.com/vllm-project/vllm-omni/pull/6279); still qualify locally |
| Base H3 + disaggregated encoder | Merged single-node path |
| FastH3 + DLO | **Unsupported**: FastH3 fuses in `load_weights()`; offload installs a different host-weight path |
| FastH3 + disaggregated encoder | **Not yet qualified**; unused for the reported FastH3 result |

**Step execution sidebar.** H3 can admit/abort between denoise steps ([#5810](https://github.com/vllm-project/vllm-omni/pull/5810)); existing co-batching tests did **not** improve latency. Request mode remains the recommendation while cancellation/reclamation and small under-utilized workloads sit in [issue #5700](https://github.com/vllm-project/vllm-omni/issues/5700).

## 5. From system optimization to FastH3

[FastH3](https://haoailab.com/blogs/fasth3-preview/) is FastVideo’s four-step **DMD2 student** of MiniMax H3. Reuses encoder, video VAE, audio VAE, tokenizers, schedulers; denoising loop is **four** transformer forwards over **five** sigma positions.

- **FastVideo** develops/releases the distilled student and adapter artifacts.
- **vLLM-Omni** validates, fuses while the checkpoint streams in, shards fused weights, serves through the optimized attention / VAE / transport / MP4 path.

FastH3 is **not** a request-switchable LoRA. Artifact includes full-rank deltas and replacement weights an ordinary LoRA layer cannot represent. Fuse **before** sharding.

**Figure 7** (not copied): Turbo leaves base weights and applies request-selected A/B sidecars; FastH3 fuses low-rank + full-rank into a dedicated student. Turbo [#6476](https://github.com/vllm-project/vllm-omni/pull/6476), DLO support [#6550](https://github.com/vllm-project/vllm-omni/pull/6550), FastH3 [#6714](https://github.com/vllm-project/vllm-omni/pull/6714).

| Profile | Activation model | Task scope | When to choose it |
|---|---|---|---|
| Base H3 | Released checkpoint | T2VA, FL2VA, Ref2VA | Full task coverage + general scaling lane |
| Turbo | Request-switchable adapter | T2VA and FL2VA | One service needs request-time switching or FL2VA |
| FastH3 | Load-time-fused dedicated student | Dense/Data-Free T2VA | Lowest validated latency on a dedicated T2VA endpoint |

FastH3 v1 **rejects** offload and VSA variants, accepts **T2VA only**, requires its four-forward schedule and checkpoint flow shifts, and cannot take another request-time LoRA. Serving contracts, not tuning knobs.

## 6. Real-time FastH3 serving on B300

Absolute FastH3 on vLLM-Omni `86b85c07`. Do not divide by the §3 base-H3 result.

### 6.1 Pin the artifact

Dense/Data-Free artifact pinned to Hugging Face revision `bcf40ca6f457ed66f8badf13514943e390205fca`:

```bash
FASTH3_REV=bcf40ca6f457ed66f8badf13514943e390205fca
FASTH3_DIR=/models/FastH3-LoRA

hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  dense-datafree/adapter_model.safetensors \
  --revision "$FASTH3_REV" \
  --local-dir "$FASTH3_DIR"

echo "4ce198c83132251b7fd0de2503823aa49c53983f068318f66cb19eaefb7fcc12  $FASTH3_DIR/dense-datafree/adapter_model.safetensors" \
  | sha256sum -c -
```

Adapter is **1,485,626,152** bytes. Pin **revision and checksum**. Repo name still says `Preview-v1`; matching Omni integration is merged.

### 6.2 Serve and request

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "$H3_MODEL" --omni \
  --host 127.0.0.1 --port 8095 --trust-remote-code \
  --task-type fl2va --served-model-name MiniMaxAI/MiniMax-H3 \
  --num-gpus 8 --usp 8 --ring 1 --ulysses-a2a-permute \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend TRTLLM_ATTN \
  --lora-path "$FASTH3_DIR/dense-datafree/adapter_model.safetensors"
```

```bash
curl -sS -X POST http://127.0.0.1:8095/v1/videos/sync \
  -F 'prompt=In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.' \
  -F 'width=1344' -F 'height=768' -F 'aspect_ratio=16:9' -F 'fps=24' \
  -F 'num_inference_steps=4' -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":10.0,"flow_shift":12.0,"audio_flow_shift":3.0}' \
  -o fasth3_10s.mp4
```

One FastH3 replica; encoder TP8; DiT DP1 × TP1 × USP8 with Ring1 and Fast Ulysses; VAE PP8 tile; `TRTLLM_ATTN`; compact output/MP4 path.

### 6.3 Ten-second critical path

Profiler timers from a **separate** instrumented pass; **clean E2E** carries the latency claim.

> **Raw benchmark bundle — pending publication gate.** Stable bundle not yet published. Before publication, the [evidence-handoff requirement](https://github.com/vllm-project/vllm-project.github.io/pull/315#issuecomment-5459581336) must become a bundle URL (raw clean/profiler samples, logs, environment manifest, media metadata/hashes, topology for critical-path row **and** duration sweep).

| Encoder | DiT total / 4 / per-forward | Video + audio VAE | Derived transport | CPU MP4 | Profiled E2E | Clean E2E | Peak HBM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.052 s | 5.532 s / 4 / 1.383 s | 1.247 s combined | 0.881 s | 0.868 s | 8.629 s | **8.678 / 8.710 s** | 94.1 GiB/GPU reserved |

### 6.4 Five-, ten-, and fifteen-second sweep

Fixed: prompt, seed, resolution, artifact, schedule, topology, attention, VAE, output path, CPU affinity. H3 aligns requested durations to **124 / 243 / 362** frames.

| Requested / aligned | Video / audio duration | DiT total / per-forward | Combined VAE | Transport + MP4 | Clean E2E | Client RTF | × real time |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 s / 124 | 5.167 / 5.175 s | 2.806 s / 0.702 s | 0.637 s | 0.929 s | 4.602 / 4.396 s | **0.889 / 0.849** | **1.125 / 1.177** |
| 10 s / 243 | 10.125 / 10.125 s | 5.532 s / 1.383 s | 1.247 s | 1.749 s | 8.678 / 8.710 s | 0.857 / 0.860 | 1.167 / 1.163 |
| 15 s / 362 | 15.083 / 15.083 s | 9.517 s / 2.379 s | 1.861 s | 2.484 s | 14.177 / 14.059 s | 0.940 / 0.932 | 1.064 / 1.073 |

All six measured requests satisfy `RTF_client <= 1.0`.

### 6.5 Representative outputs and quality boundary

Supplied FastH3 clips are **1280×736** presentation examples, **not** the 1344×768 timing artifacts of §6.4.

| Request | Frames | MP4 duration | Resolution / FPS |
|---:|---:|---:|---|
| 5 s | 124 | 5.184 s | 1280×736 / 24 FPS |
| 10 s | 243 | 10.144 s | 1280×736 / 24 FPS |
| 15 s | 362 | 15.104 s | 1280×736 / 24 FPS |

Publication-grade timing/media evidence still waits on the §6.3 raw-bundle gate.

| Quality gate | Status |
|---|---|
| Repeated same-seed FastH3 output | Byte-identical in the measured runs |
| Media structure | Expected frames/FPS, H.264, stereo AAC, nonzero video/audio signal |
| Matched base-versus-FastH3 multi-seed quality | **Pending; no parity claim** |

New tail after fewer denoise steps: on the 10 s profile, combined VAE + derived transport + CPU MP4 ≈ **three seconds** on the instrumented path. [RFC #6872](https://github.com/vllm-project/vllm-omni/issues/6872) proposes overlapping VAE chunks, D2H/IPC, and encoding. Optimistic B300 ceilings: ~**0.87 s** (~10% E2E) overlapping transport with encoding; ~**1.75 s** (~20% E2E) also overlapping incremental VAE decode; go/no-go targets at least **5%** and **10%** E2E. Draft [PR #6885](https://github.com/vllm-project/vllm-omni/pull/6885): **0.8847 s** (**26.57%**) VAE-to-complete-MP4 reduction on a **four-L20X** feasibility run with exact media parity — **not** a B300 production result.

## 7. Production guidance and limitations

| Requirement | Recommended profile |
|---|---|
| Full T2VA, FL2VA, and Ref2VA coverage | Base H3 with the system-wide stack |
| Request-time adapter switching or FL2VA with four-forward Turbo | Separate Turbo service |
| Lowest validated T2VA complete-response latency | Dedicated FastH3 service from §6 |
| Host-memory-driven fit or independently scaled encoder | Base H3 DLO or disaggregated-encoder lane; qualify locally |

Do **not** combine the reported FastH3 profile with DLO, VSA, quantization, cache policies, alternative Ulysses transports, or encoder disaggregation without a new correctness / quality / memory / latency qualification. Living tracker: [issue #5700](https://github.com/vllm-project/vllm-omni/issues/5700) — it can lag merged code. Check linked PRs and maintained recipes.

License: [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). Commercial / hosted-service operators should review territorial, attribution, revenue, acceptable-use, and safeguard terms with counsel.

Post-training: vLLM-Omni can serve H3 rollouts in [VeRL-Omni](https://github.com/verl-project/verl-omni); training is ecosystem coverage, **not** this serving benchmark.

## 8. Conclusion and focused future work

System-wide stack makes the complete H3 pipeline efficient. FastVideo’s four-forward student then puts dedicated T2VA into faster-than-playback complete-response generation on the measured B300 system.

Remaining work named on the page: FastH3 VSA variants and native fused NVFP4 kernels; [Sol-Attn](https://github.com/vllm-project/vllm-omni/pull/5851) on-the-fly sparse attention across Blackwell + multi-seed; matched base/FastH3 multi-seed quality; [chunkwise VAE→transport→MP4](https://github.com/vllm-project/vllm-omni/issues/6872) plus a GPU encoder; H3 post-training across VeRL-Omni, [UniRL](https://github.com/Tencent-Hunyuan/UniRL), [RLinf](https://github.com/RLinf/RLinf); qualify FastH3 with encoder disaggregation rather than inferring it.

## Acknowledgments

Builds on vLLM, vLLM-Omni, VeRL-Omni, MiniMax H3, [FastVideo](https://github.com/hao-ai-lab/FastVideo), FastH3, Diffusers, NVIDIA. FastVideo team for [open-sourcing FastH3](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA). Named GitHub handles on the page: Isotr0py (base H3); lishunyang12, evanchueng, Gaohan123, david6666666 (DLO / base / online-FP8); gcanlin, yuanwu2017 (encoder disaggregation); bobboli, fan2956, mo-ke-ke, mglyn, MosCloud, ultism (attention, fused kernels, quantization, VAE, transport, media); princepride (FastH3 integration + B300 validation); NancyFyong, mengchengTang (VeRL-Omni). Hongsheng Liu and Roger Wang for general support and blog prep.

## Appendix A. Reproducibility

### A.1 Timing hierarchy

Nested; **do not add** parent and child:

| Boundary | Scope |
|---|---|
| Client | Request submission through complete MP4 receipt |
| Request | Orchestrator lifetime across stages |
| Stage | One independently scheduled engine/device group |
| Engine | Queue, model execution, output-ready wait, formatting |
| Profiler | Prompt, DiT, VAE method boundaries inside engine execution |
| Server | H.264/AAC encode and mux after the final stage |

Per-forward denoise time divides by **actual DiT forward count**, not requested sigma-position count. Profiler values come from separate diagnostic requests and do **not** replace unprofiled client latency.

### A.2 Base H3 vLLM-Omni reproduction

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "$H3_MODEL" --omni \
  --host 127.0.0.1 --port 8093 --trust-remote-code \
  --task-type fl2va --num-gpus 8 --usp 8 --ring 1 \
  --ulysses-a2a-permute --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend TRTLLM_ATTN
```

Canonical request: prompt and seed in §2, 50 requested sigma points, flow shift 12, audio flow shift 3, 10-second target.

## References

- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [FastH3 technical overview](https://haoailab.com/blogs/fasth3-preview/)
- [FastH3 four-step adapter](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA)
- [MiniMax H3 model](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Diffusers MiniMax H3 pipeline](https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/minimax_h3)
- [MiniMax H3 serving recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md)
- [Distributed Layerwise Offload](https://vllm.ai/blog/2026-08-17-distributed-layerwise-offload)
- [Feature compatibility tracker](https://github.com/vllm-project/vllm-omni/issues/5700)
- [Chunkwise output pipeline RFC](https://github.com/vllm-project/vllm-omni/issues/6872)
- [VeRL-Omni](https://github.com/verl-project/verl-omni) · [UniRL](https://github.com/Tencent-Hunyuan/UniRL) · [RLinf](https://github.com/RLinf/RLinf)
