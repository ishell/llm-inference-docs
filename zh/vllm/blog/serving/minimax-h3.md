---
source: https://vllm.ai/blog/2026-09-01-minimax-h3-production-serving
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# MiniMax H3：先把整条栈削薄，再让 FastH3 比播放还快

英文对照：[en/vllm/blog/serving/minimax-h3.md](../../../../en/vllm/blog/serving/minimax-h3.md)  
原文：https://vllm.ai/blog/2026-09-01-minimax-h3-production-serving  
2026-09-01。署名 **vLLM-Omni Team**。两段故事：先把完整 MiniMax H3 serving 栈的开销削掉，再接入 FastVideo 的四步 **FastH3**，让整份 MP4 在播放结束前就交出来。文中的 **real-time** 只指这条完整响应合同——**不是**流式送达，**也不是**第一帧。同一条 Omni 线：[vllm-omni.md](vllm-omni.md)、[omni-layerwise-offload.md](omni-layerwise-offload.md)、[omni-diffusion-cache.md](omni-diffusion-cache.md)。文本亲戚：[minimax-m3.md](minimax-m3.md)。原文的 SVG / MP4 不收进仓库；本地也还没有 `assets/vllm/blog/serving/minimax-h3/`。

八卡 B300 上，FastH3 把一份 **10.125 秒** 的完整 MP4 做到 **8.678–8.710 秒**。Base H3 那条 dense BF16、50 个 sigma / 49 次 DiT 前向的对照：Diffusers 客户端 **82.239 s** / **151.699 GiB** HBM，vLLM-Omni **56.917 s** / **128.232 GiB**——延迟低 **30.8%**，**1.445×**。文中把这条叫 **lossless**：不靠量化、稀疏 attention、少步数；**不**等于逐 bit 相同。两条证据车道的 SHA、prompt、seed、制品不同，**禁止**用一条除另一条。`RTF_client = T_client / T_media`，`T_media = max(T_video, T_audio)`，过关是 `RTF_client <= 1.0`。FastH3 + DLO **不支持**；FastH3 + 分离 encoder **尚未合格**。

## 1. 为什么 H3 serving 是整系统的事

一次请求要穿过很大的 Qwen3-VL encoder、长序列音视频联合 DiT、各自独立的 video / audio VAE、设备与进程边界，最后才是 H.264/AAC 封装。只拧 DiT，别处的等待还在。

```text
request -> encoder -> joint audio/video DiT -> video + audio VAEs
        -> GPU output preparation -> D2H/IPC -> H.264/AAC MP4
```

**Figure 1**（未收录）：多模态输入 → 共享 encoder → 联合音视频扩散 → VAE 解码 → MP4。文本走 H3/Qwen3-VL encoder；视觉和音频条件还要走对应 VAE。条件与带噪目标 latent 打成一条 packed sequence，一起去噪。出处： [MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)、[vLLM-Omni recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md)、[Diffusers pipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3)。

发布的 checkpoint 覆盖三件事：

| Task | 输入 | 典型用途 |
|---|---|---|
| T2VA | 文本 | 创意生成、合成媒体 |
| FL2VA | 文本 + 首末帧图 | 可控转场、图驱动动画 |
| Ref2VA | 图 / 视频 / 音频参考混着来 | 一致性编辑、参考引导生成 |

Base 日程里 DiT 仍最大，但 encoder 常驻吃容量；去噪一缩短，VAE 就露出来；原始帧还得过进程边界变成 MP4。所以故事从整条栈开始。

## 2. 基准合同与证据边界

两条车道分开计量，**不要**合成「base → FastH3 加速比」。

| 证据车道 | 用途 |
|---|---|
| Base H3：Diffusers vs vLLM-Omni | **50 点 dense BF16** 日程下的系统运行时 |
| FastH3 时长扫 | 四次 DiT 前向下的绝对低延迟 / 完整响应 real-time |

### 2.1 冻结对照

| 控制项 | Base H3 系统车道 | FastH3 车道 |
|---|---|---|
| 硬件 | 8× NVIDIA B300 | 8× NVIDIA B300 |
| Task | T2VA 经 FL2VA 分区 | 仅 Dense/Data-Free T2VA |
| 分辨率 / FPS | 1344×768 / 24 FPS | 1344×768 / 24 FPS |
| 源码 | vLLM-Omni [`b81aeb7`](https://github.com/vllm-project/vllm-omni/commit/b81aeb7b86837f6fe8956f3aef83798ad26c5a26) | vLLM-Omni [`86b85c07`](https://github.com/vllm-project/vllm-omni/commit/86b85c078bc041e04aee4c4d9167fb10fb1994c7) |
| 模型 | MiniMax H3 [`42ed227e`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe) | 同一底座 + 钉死的 FastH3 制品 |
| Prompt / seed | 官方 `case-T2VA` 展开 prompt，SHA-256 `98f36b...f06`；seed **0** | 固定 FastH3 prompt；seed **1101** |
| 日程 | 50 个 sigma / 49 次 DiT 前向 | 5 个 sigma / 4 次 DiT 前向 |
| 拓扑 | Encoder TP8；DiT USP8, Ring1；VAE PP8 tile | 一份 replica；encoder TP8；DiT USP8, Ring1；VAE PP8 tile |
| Attention | Dense BF16 `TRTLLM_ATTN`，Fast Ulysses | Dense `TRTLLM_ATTN`，Fast Ulysses |
| 重复 | 排除一次全形状 warmup，再测 | 每个形状排除一次可行性请求，再每个时长交错两轮 |

两条车道都从 **同步提交请求** 计到 **收到完整 MP4**。下载、启动、编译、被排除的 warmup 不进这段。验收：能解成 H.264 + 立体声 32 kHz AAC，帧数 / FPS 对得上，视频方差和音频 RMS 非零，还要通过 prompt 贴合评审。媒体检查失败、缺音频、OOM、加速器错误、意外回退——该 profile **立刻停**，不再重复计量。

H200 / 数据中心 CUDA、RTX PRO 5000、RTX 4090、RTX 5090、GB10、ROCm（`gfx942` / `gfx950`）只算 **菜谱覆盖**，不是另一张结果矩阵。

## 3. vLLM-Omni 的整系统优化

Base H3 车道保住发布的 BF16 权重、50 个 sigma、dense attention。优化顺着执行路径走，不是功能目录。

### 3.1 长序列 attention 与通信

典型工作： **58,758** 个有效 token 占 **58,816** token 对齐缓冲。三处削开销：

- [`TRTLLM_ATTN`](https://github.com/vllm-project/vllm-omni/pull/5283) 吃到有效序列长；[packed-sequence refinement](https://github.com/vllm-project/vllm-omni/pull/5779) 去掉结构性后缀 padding。
- [Rank-local boundaries](https://github.com/vllm-project/vllm-omni/pull/6173) 只造本 rank 的 embedding/RoPE 行，gather 的是紧凑的 **128 通道** 投影，不是 **5,376 通道** hidden state。
- [Fast Ulysses](https://github.com/vllm-project/vllm-omni/pull/6340) 用 NCCL SymmetricMemory，分片按 attention 要的布局到齐，all-to-all 周围不再另排一次。

### 3.2 融合的 DiT 算子

49 次前向里，矩阵乘四周全是小操作。Q/K RMSNorm 跟 RoPE 焊在一起（[#5990](https://github.com/vllm-project/vllm-omni/pull/5990)）；FP32 modulation、normalization、residual（[#6281](https://github.com/vllm-project/vllm-omni/pull/6281)、[#6878](https://github.com/vllm-project/vllm-omni/pull/6878)）；SiLU + multiply 换成 fused SwiGLU（[#6283](https://github.com/vllm-project/vllm-omni/pull/6283)）。

### 3.3 并行且融合的 VAE 解码

去噪之后，视频和音频各自解码。VAE patch parallelism 把 tiled video decoder 摊到八张卡。[Exact VAE operator path](https://github.com/vllm-project/vllm-omni/pull/6607)：decoder block 物化、融合 Q/K norm + RoPE、fused SwiGLU、scaled residual；不支持的布局退回 eager。

### 3.4 GPU 出图、搬运、MP4

几百帧离开 GPU 之前，请求不算完。每一步只转一次：

1. [GPU output preparation](https://github.com/vllm-project/vllm-omni/pull/6824)：解码后的 FP32 BCTHW → 连续 uint8 BTHWC，过线前视频载荷小 **75%**。
2. Pinned D2H 和 worker→engine IPC 运这份紧凑载荷。
3. [Direct-planar encoding](https://github.com/vllm-project/vllm-omni/pull/6288)、[persistent parallel converter](https://github.com/vllm-project/vllm-omni/pull/6499)、[transported strided RGB planes](https://github.com/vllm-project/vllm-omni/pull/6776) 直接喂 H.264，不再造一整块 interleaved RGB。

`FP32 BCTHW -> uint8 BTHWC -> pinned D2H/IPC -> planar frames -> H.264/AAC MP4`

### 3.5 测到的 base H3

同一 prompt / seed、50 个 sigma、同一条完整-MP4 边界。Diffusers：复制权重 + 原生 context parallelism。vLLM-Omni：encoder TP8，DiT USP8/Ring1 + Fast Ulysses，VAE PP8 tile，`TRTLLM_ATTN`。

| Runtime | Model execution (s) | Prompt (s) | DiT total / per-forward (s) | Video / audio VAE (s) | MP4 (s) | Client E2E (s) | Peak HBM/rank (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Diffusers | - | - | - | - | - | **82.239** | 151.699 |
| vLLM-Omni | **54.246** | 0.057 | 51.800 / 1.057 | 0.952 / 0.055 | 1.528 | **56.917** | 128.232 |

页上嵌了 model-card 样片和 vLLM-Omni baseline 的 MP4，这里不镜像。lossless 只保证加速不靠量化 / 稀疏 attention / 缓存复用 / 少步；不同 kernel 和浮点归约顺序仍可能拧扩散轨迹。这些改进削的是去噪 **周围** 的税。FastH3 再把循环本身从 49 次砍到 4 次。

## 4. 把通用 H3 serving 架构撑开

DLO 和分离 encoding 改的是容量与放置；可选量化权重和近似 attention 拿数值保真换内存或延迟。这些路径 **没有** 产出第 6 节的 FastH3 数字。

### 4.1 Distributed Layerwise Offload

[DLO](omni-layerwise-offload.md) 在 HBM 里只留有界窗口的 DiT 层，其余从主机流进来。AllGather 模式从主机分片集体重建当前层；rank-local 模式流的是各 rank 正常 loader 产出的张量。选哪边看互联、主机带宽、内存、常驻层数、请求并发。

**Figure 2**（DLO 专文那张流水线；未再拷）：当前层在算，下一层在备。

#### 8× B300 BF16 的 DLO Pareto

官方 BF16 MiniMax-H3 FL2VA checkpoint（5.175 s，1344×768，SP8/Ulysses8/Ring1/DP1/TP1，AllGather，CUDNN attention）。第一次请求排除（懒 CUDA/cuDNN/JIT），后两次取平均。生成的视频和音频形状对得上。

**Figure 3**（未收录）：延迟–内存 Pareto。*r* 是常驻 DiT block 数。非支配点：无 offload，然后 50、35、30、0 个 leading block 常驻；40、20、10 被支配。**r = 35** 时报告 HBM 降 **37.5%**，延迟代价 **5.1%**；**r = 0** 是最小内存端点。

### 4.2 分离 encoding

H3 的 Qwen3-VL encoder 权重 BF16 大约 **51.5 GB**。[Disaggregated encoder](https://github.com/vllm-project/vllm-omni/pull/5885) 把这次一次性 encoder 挪进独立 vLLM stage：自己的放置、TP、replica、队列、kernel、prefix cache。编排器把第 50 层 hidden states 和 token-role 标签跟原始媒体并在一起，再进 DiT/VAE。

**Figure 4**（未收录）：encoder 与 diffusion 各自扩容。合并后的单机菜谱经编排器回传 conditioning，diffusion 仍 inline；**不**配置 OmniConnector。跨节点 SHM/RDMA 还在 [RFC #5707](https://github.com/vllm-project/vllm-omni/issues/5707)。

### 4.3 可选量化与 attention 加速

第 3 节故意用 dense BF16 attention 和发布精度。下面每条都是另一张质量–性能剖面，不是 lossless 运行时增益。

#### 权重与激活量化

- **Online FP8。** 合并的 [global FP8 path](https://github.com/vllm-project/vllm-omni/pull/5910)：从 BF16 checkpoint 出发，加载时量化合格的 DiT 和 Qwen3-VL text-decoder 线性层。Embedding、norm、RoPE、vision tower、两个 VAE、对精度敏感的投影，仍按声明精度。
- **SVDQuant NVFP4 W4A4。** 合并的 [offline loader](https://github.com/vllm-project/vllm-omni/pull/6162)：NVFP4 W4A4 底座 GEMM + BF16 低秩修正。现有证据只到 checkpoint / 正确性兼容；原生 fused residual-GEMM 性能路径仍是未来工作。

**Figure 5**（未收录）：加载时造 FP8 权重与 scale，合格激活在线量化；offline SVDQuant 走 NVFP4 底座分支加 BF16 低秩修正。

量化 profile 必须同时报 peak HBM、启动主机 RAM、checkpoint 体积、延迟、同 seed 的视频/音频质量。容量赢不等于延迟赢；loader 正确不等于 fused kernel 赢。

#### B300 上 Online FP8 的容量与延迟

这条 dense、常驻结果把 Online FP8 从发布 BF16 里单独拎出来。8× B300，Ulysses8/Ring1 + Fast Ulysses，encoder TP8，VAE PP8 tile，CUDNN attention，10 秒 1344×768 / 24 FPS，请求 50 个 sigma（49 次 DiT 前向）。排除一次 warmup；三次数的均值。「Stage generation」是 diffusion stage 自己的计时；E2E 是离线客户端墙钟，收到返回的视频/音频张量为止，**不含 MP4 mux**。

| Weights | Stage generation (mean, n=3) | E2E (mean, n=3) | Peak HBM / rank | 结果 |
|---|---:|---:|---:|---|
| BF16 | 52.572 s | 53.118 s | 87.16 GiB | Lossless 基线 |
| Online FP8 | **49.769 s** | **50.331 s** | **53.27 GiB** | stage 时间低 5.3%；peak HBM 低 38.9% |

每次都交回 **243** 帧 1344×768 RGB 和 32 kHz 立体声音频。三次用不同 seed：证明形状和能生成，**不是**跟 BF16 逐像素等价。

#### `TRTLLM_ATTN` 里的量化与稀疏

- **SAGE**：QK 和 PV 两条路都量化到 FP8。
- **Skip-Softmax**：用 QK 结果动态跳过不重要的 Softmax 和 P×V（[BLASST](https://arxiv.org/abs/2512.12087)）。

**Figure 6**（未收录）：SAGE 围着 Skip-Softmax 主循环。

| Attention policy | SAGE configuration | Skip-Softmax configuration | Model execution | Speedup | LPIPS vs. baseline |
|---|---|---|---:|---:|---:|
| TRTLLM Baseline | Off | Off | 54.246 s | 1.000x | 0 |
| SAGE FP8 | `dtype_qk=fp8_e4m3`，`q_block_size=1`，`k_block_size=16` | Off | 44.787 s | **1.211x** | 0.3697 |
| Skip-Softmax | Off | threshold 0.05；disabled until 0.97 | 50.029 s | **1.084x** | 0.0917 |
| SAGE + Skip-Softmax | 同上 SAGE | 同上 Skip-Softmax | 43.867 s | **1.237x** | 0.3750 |

测到的 Skip-Softmax 对画质偏 **保守**。阈值抬高、或让更多去噪步启用，可以拿质量换速度。旋钮在 [TRTLLM attention guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/attention_backends/trtllm.md)。

#### Cache-DiT

[Cache-DiT](https://github.com/vllm-project/vllm-omni/pull/5853) 是请求级 cache 策略，不是 attention backend。H3 上 `quality=high` 开动态逐步复用，`quality=lossless` 回到参考路径。命中行为跟部署有关，**不**进上面的 attention A/B。

### 4.4 兼容边界

| 组合 | 本文口径 |
|---|---|
| Base H3 + DLO | 维护中的 H3 菜谱支持；拓扑仍要本地合格 |
| Base H3 + DLO + online FP8 | 支持，含 AllGather（[#6279](https://github.com/vllm-project/vllm-omni/pull/6279)）；性能和质量仍要本地合格 |
| Base H3 + 分离 encoder | 已合并的单机路径 |
| FastH3 + DLO | **不支持**：FastH3 在 `load_weights()` 里融合，offload 装的是另一条主机权重路径 |
| FastH3 + 分离 encoder | **尚未合格**；报告的 FastH3 结果没用它 |

**逐步执行旁注。** H3 能在去噪步之间接纳 / 中止请求（[#5810](https://github.com/vllm-project/vllm-omni/pull/5810)），但已有的 co-batching 测试 **没有** 把延迟变好。建议仍是 request mode；取消 / 回收和吃不饱的小负载还在 [issue #5700](https://github.com/vllm-project/vllm-omni/issues/5700)。

## 5. 从系统优化走到 FastH3

[FastH3](https://haoailab.com/blogs/fasth3-preview/) 是 FastVideo 给 MiniMax H3 蒸出来的四步 **DMD2 student**。Encoder、video VAE、audio VAE、tokenizer、scheduler 都复用；去噪循环变成五个 sigma 位置上的 **四次** transformer 前向。

- **FastVideo** 做蒸馏学生和 adapter 制品。
- **vLLM-Omni** 校验制品，checkpoint 流入时融合、切分融合后的权重，再走已经削过的 attention / VAE / 搬运 / MP4。

FastH3 **不是** 请求可切换的 LoRA。制品里还有全秩 delta 和替换权重，普通 LoRA 层装不下。所以要在切分 **之前** 融合。

**Figure 7**（未收录）：Turbo 底座不动，请求时挂 A/B sidecar；FastH3 把低秩和全秩改动焊进专用学生再切分。Turbo [#6476](https://github.com/vllm-project/vllm-omni/pull/6476)，DLO 支持 [#6550](https://github.com/vllm-project/vllm-omni/pull/6550)，FastH3 [#6714](https://github.com/vllm-project/vllm-omni/pull/6714)。

| Profile | 激活方式 | Task 范围 | 什么时候选 |
|---|---|---|---|
| Base H3 | 发布 checkpoint | T2VA、FL2VA、Ref2VA | 要全任务覆盖，还要跟通用扩容车道兼容 |
| Turbo | 请求可切换 adapter | T2VA 和 FL2VA | 同一服务要请求时切换，或要 FL2VA |
| FastH3 | 加载时融合的专用学生 | Dense/Data-Free T2VA | 专用 T2VA 端点上已验证的最低延迟 |

FastH3 v1 **拒绝** offload 和 VSA 变体，只接 T2VA，必须用它的四步日程和 checkpoint flow shift，也不能再挂另一个请求时 LoRA。这是 serving 合同，不是调参建议。

## 6. B300 上的 real-time FastH3

报告的是 vLLM-Omni `86b85c07` 上的 **绝对** FastH3 结果。不要拿第 3 节那条不同源码 / prompt / seed 的 base H3 去除。

### 6.1 钉死制品

测到的 Dense/Data-Free 制品钉在 Hugging Face revision `bcf40ca6f457ed66f8badf13514943e390205fca`：

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

Adapter **1,485,626,152** 字节。revision 和 checksum 都要钉。仓库名还带着 `Preview-v1`，对应的 Omni 集成已经合入。

### 6.2 Serve 与请求

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

一份 FastH3 replica；encoder TP8；DiT DP1 × TP1 × USP8，Ring1 + Fast Ulysses；VAE PP8 tile；`TRTLLM_ATTN`；标准紧凑出图 / MP4。

### 6.3 十秒关键路径

Profiler 计时来自另一次插桩；延迟主张跟 **clean E2E**。

> **原始基准包仍卡在发布闸门。** 稳定包还没公开。发布前要把这条 [evidence-handoff](https://github.com/vllm-project/vllm-project.github.io/pull/315#issuecomment-5459581336) 换成真正的包 URL：干净 / profiler 样本、日志、环境清单、媒体元数据与哈希、关键路径行和时长扫的拓扑证据。

| Encoder | DiT total / 4 / per-forward | Video + audio VAE | Derived transport | CPU MP4 | Profiled E2E | Clean E2E | Peak HBM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.052 s | 5.532 s / 4 / 1.383 s | 1.247 s combined | 0.881 s | 0.868 s | 8.629 s | **8.678 / 8.710 s** | 94.1 GiB/GPU reserved |

### 6.4 五、十、十五秒扫

冻结 prompt、seed、分辨率、制品、日程、拓扑、attention、VAE、出图路径、CPU affinity。H3 把请求时长对齐到 **124 / 243 / 362** 帧。

| Requested / aligned | Video / audio duration | DiT total / per-forward | Combined VAE | Transport + MP4 | Clean E2E | Client RTF | × real time |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 s / 124 | 5.167 / 5.175 s | 2.806 s / 0.702 s | 0.637 s | 0.929 s | 4.602 / 4.396 s | **0.889 / 0.849** | **1.125 / 1.177** |
| 10 s / 243 | 10.125 / 10.125 s | 5.532 s / 1.383 s | 1.247 s | 1.749 s | 8.678 / 8.710 s | 0.857 / 0.860 | 1.167 / 1.163 |
| 15 s / 362 | 15.083 / 15.083 s | 9.517 s / 2.379 s | 1.861 s | 2.484 s | 14.177 / 14.059 s | 0.940 / 0.932 | 1.064 / 1.073 |

六次测量都满足 `RTF_client <= 1.0`：测过的每个时长，完整 MP4 都比播放快。

### 6.5 样片与质量边界

页上给的 FastH3 样片是 **1280×736** 的展示例，**不是** 6.4 节 1344×768 的计时制品。

| Request | Frames | MP4 duration | Resolution / FPS |
|---:|---:|---:|---|
| 5 s | 124 | 5.184 s | 1280×736 / 24 FPS |
| 10 s | 243 | 10.144 s | 1280×736 / 24 FPS |
| 15 s | 362 | 15.104 s | 1280×736 / 24 FPS |

发布级计时与媒体证据仍等 6.3 的原始包闸门。

| 质量闸门 | 状态 |
|---|---|
| 同 seed 重复 FastH3 输出 | 测到的运行里逐字节相同 |
| 媒体结构 | 帧数 / FPS、H.264、立体声 AAC、视频音频信号非零 |
| 对齐的 base vs FastH3 多种子质量 | **未完成；不作 parity 主张** |

去噪一少，新尾巴露出来：10 秒 profile 上，VAE + 推导的搬运 + CPU MP4 在插桩路径里大约 **三秒**。[RFC #6872](https://github.com/vllm-project/vllm-omni/issues/6872) 提议让 VAE 块、D2H/IPC、编码重叠，而不是各管各的。这篇 B300 profile 的乐观天花板：搬运跟编码重叠大约 **0.87 秒**（约 10% E2E）；再叠上增量 VAE 解码大约 **1.75 秒**（约 20% E2E）；go/no-go 目标至少 **5%** 和 **10%** E2E。草稿 [PR #6885](https://github.com/vllm-project/vllm-omni/pull/6885) 在 **四张 L20X** 可行性跑上报告 VAE→完整 MP4 少 **0.8847 秒**（**26.57%**），媒体精确对得上——**不是** B300 生产结果。

## 7. 生产指引与限制

| 需求 | 建议 profile |
|---|---|
| 要完整 T2VA / FL2VA / Ref2VA | 带整系统栈的 Base H3 |
| 请求时切 adapter，或四步 Turbo 的 FL2VA | 单独的 Turbo 服务 |
| 已验证的最低 T2VA 完整响应延迟 | 第 6 节那条专用 FastH3 服务 |
| 主机内存驱动的塞得下，或 encoder 独立扩容 | Base H3 的 DLO 或分离-encoder 车道；本地合格 |

报告过的 FastH3 profile **不要** 跟 DLO、VSA、量化、cache 策略、别的 Ulysses 传输、encoder 分离混用，除非重新做正确性 / 质量 / 内存 / 延迟合格。活的兼容表：[issue #5700](https://github.com/vllm-project/vllm-omni/issues/5700)——它可能落后于已合并实现。选生产组合前核对链接 PR 和维护中的菜谱。

许可证：[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)。商业和托管服务运营方应把属地、署名、营收、可接受使用、防护条款交给律师看。

后训练：vLLM-Omni 也能在 [VeRL-Omni](https://github.com/verl-project/verl-omni) 里给 H3 rollout 当 serving；训练是生态覆盖，**不是** 这篇 serving 基准的一部分。

## 8. 收束与还要做的

整系统优化让完整 H3 流水线变便宜。FastVideo 的四步学生再把专用 T2VA 送进「比播放还快」的完整响应，至少在测过的 B300 上如此。

页上点名的后续：接入并合格 FastH3 的 VSA 变体和原生 fused NVFP4 kernel；在目标 Blackwell 和多种子上合格 [Sol-Attn](https://github.com/vllm-project/vllm-omni/pull/5851) 即时稀疏 attention；做对齐的 base/FastH3 多种子质量评估；实现 [分块 VAE→搬运→MP4](https://github.com/vllm-project/vllm-omni/issues/6872) 并合格 GPU encoder；把 MiniMax H3 后训练接到 VeRL-Omni、[UniRL](https://github.com/Tencent-Hunyuan/UniRL)、[RLinf](https://github.com/RLinf/RLinf)——可扩展 rollout、显式资源放置、端到端训练校验；合格 FastH3 与 encoder 分离等扩容特性的组合，而不是靠推断兼容。

## 致谢

工作站在 vLLM、vLLM-Omni、VeRL-Omni、MiniMax H3、[FastVideo](https://github.com/hao-ai-lab/FastVideo)、FastH3、Diffusers、NVIDIA 上。特别感谢 FastVideo 团队 [开源 FastH3](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA) 并和 Omni 社区把 serving 集成合进去。页上点名的 GitHub：Isotr0py（base H3）；lishunyang12、evanchueng、Gaohan123、david6666666（DLO / 底座 / online-FP8）；gcanlin、yuanwu2017（encoder 分离）；bobboli、fan2956、mo-ke-ke、mglyn、MosCloud、ultism（attention、融合 kernel、量化、VAE、搬运、媒体）；princepride（FastH3 集成与 B300 验证）；NancyFyong、mengchengTang（VeRL-Omni）。Hongsheng Liu 与 Roger Wang 提供一般支持与成文。

## 附录 A. 可复现性

### A.1 计时层级

vLLM-Omni 的测量是嵌套的；**不要** 把父子相加：

| 边界 | 范围 |
|---|---|
| Client | 提交请求到收到完整 MP4 |
| Request | 编排器跨 stage 的一生 |
| Stage | 一组独立调度的 engine/device |
| Engine | 排队、模型执行、等输出就绪、格式化 |
| Profiler | engine 执行内部的 Prompt / DiT / VAE 方法边界 |
| Server | 最后 stage 之后的 H.264/AAC 编码与 mux |

逐步去噪时间按 **实际 DiT 前向次数** 除，不是按请求的 sigma 点数。Profiler 来自另一次诊断请求，**不能** 代替未插桩的客户端延迟。

### A.2 Base H3 在 vLLM-Omni 上复现

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

规范请求用第 2 节的 prompt 和 seed，请求 50 个 sigma，flow shift 12，audio flow shift 3，目标 10 秒。

## 参考文献

- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [FastH3 技术概述](https://haoailab.com/blogs/fasth3-preview/)
- [FastH3 四步 adapter](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA)
- [MiniMax H3 模型](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Diffusers MiniMax H3 pipeline](https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/minimax_h3)
- [MiniMax H3 serving 菜谱](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md)
- [Distributed Layerwise Offload](https://vllm.ai/blog/2026-08-17-distributed-layerwise-offload)
- [特性兼容跟踪](https://github.com/vllm-project/vllm-omni/issues/5700)
- [分块输出流水线 RFC](https://github.com/vllm-project/vllm-omni/issues/6872)
- [VeRL-Omni](https://github.com/verl-project/verl-omni) · [UniRL](https://github.com/Tencent-Hunyuan/UniRL) · [RLinf](https://github.com/RLinf/RLinf)
