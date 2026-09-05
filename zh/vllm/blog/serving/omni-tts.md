---
source: https://vllm.ai/blog/2026-06-23-vllm-omni-tts
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# TTS：Talker 要 TTFP，Code2Wav 要吞吐

英文对照：[en/vllm/blog/serving/omni-tts.md](../../../../en/vllm/blog/serving/omni-tts.md)  
原文：https://vllm.ai/blog/2026-06-23-vllm-omni-tts  
2026-06-23。署名 **vLLM-Omni TTS Team**。Omni 从 omni-modality 扩到语音：Qwen3-TTS、VoxCPM2、Higgs Audio V3、Fish Speech S2 Pro。怎么伺候：分阶段 serving、batch、CUDA Graphs、模型专用 kernel。同一条 Omni 线：[vllm-omni.md](vllm-omni.md)、[qwen3-omni.md](qwen3-omni.md)。学习笔记；页上 cookbook 数字不是你的 SLA。

TTS 不是单只 LLM。Talker 是延迟绑的单 token Decode；Code2Wav 是吞吐绑的并行 decode。同一只调度两边都亏。块太小，跨块不连续；太大，**TTFP**（Time To First Audio Packet）爆。没有一只菜谱：Qwen3-TTS 阶段分离 + connector 切块 + Talker 预处理批量化；VoxCPM2 整段 `torch.compile` + CFM/LocDiT 尾部批；Higgs 多 codebook 状态留 GPU；Fish `q_len=1` Decode attention。

本地图（原文版权仍归原站；学习对照用）：

![tts serving pipeline](../../../../assets/vllm/blog/serving/omni-tts/01-tts-serving-pipeline.png)

![qwen3 tts connector chunking](../../../../assets/vllm/blog/serving/omni-tts/02-qwen3-tts-connector-chunking.png)

![qwen3 tts stage0 dispatch consolidation](../../../../assets/vllm/blog/serving/omni-tts/03-qwen3-tts-stage0-dispatch-consolidation.png)

![voxcpm2 single stage pipeline](../../../../assets/vllm/blog/serving/omni-tts/04-voxcpm2-single-stage-pipeline.png)

![voxcpm2 compile dispatch combined](../../../../assets/vllm/blog/serving/omni-tts/05-voxcpm2-compile-dispatch-combined.png)

![fish speech stage0 runtime shape](../../../../assets/vllm/blog/serving/omni-tts/06-fish-speech-stage0-runtime-shape.png)

## TTS 和文本 LLM 不是同一条调度

两边都用自回归模型；serving 瓶颈不一样。

**TTS 是管线，通常多阶段。** 典型：Talker 自回归预测 codec token；Code2Wav 从这些 token 还原波形。Talker 是延迟绑的单 token Decode；Code2Wav 是吞吐绑的并行 decoder。一只调度伺候两边：Talker 延迟堵住 Code2Wav 的输入，Code2Wav 的并行吃不饱。延迟和吞吐一起亏。

**Streaming 输出有硬延迟预算。** 用户指望几百毫秒内听到第一包。Connector 必须支持切块 streaming；块大小直接打 **TTFP**。太小：Code2Wav 跨块缺上下文。太大：第一包来不及。

**吞吐仍然要紧。** 在线 serving 的成本看一张 GPU 能撑多少并发，以及墙钟每秒能吐出多少秒音频。Talker 和 Code2Wav 瓶颈不同；connector 还要付传输税。提吞吐就是把两阶段配平，再各自拆掉内部瓶颈。

**Figure（pipeline）。** Talker → connector → Code2Wav serving 路径。

后文：优化总览，再拿 Qwen3-TTS 走完整路径，然后 VoxCPM2 / Higgs Audio V3 / Fish Speech S2 Pro 各讲架构专用策略。

## 优化总览

vLLM-Omni 不定一只死菜谱。选哪根杠杆，看管线结构、Decode 状态、batch 形状、数值约束。

| Technique | Applies to | Why it matters |
|---|---|---|
| Stage separation and connector chunking | Qwen3-TTS, Higgs Audio V3 | Talker 延迟和 Code2Wav 吞吐可以各自拧。 |
| Batched decode preprocessing | Qwen3-TTS | 砍掉 Talker Decode 热路径上反复的 per-request Python。 |
| Whole-forward `torch.compile` | VoxCPM2 | 让 Dynamo 看见更多 MiniCPM4 前向循环，少 Python↔compiled 边界。 |
| CFM/LocDiT decode-tail batching | VoxCPM2 | 许多 per-request 小扩散调用收成更大的 GPU batch。 |
| GPU-resident decode state | Higgs Audio V3 | 多 codebook 状态更新离开 Python 循环，少同步。 |
| Model-specific q_len=1 attention | Fish Speech S2 Pro | 专伺候纯 Decode attention，不付通用 paged/varlen 的税。 |

不是每项优化都能套到每只 TTS 架构。活是给模型形状挑对杠杆。

## Qwen3-TTS：一整条优化路

Qwen3-TTS 是 Qwen 团队的语音生成家族：离散多 codebook 语言模型架构，**12 Hz** tokenizer，兼顾声学压缩和高保真还原。三个变体共用同一套两阶段（Talker AR codec token，Code2Wav 并行 decode）：

- **Base** — 声音克隆
- **CustomVoice** — 预定义说话人，指令控情绪和风格
- **VoiceDesign** — 用自然语言描述音色、情绪、韵律，造新声

Qwen3-TTS 的 Code2Wav 是轻量 **non-DiT** decoder；**不**需要迭代去噪循环。四只模型里，这是最标准的 Talker → connector → Code2Wav 形状，所以拿它当 walkthrough。

### 1. Streaming：connector 切块和 Code2Wav decode 窗口脱钩

早期 Qwen3-TTS 把 connector streaming 块和 Code2Wav decode 块绑在同一参数上，主要是 `codec_chunk_frames`。Connector 块小，Code2Wav 看见的 decode 块也小（连续性吃亏）。为了质量把块加大，第一包延迟跟着涨。

旋钮拆开：

- `codec_chunk_frames`：connector streaming 块大小（Talker 往 Code2Wav 的传输节奏）
- `decode_chunk_frames` 和 `decode_left_context_frames`：Code2Wav 内部 decode 窗口和左上下文，跟 connector 切块无关
- `initial_codec_chunk_frames`：更小的第一块 codec，让 Code2Wav 早点开工；后面块回到常规尺寸

Connector 可以用小块砍第一包延迟，同时 Code2Wav 保住 **300** 帧 decode 窗口加 **25** 帧左上下文。各自拧（[PR #3485](https://github.com/vllm-project/vllm-omni/pull/3485)）。

**Figure。** Connector 切块脱钩。

### 2. Throughput：Stage 0 Decode 预处理

下一个瓶颈：Talker Decode。每一步都要请求级预处理：speaker embedding、`trailing_text` 维护、输入 embedding 构造。c=1 时开销小。c=64 时每步 Decode 要扫 64 条请求；Python 循环和张量切片开始显眼。

在 **H20 × 2** 上 profile 过 Talker Decode，声音克隆，c=64。更宽的热路径动手之前的 warm run：模型外的 Python 和 runner 侧工作——`preprocess_decode_batch`、`make_omni_output`、`process_additional_info`、`build_mm_cpu`、bookkeeping sync——落在 **每 Decode 步毫秒级**。一句大约要 **200** 步 Decode。c=64 时这笔税沿整段序列重复。

c=64 时 `nvidia-smi`：基线平均 GPU 利用率大约 **14%**（Stage 0）和 **6%**（Stage 1）。GPU 在等 Python 调度、小张量分配、kernel launch——不是缺 FLOPs。

第一个具体靶：speaker embedding。声音克隆从参考音频抽 embedding，Decode 时还要做 mel/STFT。原路径：每请求在 CPU 上算 mel，再拷到 GPU。高并发 → 许多小 H2D 传输和 launch。修法：mel basis 和 window buffer 缓存在 GPU 上；mel/STFT 在 GPU 上批算。

下一个：`trailing_text`。Talker 给已生成 token 留滑动窗口 embedding。原路径：切片再拼接，频繁分配新张量。优化后：跟踪 offset；只有 offset 越过阈值或到 buffer 末尾才 compact（`_TRAILING_TEXT_COMPACT_MIN_FRAMES = 64`）。中间步按 offset 索引，不分配。

批量化的 `preprocess_decode_batch` 拿掉一块主要的 per-request Decode 开销（[PR #3662](https://github.com/vllm-project/vllm-omni/pull/3662)）。最终叠起来的数字含 Stage 0 batching、connector 改动、async D2H、runner 热路径清理、CUDA Graph 调参（[PR #3689](https://github.com/vllm-project/vllm-omni/pull/3689)、#3485、#3662）。最终叠跑，Qwen3-TTS on H20 × 2：音频吞吐 **26.55 → 42.88 audio-s/s**（+**61.5%**）；P99 E2EL **17.7s → 9.0s**。

**Figure。** Stage 0 dispatch 收拢：更少 CPU launch、更少小 GPU kernel 切片。**不是**在声称 GPU 利用率变高。

### 3. Hot-path 清理

Batch 之后剩下的 profile：许多小 Python 开销，在高频 c=64 Decode 循环里会叠起来。

`req_id_to_index` 用 `req_ids.index()`——每 Decode 步一次 O(N²) 列表扫描。换成字典（O(1)）。非 streaming 请求在 orchestrator 里早点跳过 per-output streaming 路径。Codec-disallowed mask 预计算进 buffer，`compute_logits` 直接 `masked_fill`，不必每步重建 mask。

Qwen3-TTS 多处用 CUDA Graph。Talker code predictor 按 deploy profile 有自己的 graph 路径。这里盯的是 Code2Wav decoder CUDA Graph。Decoder 输入形状 `(batch, num_quantizers, codec_frames)`。切块 decode 里，`codec_frames` 是一小集：streaming 块加左上下文；非 streaming 的 `decode_chunk_frames + decode_left_context_frames`（**300 + 25 = 325**）；尾块。Warmup 时可以枚举。`CUDAGraphDecoderWrapper` 按 `(batch_size, frames)` 捕获 graph，推理时 `bisect_left` 挑最近的 padded bucket。对不上 → eager。

用 `qwen3_tts.yaml` 反复 c=16：Code2Wav CUDA Graph hit rate 起步 **88%**，连续五轮后落在大约 **81%**。主要单样本形状打中捕获的 bucket：`(1, 98) -> 169`、`(1, 73) -> 73`、`(1, 123) -> 169`、`(1, 325) -> 325`。Fallback 大多是 batch-size > 1：`(2, 98, 169)`、`(8, 73, 73)`。整段跑下来 `stream_capture_fallbacks=0`——没有因为 stream capture 失败而 fallback。

### 4. 数值精度：code predictor 对齐 fp32

Talker code predictor 对精度敏感。很短的序列，通常 **2–8** token，反复 Prefill。vLLM 的 fused kernel 走 bfloat16，跟参考实现会有一点差。这条短序列、高频路径上，差会累积，几十步之后能伤音质。

修法：拆开 code predictor 层；选定的 op 留 fp32：RMSNorm 方差、RoPE cos/sin、attention、QKV projection 用 PyTorch 原生实现，跟参考做 bit-level 对齐。

### 5. Validation

叠起来的优化，Qwen3-TTS on H20 × 2，c=64 声音克隆：音频吞吐 **+61.5%**，P99 E2EL 几乎腰斩。完整数字在 Performance Data。

Warm 并发扫描，H20 × 2，声音克隆，streaming：

| c | Mean TTFP | Mean E2E | P50 TTFP | P50 E2E |
|---:|---:|---:|---:|---:|
| 1 | 70.61ms | 564ms | 70.61ms | 564ms |
| 8 | 268.75ms | 1.55s | 287.15ms | 1.70s |
| 16 | 451.32ms | 2.62s | 516.15ms | 2.75s |
| 32 | 637.43ms | 5.05s | 634.22ms | 5.10s |
| 64 | 1127.93ms | 8.73s | 1051.05ms | 8.78s |

c=1 → c=64，E2E **0.56s → 8.73s**，不是线性 64×。Warm 高并发会摊掉固定成本，但 c=64 时 Talker 和调度仍然排队。所以 hot-path 清理和 CUDA Graph 还是要紧。

## VoxCPM2：单阶段 hybrid TTS

VoxCPM2（OpenBMB）是无 tokenizer 的 TTS：在 AudioVAE V2 latent 空间里做 diffusion-autoregressive hybrid。Talker 是四段级联：

```text
MiniCPM4 (28 layers, PagedAttention) → FSQ → MiniCPM4 ResidualLM (8 layers) → LocDiT (CFM solver) → AudioVAE
```

LocDiT 做 CFM（Conditional Flow Matching）去噪；AudioVAE 还原 **48 kHz** 波形。在 vLLM-Omni 里，VoxCPM2 **不**拆成多个运行时阶段。单阶段 AR TTS：MiniCPM4、FSQ、ResidualLM、LocDiT、AudioVAE 在同一模型实例里，直接出音频。省掉阶段之间的 latent 传输；decode 尾部的 CFM/LocDiT 和 VAE 更容易跨请求 batch。

**Figure。** 单阶段 hybrid 管线。

跟两阶段 Qwen3-TTS 不同，VoxCPM2 问的是：28 层 MiniCPM4 怎么更快，以及高并发时怎么不让 CFM/LocDiT 把 GPU 吃不饱。

### 试 torch.compile

28 层 MiniCPM4 是 Talker 最重的一块。第一个靶：`torch.compile`。最好的路不是第一次尝试。

第一次：分别 compile 每层的 `mlp` 和 `o_proj`——28 × 2 = **56** 个 compiled region，`fullgraph=True`（[PR #2690](https://github.com/vllm-project/vllm-omni/pull/2690)）。Dynamo 不能跨 compiled-region 边界优化。每个边界都是 Python → compiled → Python；56 个 region 意味着每 Decode 步许多次切换。

然后把整段 `Model.forward` 包进 `torch.compile`，`fullgraph=False`（[PR #2758](https://github.com/vllm-project/vllm-omni/pull/2758)）。Dynamo 看见完整的 28 层循环。PagedAttention 仍会 graph-break，但 Dynamo 只 memoize 少数 subgraph。每步 dispatch：许多小 region → 几块大的。RTF **~0.21 → ~0.13**——VoxCPM2 单次最大的优化。

三套配置 profile 过：eager、per-layer compile、whole/unified graph。Per-layer compile 砍了一些 kernel 数和 kernel 时间，但 **launch count 没掉**。Whole/unified graph：`cudaLaunchKernel` 计数 **~−71%**，kernel events **~−30%**，kernel time **~−27%**。单请求 E2E **~−2.6%**（per-layer）对 **~−6.5%**（whole graph）。

**Figure。** Compile dispatch 时间线和计数。Launch count 一直平，直到整段 forward compile。

`mode="reduce-overhead"`（自动 CUDA Graph capture）跟 PagedAttention 有状态的 KV cache 打架。Capture 时 `slot_mapping` 被钉死；replay 可能把 attention 结果写到错误的 KV 位置 → stop logits 错、提前截断。

`fullgraph=True` 受不了 PagedAttention 和自定义精度边界带来的 graph break。`fullgraph=False` 保住整段 forward 的视野，同时允许那些边界回 eager。

### CFM/LocDiT decode-tail batching

单请求延迟改善之后，高并发瓶颈挪到 CFM/LocDiT。每条请求在 CFM 去噪时跑 LocDiT attention/GEMM，但 per-request batch 极小，CFG 下通常 **B=2**——远填不满 GPU。高并发时各自独立的 LocDiT 让 GPU 闲着。

办法：跨请求 batch CFM/LocDiT decode 尾。从多条请求收 `lm_h`、residual 输出、prefix feature condition；`dit_proj`、CFM/LocDiT、`feat_encoder`、`stop_head` 作为一次 batch 跑；再 scatter 回去。再配上每 **三** 个 latent chunk 做一次 VAE decode、batched VAE decode、合并的音频 D2H 拷贝、LocDiT fused-QKV / fused gate-up MLP：H20 × 1 在 c=64 **4.19 → 10.83 req/s**（+**158.8%**），音频吞吐 **12.16 → 33.07 audio-s/s**（+**172.0%**）（[PR #3882](https://github.com/vllm-project/vllm-omni/pull/3882)）。

Euler 积分循环里的同步：对 0 维 GPU 张量 `.item()` 会逼 GPU-to-CPU sync。原来：每扩散步 **四** 次。**10** 个 timestep、大约 **60** 步 Decode，一条请求能触发大约 **2,400** 次同步。把 `.item()` 换成 GPU 侧 `.copy_()` 广播——CPU 离开那条循环。

VAE 的结构问题：第一版是累积再重 decode——每 **五** 步把此前所有 latent patch 拼起来，再把整段前缀 decode 一遍。工作量 **O(N²)**。滑动窗口 decode：**12** 帧 pad 上下文、每次 **四** 帧新内容 → **O(N)**。长文本 RTF 不再随文本长度涨；所有长度都落在 RTF **0.132–0.138**（[PR #2758](https://github.com/vllm-project/vllm-omni/pull/2758)）。

## Higgs Audio V3：动态 batch 和多 codebook 状态

Higgs Audio V3（Boson AI）：超过 **100** 种语言，zero-shot 声音克隆。Qwen3 骨架，**36** 层，hidden size **2560**，GQA，融合多 codebook embedding（一张大 `[N × V, D]` 矩阵加 offset lookup），MusicGen 风格 delay pattern `[0, 1, 2, ..., 7]`，带 BOC/EOC 特殊 token。

Talker → Code2Wav 形状接近 Qwen3-TTS；Talker 内部不同，因为多 codebook 预测和 delay pattern。

Qwen3-TTS 受限于 Python 热路径和 streaming 块边界；Higgs v3 受限于复杂的多 codebook Decode 状态，以及跟 CUDA Graph 的兼容。

### 把 Decode 状态搬上 GPU

主要吞吐收益：把 per-request Python dict 状态机搬成 GPU 常驻的 batched 张量（[PR #4204](https://github.com/vllm-project/vllm-omni/pull/4204)）。状态包括 `_decode_last_codes`、`_decode_has_codes`、delay count、EOC countdown、generation-done flags、相关 Decode metadata。好处：更少 Python per-request 循环、更少 D2H 同步，采样/状态更新走 batched GPU 热路径。报出的 **35.26 audio-s/s** 是在 **单张 H20**、**c=16**、**eager + local MLP CUDA Graph** profile 上测的，**不是** PIECEWISE 整段 Decode graph 路径。

难处：vLLM 调度器在 Decode 期间可能重排、收缩、结束或移除请求。行级状态 ≠ 请求级状态。音频 AR 状态比文本复杂：delay codebook、EOC 收尾、终端帧都有语义。差一步不是干净地崩，是音质问题。GPU 状态、CPU override 状态、调度器 token 需要单一真相源，否则 stop 语义会对不齐。

### 让 CUDA Graph 适应动态 batch 形状

Talker CUDA Graph capture：音频反馈用上一个音频 token 的 embedding 替换下一个 continuation token 的 embedding。实现用布尔 mask 选出当前在 Decode 的请求。得到的张量形状取决于运行时有多少请求在 Decode。

CUDA Graph capture 要固定的 stream 操作和固定的 I/O 形状。数据依赖的布尔 mask 选择违例。

绕法：CUDA Graph 路径走均匀的单 token Decode batch。每个 span 长度是 1，所以 `decode_mask` 全 True。选择变成 no-op，返回原张量。Graph 看见稳定的 full-batch 形状，而不是数据依赖的 compacted 形状。

### Local MLP CUDA Graph vs PIECEWISE

Local MLP CUDA Graph 仍是 Higgs v3 最重要的 graph 优化。它罩住 `post_attention_layernorm + mlp` 里主要的 GPU 成本。vLLM PIECEWISE CUDA Graph 看起来更完整（更大的 Decode 步）。实践里，Higgs v3 的多 codebook delay pattern 让 token layout 跨 Decode 步变化。Embedding lookup 和 pre-attention index op 是数据依赖的。PIECEWISE 要么 graph-break 回 eager，要么要额外的 metadata 同步。

E2E：PIECEWISE 要求关掉 local MLP graph，这笔买卖亏大于赚。Eager 加 local MLP graph 比 PIECEWISE graph 更快。

### 一条被否掉的 staging-overlap 设计

记下来仍有用：一步音频 staging overlap——把音频 staging 的 D2H 拷贝和下一步 Decode 重叠，砍 GPU 空闲。Dry run 过了；压测发现调度器在 Decode 期间可能重排、收缩或结束请求。指向某一行的 cursor 会丢掉跟请求的映射。在动态 batch 下结构不安全，不是边界条件 bug。以后的 overlap 设计应按 request-id 做键，并带 finish/remove 的 drain hook。

## Fish Speech S2 Pro：通用 attention 变成瓶颈的时候

Fish Speech S2 Pro（Fish Audio）：Dual-AR，训在超过 **1000 万小时**、超过 **80** 种语言上。在 vLLM-Omni 里：slow_ar + Fast AR + DAC decoder。slow_ar 沿时间预测 semantic codebook；Fast AR 在每步 Decode 预测 residual codebook；DAC 从 **10** 个 codebook 还原波形。

不像 Qwen3-TTS（Python 预处理），Fish 是 GPU 侧瓶颈。高并发时，**q_len=1** attention 占主导。通用 paged/varlen attention 带着给 Prefill、chunked Prefill、Decode、其他模型形状准备的形状检查和分支。对 Fish 的纯 Decode 形状，这份灵活是开销。

### 模型专用 attention kernel

Profiling：高并发时 Fish slow_ar 大部分时间花在 q_len=1 SlowAR attention，以及 DAC↔runtime 交接。Fish Decode 很窄：q_len=1，fp16/bf16，head_dim=**128**，block size **16**，Fish GQA layout。

给 SlowAR Decode attention 写的 Fish 专用 Triton kernel（[PR #3773](https://github.com/vllm-project/vllm-omni/pull/3773)）。**不**伺候 Prefill 或其他模型。形状对不上 → 原来的 attention 路径。

两条路。短序列到 **1024** token：标准 online softmax 一遍过。Grid `(batch_size, num_kv_heads)`；每个 program 处理一行 batch 和一只 KV head，罩住它的 Q heads。Block size 写死 **16**，跟 vLLM 的 KV cache block size 对齐，所以 block table lookup 是直接 `tl.load`，不必额外 gather。长序列：split-partial-combine——切成段，各自算 partial m/l/acc，再用 online softmax 递推合并。让带参考音频的长上下文请求仍走快路径。

Dispatch 的细处：kernel 要序列长度才能选短/长，但精确长度在 GPU 上。读回 CPU 会同步。Runner 用 computed tokens 加 scheduled tokens 在 CPU 侧算 `seq_lens_cpu_upper_bound`。上界永远 ≥ 真序列长度。短路径不会少读；长 split 路径不会少覆盖。CUDA Graph capture 期间，上界是 `max_model_len`，所有 graph 路径都罩住。

**Figure。** Stage 0 runtime 形状，q_len=1 快路径前后。补 kernel 设计；**不**替代基准数字。

快路径只给 Fish SlowAR attention 层。加载时走 `model.layers`，把每层 attention 的 `impl.forward` 换成 wrapper：约束对上就 dispatch 到 Fish 快路径。Prefill、非 Fish 模型、不支持的 Decode 形状走原来的 attention。

### Fast AR buffer 复用和 compile

Fish Speech Fast AR：四层轻量 transformer，在每步 slow_ar 之后预测 residual codebook。Per-call KV cache：每个 residual codebook 步只 Decode 一个新 token，把 K/V 写进预分配的 `_k_cache` 和 `_v_cache`。

每步 Fast AR Decode：投影 slow_ar hidden state，embed 当前 semantic token，一层层 attention + MLP，从 logits 采样。序列最多 **10** token，但 c=64 时反复分配和 Prefill 会显眼。

一次分配 `_embed_buf`、`_pos_ids`、`_k_cache`、`_v_cache` 再复用。`_embed_buf` 形状 `(batch_size, num_codebooks + 1, hidden_dim)` 罩住一次 Fast AR Decode 的所有时间步。`_k_cache` / `_v_cache` 按层、batch、KV head、序列位置、head dim 预分配，所以 `forward_one` 原地读写。

Fast AR 也 `torch.compile`。不像 VoxCPM2 的 MiniCPM4，Fast AR 只有四层——compile 开销小。`fullgraph=False`，因为 attention 用 `F.scaled_dot_product_attention` 而不是 paged attention；SDPA 内部可能 graph-break。Dynamo memoize 少数 subgraph。`dynamic=True` 让 compiled 结果能伺候 batch-size 变化。

### DAC 和 runtime 侧优化

Codec payload 传输：Python `list[int]` → 张量 payload——2D code 张量直接序列化，不再展开成 Python 整数，砍高并发下的分配和 GC。fp16 DAC 把内存和计算减半。按帧数封顶的 DAC batching 限制一次 DAC forward 的帧数，不让一条长请求堵住别人。Async chunk 处理把 connector 传输和 DAC 重叠：slow_ar 和 Fast AR 每 Decode 步产出一帧 10-codebook codec；connector 攒到 `codec_chunk_frames`；DAC 处理当前块时，connector 在攒下一块。

## Performance data

vLLM-Omni cookbook 基准。

- **RTF**：生成时间 / 音频时长。低于 1 表示快过实时。
- **TTFP**：Time To First Audio Packet。
- **Tput**：音频吞吐——墙钟每秒生成的音频秒数。
- **E2EL**：端到端延迟。

### Qwen3-TTS (c=64, p=512, H20 × 2, voice clone)

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Audio throughput | 26.55 audio-s/s | 42.88 audio-s/s | +61.5% |
| Median E2EL | 9654ms | 5699ms | −41.0% |
| P99 E2EL | 17686ms | 8956ms | −49.4% |
| P99 TTFP | 7558ms | 5563ms | −26.4% |

### VoxCPM2 (c=64, H20 × 1, before/after CFM batching)

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Request throughput | 4.19 req/s | 10.83 req/s | +158.8% |
| Audio throughput | 12.16 audio-s/s | 33.07 audio-s/s | +172.0% |

### Fish Speech S2 Pro (H20, single GPU, c=64, Triton KV cache + tensor payload)

| Metric | Value |
|---|---:|
| Audio throughput | 23.72 audio-s/s |
| Request throughput | 5.95 req/s |
| Mean TTFP | 899.67 ms |
| Mean E2EL | 10.47 s |

### Higgs Audio V3 (H20, single GPU, c=16, eager + local MLP graph)

| Metric | Value |
|---|---:|
| Request throughput | 5.18 req/s |
| Audio throughput | 35.26 audio-s/s |
| Wall time | 96.5s |
| Speedup vs. baseline | 2.70× |

## Acknowledgements

Minghui Jiang, Yueqian Lin, Canlin Guo, Shunyang Li, Taichang Zhou, Yuekai Zhang, Juan Pablo Zuluaga, Nick Cao, Ruirui Yang, Wenjing Chen, Haiyan Wu, Han Gao, Hongsheng Liu, and Roger Wang.

## References

- Qwen3-TTS hot-path micro-optimizations — [PR #3689](https://github.com/vllm-project/vllm-omni/pull/3689)
- VoxCPM2 per-layer compile + PagedAttention — [PR #2690](https://github.com/vllm-project/vllm-omni/pull/2690)
- VoxCPM2 whole-model compile + streaming VAE + CFM sync fix — [PR #2758](https://github.com/vllm-project/vllm-omni/pull/2758)
- VoxCPM2 CFM/LocDiT batching + decode-tail optimizations — [PR #3882](https://github.com/vllm-project/vllm-omni/pull/3882)
- Qwen3-TTS streaming connector decoupling — [PR #3485](https://github.com/vllm-project/vllm-omni/pull/3485)
- Qwen3-TTS high-concurrency Stage 0 batching — [PR #3662](https://github.com/vllm-project/vllm-omni/pull/3662)
- Fish Speech S2 Pro KV cache fast path + DAC optimizations — [PR #3773](https://github.com/vllm-project/vllm-omni/pull/3773)
- Higgs Audio V3 GPU-resident state machine + CUDA Graph — [PR #4204](https://github.com/vllm-project/vllm-omni/pull/4204)
- Qwen3-TTS — [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- VoxCPM2 — [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- Fish Speech S2 Pro — [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)

TTS 推理：vLLM Slack 的 `#sig-omni`（[vLLM Slack](https://slack.vllm.ai)），或 [vLLM-Omni GitHub](https://github.com/vllm-project/vllm-omni) 上开 issue。
