---
source: https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Qwen3-Omni：Thinker / Talker / Code2Wav 三截

英文对照：[en/vllm/blog/serving/qwen3-omni.md](../../../../en/vllm/blog/serving/qwen3-omni.md)  
原文：https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization  
2026-07-01。署名 **vLLM-Omni Team and Ant Group SCT Team**。不是一只 Decode 环，是 Thinker → Talker → Code2Wav。TTS 工程细节见 [omni-tts.md](omni-tts.md)。同一条 Omni 线：[vllm-omni.md](vllm-omni.md)。Seed-TTS `en` 扫描和 DFX 数字是他们的合同，不是你的 SLA。音频 **TTFP**（第一包音频）和文本 **TTFT** 不是同一只表。

`--omni` 会解析默认部署档案。`/v1/chat/completions` 出文本和音频；请求体里 `modalities`：`["text"]` 或 `["text", "audio"]`。三阶段各自 batch + CUDA Graph；async chunk / async omni output 避免等整包；Talker / Code2Wav 可 replica；hot-path 再削逐步开销。并发 64 的扫描（`Qwen3-Omni-30B-A3B-Instruct`）：Batch **2.2** req/s、音频 TTFP **5884 ms**、RTF **1.15** → 叠完 **11.7** req/s、**632 ms**、RTF **0.47**。吞吐最大一跳是 CUDA Graph（约 **4×**）；TTFP 砍得最深的是 async chunk（**2790 → 655 ms**）。yaml 的 `platforms:` 自动合并 CUDA / NPU / ROCm / XPU，不必再加旗标。

本地图（原文版权仍归原站；学习对照用）：

![qwen3 omni serving flow](../../../../assets/vllm/blog/serving/qwen3-omni/01-qwen3-omni-serving-flow.svg)

![qwen3 omni optimization stack](../../../../assets/vllm/blog/serving/qwen3-omni/02-qwen3-omni-optimization-stack.svg)

![qwen3 omni cuda graph stages](../../../../assets/vllm/blog/serving/qwen3-omni/03-qwen3-omni-cuda-graph-stages.svg)

![qwen3 omni async chunk timeline](../../../../assets/vllm/blog/serving/qwen3-omni/04-qwen3-omni-async-chunk-timeline.svg)

![qwen3 omni async output step gap](../../../../assets/vllm/blog/serving/qwen3-omni/05-qwen3-omni-async-output-step-gap.svg)

![qwen3 omni async replica](../../../../assets/vllm/blog/serving/qwen3-omni/06-qwen3-omni-async-replica.svg)

![qwen3 omni bench reqps](../../../../assets/vllm/blog/serving/qwen3-omni/07-qwen3-omni-bench-reqps.svg)

![qwen3 omni bench rtf](../../../../assets/vllm/blog/serving/qwen3-omni/08-qwen3-omni-bench-rtf.svg)

![qwen3 omni bench ttfp](../../../../assets/vllm/blog/serving/qwen3-omni/09-qwen3-omni-bench-ttfp.svg)

原文 Figure 7–9 的 **alt text** 写并发 `1, 8, 16, and 32`。图注、扫描段、表用的是 `1 / 16 / 32 / 64`。这篇跟图注和表。

## Quickstart

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091
```

显式 staged 档案：

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091 \
  --deploy-config vllm_omni/deploy/qwen3_omni_moe.yaml
```

请求走 `/v1/chat/completions`。部署、async-chunk、多 replica：[Qwen3-Omni online serving guide](https://github.com/vllm-project/vllm-omni/blob/main/examples/online_serving/qwen3_omni/README.md)。原文**没有**内嵌一份请求 JSON；点名的请求体字段就是 `modalities`。

## Qwen3-Omni 的 serving 模型

文本 LLM 是一只环：Prefill、Decode、detokenize。Qwen3-Omni 在多模态推理后面再加两截语音，计算画像不一样：

```text
Thinker   -> multimodal understanding + text generation
Talker    -> hidden states and embeddings to RVQ codec codes
Code2Wav  -> codec codes to waveform audio
```

**Figure 1。** Thinker 出文本和 hidden states；Talker 出 codec；Code2Wav 还原音频。

## 优化总览

没有一只菜谱。每根杠杆对着一个阶段或一次交接。下面按他们验证的顺序：**Why** / **Why it works** / **What you gain**。

| Technique | Target stage / path | Problem it addresses | Primary benefit |
|---|---|---|---|
| Stage decomposition | Thinker → Talker → Code2Wav | 一只环、一套 batch/graph/设备政策；最慢的子路径卡住其余 | 每阶段独立 runtime |
| AR + Code2Wav batching | Talker MTP、Code2Wav async chunks | 单请求微工作在高并发下让 SM 空转 | 占满、req/s 上去 |
| CUDA Graph | Thinker / Talker / Code2Wav Decode | 每步 CPU kernel dispatch 把 TPOT 撑高；音频 RTF 过不了实时 | TPOT、音频 RTF 下来；扫描里吞吐约 4× |
| Async chunk | Thinker→Talker，Talker→Code2Wav | 整包屏障拖第一包音频 | 流水交接；音频 TTFP 砍得最深 |
| Async omni output | Thinker connector payload | 同步拼 payload 卡住 Thinker Decode worker | 吞吐回来，第一包音频不回潮 |
| Stage replicas | Talker、Code2Wav | 语音侧先饱和，Thinker 还有余量 | 只水平扩瓶颈阶段 |
| Hot-path cleanup | Talker code predictor、connector payload | 逐步 Python / 分配 / 同步随话语变长 | 每步延迟下来；叠在上面各层上 |

扫描：Seed-TTS `en`，`Qwen3-Omni-30B-A3B-Instruct`，prompt **10 / 160 / 320 / 640**，并发 **1 / 16 / 32 / 64**，**5** 次 warmup，三张可见 GPU 映射 `0/1/2`。每行单独重启、单独部署档案、只加一层。**Batch** 到 **Async output**：一阶段一卡（Thinker / Talker / Code2Wav 在 0 / 1 / 2，各 1 replica）。**Stage replicas**：Thinker 在 GPU 0；**2×** Talker + **2×** Code2Wav 在 GPU 1、2。下表是并发 **64**；图覆盖四个并发。GPU 型号、驱动、vLLM/Omni 版本原文**没写**。

| Step | Config added | Talker / Code2Wav replicas | Req/s | Mean audio TTFP | Mean audio RTF |
|---|---|---|---:|---:|---:|
| Baseline | Batch | 1 / 1 | 2.2 | 5884 ms | 1.15 |
| + CUDA Graph | Graph capture on Thinker, Talker, Code2Wav | 1 / 1 | 8.6 (+299%) | 2790 ms (−53%) | 0.59 (−49%) |
| + Async chunk | Async-chunk stage handoffs | 1 / 1 | 9.3 (+8%) | 655 ms (−77%) | 0.63 |
| + Async output | Async omni output path | 1 / 1 | 11.3 (+22%) | 631 ms (−4%) | 0.47 (−25%) |
| + Stage replicas | 2× Talker + 2× Code2Wav | 2 / 2 | 11.7 (+4%) | 632 ms | 0.47 |

**Figure 2。** staged 执行、batch 与 replica、CUDA Graph、async chunk、hot-path。原图注：成绩来自 staged 数据流、阶段 runtime、Decode 热路径一起拧。

## 一层一层叠

每一步的数字都假定上面各层已经开着。

### 1. 阶段拆开再 batch：基线

**Why。** Thinker 是多模态 AR 文本；Talker 是 codec-predictor 的 AR；Code2Wav 是并行 vocoder Decode。塞进同一条 serving，就只能共用一套 batch / graph / 设备政策。拆开以后第二桩麻烦才露出来：语音侧仍在做单请求微工作。Talker 每步一次短的 code-predictor forward；Code2Wav 每块一次小 vocoder forward。并发 64，一个一个跑，SM 空着，固定每步成本摊不掉。

**Why it works。** 阶段边界变成一等公民：connector 运 hidden states、embedding、codec、chunk metadata；调度器按各自关键路径 batch、graph。并发请求收进一次 Talker MTP、一次 Code2Wav forward。

**What you gain。** 各自的 `max_num_seqs`、sampling、connector、graph/eager、可选 replica——下面所有优化的前提。这份拆开再 batch 的配置**就是** Batch 基线。

### 2. CUDA Graph：每阶段 Decode 捕获

**Why。** Batch 把占用拉上来了；每步 Decode 仍在付 CPU kernel dispatch。Talker 一句话可能几百步短计算。并发 64，这笔发射税压住 **TPOT**，音频 RTF 过不了实时。

**Why it works。** 固定算子序列捕获一次，热路径几乎不走 CPU。Decode 形状打进稳定的 `(batch, seq, frames)` 桶；warmup 时记下，热路径重放。每阶段捕获点不同，道理一样。

**Figure 3。** Thinker、Talker 走 vLLM 外层 Decode graph；Talker 内侧 code predictor 用 `torch.compile`（不是第二张 graph）；Code2Wav 用内侧 `CUDAGraphDecoderWrapper`。

**Stage 0 — Thinker（`LLM_AR`）。** `enforce_eager` 为 false 时，Decode 上 vLLM 标准 CUDA Graph，和文本 serving 同一套。

**Stage 1 — Talker。** `enforce_eager: false` 时外层 CUDA Graph。每步还跑 **code predictor**（短 re-prefill transformer → RVQ codes）：

- `torch.compile` 融合 5 层 predictor（`dynamic=False`，`epilogue_fusion=False`），RMSNorm/RoPE 仍对齐参考路径，kernel 数下来。
- CUDA 上默认**不开**第二层手工 CUDA Graph（`use_cuda_graphs=False`）——会和 Talker 的 `CUDAGraphWrapper` 打架。外层 graph + 编译后的内侧 forward 是互补：一张抓住 AR 阶段环，另一张融合 codec 预测那截微 forward。
- 可选 prefix-graph 桶：connector 配置里的 `code_predictor_prefix_graphs`，要显式打开。

**Stage 2 — Code2Wav（`LLM_GENERATION`）。** 内侧 `CUDAGraphDecoderWrapper`，不是 vLLM 外层 wrapper：

```python
# Enabled during weight load when stage enforce_eager is false
self.code2wav.enable_cudagraph(
    codec_chunk_frames=chunk_frames,
    codec_left_context_frames=left_frames,
)
```

形状从 connector 读：`codec_chunk_frames`、`codec_left_context_frames`。捕获枚举 async-chunk 和整包 Decode 会撞上的 `(batch, num_quantizers, frames)` 桶，包括 `initial_codec_chunk_frames` 那块更小的第一包。vocoder warmup：捕获前 `precompute_snake_caches()`，SnakeBeta 不要在 graph 里付启动。async-chunk：`chunked_decode_streaming` → `_cudagraph_wrapper.chunked_decode_with_cudagraph`；整包路径形状对上捕获桶就走 batched decode。

**What you gain。** 三阶段都上 graph：req/s **2.2 → 8.6**（+299%），平均音频 TTFP **5884 → 2790 ms**，平均音频 RTF **1.15 → 0.59**。页上把这一跳的大头记在 Thinker 文本生成、Talker codec Decode、Code2Wav vocoder **一起**卸掉发射税。

### 3. Async chunk：阶段之间流水交接

**Why。** 每截都快了，管线仍是**屏障同步**。Talker 要等 Thinker 完；Code2Wav 要等 Talker 整包。第一包音频要等完整 Thinker 生成再加完整 Talker Prefill——哪怕几帧 codec 就够发出第一段能听的声音。

**Why it works。** 部分交接：Thinker 增量吐 embedding 行；Talker 按 `initial_codec_chunk_frames` / `codec_chunk_frames` 切片；异步调度让传块和计算重叠。

**Figure 4。** 屏障路径等整包；async chunk 重叠之后，Code2Wav 在几帧 codec 之后就能开口。

**What you gain。** 音频 TTFP 最大一刀：**2790 → 655 ms**。

### 4. Async output：payload 别堵 Decode

**Why。** 交接已经是增量的，**同步**拼 payload（每个 chunk 边界拷 embedding 和 hidden states）仍能把 Thinker Decode worker 卡住。阶段交接已经是增量，GPU 时间却还在付给 Python 调度。

**Why it works。** `async_omni_output`：Thinker 把 Decode 状态交给非阻塞输出路径，立刻去下一个 token；connector 在旁边拼块、送走。

**Figure 5。** 同步 payload：Talker 步之间 GPU 闲约 **2.8 ms**。之后步间隙约 **41 µs**。

**What you gain。** 叠在 async chunk 上、并发 64：平均音频 TTFP 约 **631 ms**；平均音频 RTF **0.63 → 0.47**。TTFP 几乎不动；动的是 RTF 和 req/s（**9.3 → 11.3**）。

### 5. Stage replicas：只扩 Talker 和 Code2Wav

**Why。** Thinker 出一次文本；Talker 和 Code2Wav 再跑几百步短计算。并发 64，单只语音 replica 成了尾巴，Thinker 还有余量。整条管线克隆会把那只大的多模态 Thinker 再买一份。

**Why it works。** 只复制先饱和的阶段。GPU 0 上一只 Thinker，喂 GPU 1、2 上的 2× Talker 和 2× Code2Wav：

```json
{
  "stage_overrides": {
    "1": {"num_replicas": 2, "devices": "1,2"},
    "2": {"num_replicas": 2, "devices": "1,2"}
  }
}
```

**Figure 6。** 语音侧的 async chunk + replica。

**What you gain。** 并发 64 到 **11.7** req/s——扫描峰值——音频 TTFP 约 **632 ms**，RTF 约 **0.47**。并发越高，replica 的余量越明显。表上 replica 相对 async output 在 c=64 只 **+4%** req/s；图里还有 c=32 的 **6.8** req/s。

### 6. Hot-path：Talker Decode 和 connector payload

**Why。** 框架级瓶颈走了。profile 里还剩一条随话语变长的尾巴：多余的 connector 流量、逐步 `torch.cat` 和 CPU 序列化、codec predictor 里的 Python dispatch、下一步还要回到 GPU 的 Decode 状态却先 D2H。

**Why it works**（音频输出不变）：

- **Decode-only connector。** 第 0 块仍送完整 Thinker Prefill；之后每步只送新的 `embed.decode` 行。connector 流量 **O(1) per step**，不随 prompt 变长。
- **单卡 executor 默认。** 去掉 `distributed_executor_backend` 隐式 `"mp"`，单卡走 `uni`，少付多进程启动 / IPC / worker 同步。
- **Connector payload。** 按 token 送 Decode embedding，少反复 `torch.cat`。最终阶段已在本地时，不建下游 pooler/multimodal 的 CPU payload（没有 hidden-state D2H）。
- **Talker code predictor 重写。** 不再把极短序列丢给 Hugging Face `generate()`。re-prefill + SDPA、原生 GQA、inline top-k、缓存模块引用、内侧 transformer `torch.compile`。CUDA 上这条编译路径坐在 Talker CUDA Graph **下面**（见 §2），不是第二张会打架的 graph。
- **Decode 状态留 GPU。** `hidden_states.last`、`hidden_states.trailing_text`、`embed.tts_pad_projected`、`codes.audio` 放在 `model_intermediate_buffer`。Talker / Code2Wav 不做多模态 `get_mrope_input_positions`（只要便宜的线性位置）。`_store_value` 在已经在 CPU 时不再 `.to("cpu")`。
- **数值护栏。** RMSNorm 方差和 RoPE 留 fp32（`epilogue_fusion=False`）；每次调用自己的 embedding buffer，避免跨请求别名。

**What you gain。** 长上下文单请求：E2EL **21.28 s → 7.37 s**，音频 TTFP **3197 → 1796 ms**，音频 RTF **0.71 → 0.28**。叠在上面各层上；出现在 DFX perf 套件里，**不是**扫描表单独一行。这组 E2EL 的 prompt 长度、GPU、并发原文**没写**。

## 验证结果

和总览同一套扫描，四个并发（`1` / `16` / `32` / `64`），从 Batch 起。

**Figure 7。** req/s：c=1 橙、c=16 紫、c=32 绿、c=64 红。replica：**11.7** req/s（c=64）、**6.8**（c=32），相对 Batch c=64 的 **2.2**。

**Figure 8。** 平均音频 RTF。Batch 在负载下达到或超过实时（c=64 到 **1.15**）；async output 和 replica 把 c=32 / c=64 压在约 **0.47** 及以下。

**Figure 9。** 平均音频 TTFP，毫秒，对数轴。async chunk 把 c=64 从约 **5884 ms**（Batch）打到约 **655 ms**。

### 三张图一起读

- **吞吐（Figure 7）。** c=64：**2.2 → 11.7** req/s（约 **5.4×**）。c=32：**1.1 → 6.8**。最大一跳：CUDA Graph（约 **4×**）。高并发最后一大推是 async output；replica 把峰值顶上。
- **RTF（Figure 8）。** c=64 **1.15 → 0.47**——Decode 从跟不上播放变成跑在前面。
- **第一包（Figure 9）。** c=64 约 **5884 → ~632 ms**；async chunk 砍得最深（到约 **655 ms**）；后面几层把这个收益按住。

延迟主要靠 CUDA Graph 和 async chunk；高并发吞吐靠 async output 和 replica。叠完他们说有实时余量——那是他们这套机器上的合同，不是另一集群的保证。

## 致谢

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) 里 Qwen3-Omni 的贡献者：Haiyan Wu、Taichang Zhou、Canlin Guo、Ruirui Yang、Ziming Huang、Wengang Zheng、Lianhao Xu、Han Gao、Junhong Liu、Samit Huang、Hao Chen、Alex Brooks、Chenguang Zheng、Peiqi Yin、Wenjing Chen、Nick Cao、Shunyang Li、Yong Yang、Divyansh Singhvi、Yueqian Lin、Dayu Qiu、Roger Wang、Hongsheng Liu。

## 参考文献

**源码与配置：** [`pipeline.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen3_omni/pipeline.py)；[`qwen3_omni.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py)；[`stage_input_processors/qwen3_omni.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/stage_input_processors/qwen3_omni.py)；部署 [`qwen3_omni_moe.yaml`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/qwen3_omni_moe.yaml)；DFX [`test_qwen3_omni_async_chunk.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_qwen3_omni_async_chunk.json)、[`test_qwen3_omni_multi_replicas.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_qwen3_omni_multi_replicas.json)；模型 [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)。

**PR：** CUDA Graph（§2）Thinker [vllm-omni#523](https://github.com/vllm-project/vllm-omni/pull/523)、Talker [#669](https://github.com/vllm-project/vllm-omni/pull/669)、Code2Wav [#2376](https://github.com/vllm-project/vllm-omni/pull/2376)。Async chunk（§3）跨阶段切块计算/通信 [#727](https://github.com/vllm-project/vllm-omni/pull/727)、异步调度让 chunk IO 和计算重叠 [#951](https://github.com/vllm-project/vllm-omni/pull/951)、包间延迟 [#1656](https://github.com/vllm-project/vllm-omni/pull/1656)。Async output（§4）async omni output 物化 [#4476](https://github.com/vllm-project/vllm-omni/pull/4476)。Stage replicas（§5）多阶段部署 [#2396](https://github.com/vllm-project/vllm-omni/pull/2396)、阶段 runtime 与分布式 replica 控制面 [#3855](https://github.com/vllm-project/vllm-omni/pull/3855)。Hot-path（§6）[#3007](https://github.com/vllm-project/vllm-omni/pull/3007)、[#3164](https://github.com/vllm-project/vllm-omni/pull/3164)、[#3878](https://github.com/vllm-project/vllm-omni/pull/3878)。

[vLLM Slack](https://slack.vllm.ai) 的 `#sig-omni`；issue 去 [vLLM-Omni GitHub](https://github.com/vllm-project/vllm-omni)。
