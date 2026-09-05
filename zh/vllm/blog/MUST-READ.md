---
source: https://vllm.ai/llms.txt
lang: zh
fetched: 2026-08-31
---

# 建议先读的 vLLM 博客

完整机器表：英文 [`CATALOG.md`](../../../en/vllm/blog/CATALOG.md)。入口 https://vllm.ai/blog ，RSS https://vllm.ai/blog/rss.xml

旋钮对照：[FLAG-MAP.md](FLAG-MAP.md)（`optimization.md` 里的旗标 → 这篇博客）。

**阅读顺序（最佳）：** 立项 → Anatomy → V1 → MRV2 →（可插 DeepSpeed 笔记）→ v0.6 CPU → 投机解码 → FP8 KV → 生产级 CI → 分布式切卡 → production-stack → AIBrix → Router → Encoder 分离 → 大规模 Wide-EP → Mooncake → Elastic EP。

V1 / spec-decode 文中的「还不支持」是当时的缺口。EPD 那篇是 **视觉编码器分离**，文本 Prefill/Decode 分离在 Router 与大规模两篇。Mooncake 是跨实例 KV 池（agent 前缀）；Elastic EP 是运行时改 DP 宽度。

## architecture

| 文 | 本地 | URL |
|---|---|---|
| PagedAttention 立项 | [paged-attention.md](architecture/paged-attention.md) | https://vllm.ai/blog/2023-06-20-vllm |
| Anatomy of vLLM | [anatomy.md](architecture/anatomy.md) | https://vllm.ai/blog/2025-09-05-anatomy-of-vllm |
| V1 alpha | [v1-alpha.md](architecture/v1-alpha.md) | https://vllm.ai/blog/2025-01-27-v1-alpha-release |
| Model Runner V2 | [mrv2.md](architecture/mrv2.md) | https://vllm.ai/blog/2026-03-24-mrv2 |
| vLLM vs DeepSpeed | [vs-deepspeed.md](architecture/vs-deepspeed.md) | https://vllm.ai/blog/2023-11-14-notes-vllm-vs-deepspeed |

## performance

| 文 | 本地 | URL |
|---|---|---|
| v0.6 吞吐 | [v0.6-throughput.md](performance/v0.6-throughput.md) | https://vllm.ai/blog/2024-09-05-perf-update |
| Speculative decoding | [spec-decode.md](performance/spec-decode.md) | https://vllm.ai/blog/2024-10-17-spec-decode |
| FP8 KV cache | [fp8-kvcache.md](performance/fp8-kvcache.md) | https://vllm.ai/blog/2026-04-22-fp8-kvcache |
| 生产级 CI | [production-quality.md](performance/production-quality.md) | https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality |

## serving

| 文 | 本地 | URL |
|---|---|---|
| 分布式推理 | [distributed-inference.md](serving/distributed-inference.md) | https://vllm.ai/blog/2025-02-17-distributed-inference |
| Production stack | [production-stack.md](serving/production-stack.md) | https://vllm.ai/blog/2025-01-21-stack-release |
| AIBrix | [aibrix.md](serving/aibrix.md) | https://vllm.ai/blog/2025-02-21-aibrix-release |
| Router | [router.md](serving/router.md) | https://vllm.ai/blog/2025-12-13-vllm-router-release |
| Encoder 分离 (EPD) | [epd.md](serving/epd.md) | https://vllm.ai/blog/2025-12-15-vllm-epd |
| 大规模 serving | [large-scale.md](serving/large-scale.md) | https://vllm.ai/blog/2025-12-17-large-scale-serving |
| Mooncake Store | [mooncake.md](serving/mooncake.md) | https://vllm.ai/blog/2026-05-06-mooncake-store |
| Elastic EP | [elastic-ep.md](serving/elastic-ep.md) | https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism |

## 第二波（机制，必读之后）

主线走完再读。不是 day-0 模型文。顺序：torch.compile → Sleep Mode → structured decoding → DCP → KV offload → 单机 P/D（MORI-IO）→ Hybrid SSM → AFD。

| 文 | 本地 | URL |
|---|---|---|
| torch.compile | [torch-compile.md](architecture/torch-compile.md) | https://vllm.ai/blog/2025-08-20-torch-compile |
| Sleep Mode | [sleep-mode.md](architecture/sleep-mode.md) | https://vllm.ai/blog/2025-10-26-sleep-mode |
| Structured decoding | [struct-decode.md](performance/struct-decode.md) | https://vllm.ai/blog/2025-01-14-struct-decode-intro |
| Decode Context Parallelism | [dcp.md](performance/dcp.md) | https://vllm.ai/blog/2026-08-07-decode-context-parallelism |
| KV offloading connector | [kv-offload.md](serving/kv-offload.md) | https://vllm.ai/blog/2026-01-08-kv-offloading-connector |
| 单机 P/D（MORI-IO） | [moriio.md](serving/moriio.md) | https://vllm.ai/blog/2026-04-07-moriio-kv-connector |
| Hybrid SSM 分离 | [hybrid-ssm.md](serving/hybrid-ssm.md) | https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg |
| AFD Plugin | [afd.md](serving/afd.md) | https://vllm.ai/blog/2026-07-23-vllm-afd-plugin |

## 第三波（插件 / KV 池 / 量化 / RL）

第二波之后。仍不是 day-0。顺序：插件 → 硬件插件 → Triton attention → SHM IPC → PegaFlow → TurboQuant → Native RL → Ray symmetric-run。

| 文 | 本地 | URL |
|---|---|---|
| 插件系统 | [plugin-system.md](architecture/plugin-system.md) | https://vllm.ai/blog/2025-11-20-vllm-plugin-system |
| Hardware plugin | [hardware-plugin.md](architecture/hardware-plugin.md) | https://vllm.ai/blog/2025-05-12-hardware-plugin |
| Triton attention | [triton-attn.md](architecture/triton-attn.md) | https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive |
| SHM IPC cache | [shm-ipc.md](serving/shm-ipc.md) | https://vllm.ai/blog/2025-11-13-shm-ipc-cache |
| PegaFlow | [pegaflow.md](serving/pegaflow.md) | https://vllm.ai/blog/2026-05-18-pegaflow |
| TurboQuant | [turboquant.md](performance/turboquant.md) | https://vllm.ai/blog/2026-05-11-turboquant |
| Native RL APIs | [native-rl.md](serving/native-rl.md) | https://vllm.ai/blog/2026-05-28-native-rl-apis |
| Ray symmetric-run | [ray-symmetric.md](serving/ray-symmetric.md) | https://vllm.ai/blog/2025-11-22-ray-symmetric-run |

## 第四波（投机后续 / 异构 serving / 历史路线图）

第三波之后。顺序：hidden 导出 → P-EAGLE → 并行草稿 → DSpark 自适应 → 流式输入 → TileRT → DeepSeek V4 → ROCm attention → GB200 Wide-EP → OpenRLHF → LF AI 路线图。

| 文 | 本地 | URL |
|---|---|---|
| Hidden states 导出 | [extract-hidden-states.md](architecture/extract-hidden-states.md) | https://vllm.ai/blog/2026-03-30-extract-hidden-states |
| P-EAGLE | [p-eagle.md](performance/p-eagle.md) | https://vllm.ai/blog/2026-03-13-p-eagle |
| 并行草稿 | [parallel-drafting.md](performance/parallel-drafting.md) | https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting |
| DSpark 自适应验收 | [dspark-adaptive.md](performance/dspark-adaptive.md) | https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification |
| 流式输入 / realtime | [streaming-realtime.md](serving/streaming-realtime.md) | https://vllm.ai/blog/2026-01-31-streaming-realtime |
| TileRT P/D | [tilert.md](serving/tilert.md) | https://vllm.ai/blog/2026-07-14-vllm-tilert-pd |
| DeepSeek V4 | [deepseek-v4.md](architecture/deepseek-v4.md) | https://vllm.ai/blog/2026-04-24-deepseek-v4 |
| ROCm attention | [rocm-attention.md](architecture/rocm-attention.md) | https://vllm.ai/blog/2026-02-27-rocm-attention-backend |
| GB200 Wide-EP | [gb200-wideep.md](serving/gb200-wideep.md) | https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1 |
| OpenRLHF | [openrlhf.md](serving/openrlhf.md) | https://vllm.ai/blog/2025-04-23-openrlhf-vllm |
| LF AI 路线图 | [lfai-roadmap.md](architecture/lfai-roadmap.md) | https://vllm.ai/blog/2024-07-25-lfai-perf |

## 第五波（对齐 / 草稿训练 / TPU / 路由 / Omni）

第四波之后。顺序：AMD 投机 → RDT 权重 → IsoExec → bitwise RL → token IDs → Speculators 0.3/0.5 → EAGLE 3.1 → CUDA dump → TPU → PTPC-FP8 → Transformers backend → Semantic Router → Iris → Omni。

| 文 | 本地 | URL |
|---|---|---|
| AMD 投机解码 | [spec-decode-amd.md](performance/spec-decode-amd.md) | https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus |
| RDT 分片权重 | [rdt-weight-transfer.md](serving/rdt-weight-transfer.md) | https://vllm.ai/blog/2026-08-22-rdt-weight-transfer |
| IsoExec | [isoexec.md](serving/isoexec.md) | https://vllm.ai/blog/2026-08-21-isoexec |
| 逐 bit RL | [bitwise-rl.md](serving/bitwise-rl.md) | https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference |
| return_token_ids | [agent-lightning.md](serving/agent-lightning.md) | https://vllm.ai/blog/2025-10-22-agent-lightning |
| Speculators 0.3 | [speculators-v030.md](performance/speculators-v030.md) | https://vllm.ai/blog/2025-12-13-speculators-v030 |
| Speculators 0.5 | [speculators-v050.md](performance/speculators-v050.md) | https://vllm.ai/blog/2026-05-28-speculators-v050 |
| EAGLE 3.1 | [eagle-3-1.md](performance/eagle-3-1.md) | https://vllm.ai/blog/2026-05-26-eagle-3-1 |
| CUDA core dump | [cuda-debugging.md](architecture/cuda-debugging.md) | https://vllm.ai/blog/2025-08-11-cuda-debugging |
| Hang → 源码行 | [cuda-debugging-source.md](architecture/cuda-debugging-source.md) | https://vllm.ai/blog/2025-12-03-improved-cuda-debugging |
| vLLM TPU | [vllm-tpu.md](architecture/vllm-tpu.md) | https://vllm.ai/blog/2025-10-16-vllm-tpu |
| PTPC-FP8 | [ptpc-fp8.md](performance/ptpc-fp8.md) | https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm |
| Transformers backend | [transformers-backend.md](architecture/transformers-backend.md) | https://vllm.ai/blog/2025-04-11-transformers-backend |
| Semantic Router | [semantic-router.md](serving/semantic-router.md) | https://vllm.ai/blog/2025-09-11-semantic-router |
| Iris v0.1 | [semantic-router-iris.md](serving/semantic-router-iris.md) | https://vllm.ai/blog/2026-01-05-vllm-sr-iris |
| vLLM-Omni | [vllm-omni.md](serving/vllm-omni.md) | https://vllm.ai/blog/2025-11-30-vllm-omni |

## 其余 CATALOG（已齐，按需）

带日期的博客 **129/129** 都有压缩学习译文。全表：[README.md](README.md)。不必按波次读完。下面几簇机制含量高，主线走完再抽：

- Omni / TTS / 扩散 RL：`minimax-h3`、`omni-tts`、`omni-diffusion-cache`、`omni-autoround`、`qwen3-omni`、`omni-layerwise-offload`、`verl-omni`、`verl-omni-v020`
- RL 对齐：`vime`、`vime-rocm`、`openrlhf`（第四波已列）
- Pareto / 硬件：`qwen35-25k-tps`、`glm52-b300`、`hpc-ops`、`gb300-deepseek`、`gpt-oss-optimizations`、`blackwell-inferencemax`、`artificial-analysis`、`eagle3-amd`
- Day-0 / 社区：Nemotron 3 系、Gemma 4、Llama 3.1/4、gpt-oss 上手、meetup、playground、vllm.ai 网站

Semantic Router 后续（signal / HaluGate / Athena / Themis / MoM / Fusion / session）在 `serving/semantic-router-*.md`。
