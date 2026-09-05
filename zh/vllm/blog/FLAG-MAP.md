---
source: https://docs.vllm.ai/en/stable/configuration/optimization/
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# 旋钮 → 博客：从 `optimization.md` 走到必读线

英文对照：[en/vllm/blog/FLAG-MAP.md](../../../en/vllm/blog/FLAG-MAP.md)  
文档页告诉你旋钮的礼貌顺序；博客告诉你这些旋钮是怎么长出来的。CLI 与指标名保持英文。数字以各篇原文为准。

完整阅读顺序：[MUST-READ.md](MUST-READ.md)。优化页：[optimization.md](../optimization/optimization.md)。serve 旗标摘录：[serve.md](../getting-started/serve.md)。

| 你拧的 | 文档里在哪 | 先读哪篇博客 | 一句话 |
|---|---|---|---|
| KV 块 / PagedAttention | 抢占、cache 大小 | [立项](architecture/paged-attention.md) → [Anatomy](architecture/anatomy.md) | 房子是 KV，不是权重。 |
| `-O0`…`-O3`、CUDA graph、`torch.compile` | 优化等级；`--enforce-eager` | [torch.compile](architecture/torch-compile.md)、[v0.6](performance/v0.6-throughput.md)、[V1](architecture/v1-alpha.md)、[MRV2](architecture/mrv2.md) | 启动时间和稳态 decode 的交易；融合发生在编译时。 |
| `max_num_batched_tokens`、chunked prefill | Chunked Prefill | [vs DeepSpeed](architecture/vs-deepspeed.md)、Anatomy | 2023 年叫 SplitFuse；V1 默认切开。小值护 ITL，大值护 TTFT，吞吐常要 >8192。 |
| prefix cache | features 页 | Anatomy、[production-stack](serving/production-stack.md)、[Router](serving/router.md)、[Mooncake](serving/mooncake.md)、[KV offload](serving/kv-offload.md) | 本机命中 → 粘会话；被抢占 → CPU；粘不住 → 分布式池。 |
| speculative decoding | features 页 | [spec-decode](performance/spec-decode.md) | draft/verify；文中「还不支持」是历史。 |
| FP8 KV / attention 量化 | 显存与精度 | [FP8 KV](performance/fp8-kvcache.md)、[TurboQuant](performance/turboquant.md) | 默认 FP8；3–4 bit 存储要付反量化，别当免费午餐。 |
| `gpu_memory_utilization`、preemption | Preemption | Anatomy、立项 | V1 默认 RECOMPUTE。频繁抢占先给 KV 房间。 |
| TP / PP | 并行策略 | [分布式推理](serving/distributed-inference.md)；TRT-LLM 手册切卡章 | 节点内 TP、节点间 PP；MLA 上别用纯 TP 硬切。 |
| EP / DP / `--enable-expert-parallel` | Expert / Data Parallelism | [Wide-EP](serving/large-scale.md) | 密层 DP Attention，稀疏层 EP。 |
| `--enable-dbo`、EPLB | （博客先于文档页） | Wide-EP | 通信胖时重叠微批；线上专家负载要热替换。 |
| `--enable-elastic-ep` | （博客） | [Elastic EP](serving/elastic-ep.md) | 运行时改 DP 个数；当时 TP=1、无 DBO、Ray only。 |
| P/D 分离（文本） | 部署走廊 | [Router](serving/router.md)、Wide-EP | 一条胖 prefill 能拖住整组 EP。 |
| `mm_encoder_tp_mode="data"`、多模态 cache | Encoder DP；多模态缓存 | [EPD](serving/epd.md) | 单机按 batch 切编码器；集群把 ViT 拆到另一栋楼。 |
| KVConnector / 外置 KV | （博客） | Mooncake、[KV offload](serving/kv-offload.md)、[MORI-IO](serving/moriio.md)、[PegaFlow](serving/pegaflow.md)、production-stack | 同一扇门：本机 DRAM、集群池、单机 RDMA、独立 Rust 守护进程。 |
| `--api-server-count`、CPU 核 | API 横向扩展；CPU 资源 | [v0.6](performance/v0.6-throughput.md)、Anatomy | V1 多进程；核不够时 GPU 在等端菜的人。 |
| 生产是否可发 | CI / 发布节奏 | [生产级 CI](performance/production-quality.md) | 夜测、多加速器、两周发布。 |
| `--enable-sleep-mode` | （博客） | [Sleep Mode](architecture/sleep-mode.md) | 换模型不拆进程；L1 卸 CPU，L2 丢权重。 |
| guided / structured decoding | sampling | [structured decoding](performance/struct-decode.md) | schema 当 logit mask；JSON / tool-call 的栅栏。 |
| `-dcp` / `--decode-context-parallel-size` | 并行；serve CLI | [DCP](performance/dcp.md) | 按序列切 decode KV；MLA/GQA 约束不同。 |
| `--kv_offloading_*` | Preemption 的下一层 | [KV offload](serving/kv-offload.md) | 异步卸到 CPU，避免 RECOMPUTE。 |
| 混合 SSM 的 P/D | （博客） | [Hybrid SSM](serving/hybrid-ssm.md) | FA 与 Mamba 两套 NIXL 描述符。 |
| AFD / Attention-FFN 分离 | （插件） | [AFD](serving/afd.md) | MoE 层内拆服务；实验性，配比决定输赢。 |
| 单机 P/D | （博客） | [MORI-IO](serving/moriio.md) | 8 卡盒子里也能拆；Write 默认更好 TTFT。 |
| `mm_processor_cache_type="shm"` | 多模态缓存 | [SHM IPC](serving/shm-ipc.md) | 大图走共享内存，别在进程间复印。 |
| 硬件 / 平台插件 | （博客） | [插件](architecture/plugin-system.md)、[hardware plugin](architecture/hardware-plugin.md) | 改调度或换卡，不必养 fork。 |
| Attention backend | 自动选择 | [Triton attention](architecture/triton-attn.md) | ROCm 默认；CUDA 上 FA 缺席时的便携路径。 |
| `turboquant_*` | `--kv-cache-dtype` | [TurboQuant](performance/turboquant.md) | 先读对照；生产默认仍是 FP8。 |
| 权重同步 / `pause keep` | （RL） | [Native RL](serving/native-rl.md) | 别再给每家框架补 worker；DPEP 两阶段 pause。 |
| `ray symmetric-run` | 多机启动 | [Ray symmetric-run](serving/ray-symmetric.md) | SLURM / mpssh 上每台同一条命令。 |
| `method: extract_hidden_states` | 投机配置 | [hidden 导出](architecture/extract-hidden-states.md) | dummy draft + KV Connector；只存 prompt。 |
| `"parallel_drafting": true` / `dflash` / `dspark` | 投机 | [P-EAGLE](performance/p-eagle.md)、[并行草稿](performance/parallel-drafting.md)、[AMD 投机](performance/spec-decode-amd.md) | 一次前向猜 K 个；N 不是越大越好。 |
| `enable_adaptive_verification` | 投机 | [DSpark 自适应](performance/dspark-adaptive.md) | 预算 = 信心 × 负载；要 FULL varlen decode graph。 |
| `/v1/realtime`、StreamingInput | （博客） | [流式输入](serving/streaming-realtime.md) | 模型必须因果；chunked prefill 是另一件事。 |
| `--kv-cache-dtype fp8` + `--block-size 256`（V4） | 显存 | [DeepSeek V4](architecture/deepseek-v4.md) | 逻辑 block 256 原生位置；压缩机残差当滑窗。 |
| `--quantization ptpc_fp8` | ROCm 量化 | [PTPC-FP8](performance/ptpc-fp8.md) | 权重量化，不是 KV dtype。 |
| `model_impl="transformers"` | 模型实现 | [Transformers backend](architecture/transformers-backend.md) | 覆盖面，不是性能默认。 |
| `return_token_ids` | OpenAI API | [Agent Lightning](serving/agent-lightning.md) | Agent RL 禁止二次分词。 |
| RDT / `sharded_rdt` | RL 权重同步 | [RDT](serving/rdt-weight-transfer.md) | 缓冲不计入 `gpu_memory_utilization`；当时无 EPLB。 |
| 路由 / 控制面 | （不在 optimization 页） | production-stack / [AIBrix](serving/aibrix.md) / Router | 引擎上面的盘子可以换；记忆亲和不会消失。 |
| `--async-scheduling`、`--stream-interval` | API / 调度 | [gpt-oss 优化](performance/gpt-oss-optimizations.md)、[Qwen3.5 25K](serving/qwen35-25k-tps.md) | 藏 CPU；stream-interval 缓冲后续 token，伤 ITL。hybrid P/D 要先修竞态。 |
| `--gdn-prefill-backend`、`VLLM_SSM_CONV_STATE_LAYOUT=DS` | hybrid / P/D | [Qwen3.5 25K](serving/qwen35-25k-tps.md)、[Hybrid SSM](serving/hybrid-ssm.md) | GDN/Mamba 状态跟 KV 不是同一套搬运。 |
| `--language-model-only` | 多模态 | Qwen3.5 25K | 纯文本负载关掉视觉，才能走 fused QK-norm+RoPE。 |
| `--enable-distributed-layerwise-offload` | Omni | [DLO](serving/omni-layerwise-offload.md) | DiT 权重按层流；设备上只双缓冲两层。 |
| `--omni`、`cache_backend` | Omni | [Omni](serving/vllm-omni.md)、[扩散 cache](serving/omni-diffusion-cache.md)、[TTS](serving/omni-tts.md) | 文本 TTFT ≠ 音频 TTFP；cache 吃时间冗余。 |
| `--attention-backend HPC_ATTN`、`--moe-backend hpc` | backend | [HPC-Ops](performance/hpc-ops.md) | 当时 Hy3 / FP8 / Hopper，不是通用默认。 |
| `VLLM_USE_V2_MODEL_RUNNER=1` | MoE 运行时 | [GLM-5.2 SLA](serving/glm52-b300.md)、[MRV2](architecture/mrv2.md) | dense 已默认 V2；MoE 要显式开。 |
| `--block-size 128`（MSA） | 长上下文 | [MiniMax M3](serving/minimax-m3.md) | 对齐稀疏 attention 的 128-token 块，不是随便选。 |

NVIDIA 侧同一张切卡地图：[trtllm-sharding](../../nvidia/performance-tuning/trtllm-sharding.md)。官方 sharding CLI 有一处把 `--tp_size` 写了两次、PP 应为 `--pp_size`，那章里注过。
