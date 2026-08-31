---
source: https://docs.vllm.ai/en/stable/configuration/optimization/
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# 旋钮 → 博客：从 `optimization.md` 走到必读线

英文对照：`en/vllm/blog/FLAG-MAP.md`  
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
| 路由 / 控制面 | （不在 optimization 页） | production-stack / [AIBrix](serving/aibrix.md) / Router | 引擎上面的盘子可以换；记忆亲和不会消失。 |

NVIDIA 侧同一张切卡地图：[trtllm-sharding](../../nvidia/performance-tuning/trtllm-sharding.md)。官方 sharding CLI 有一处把 `--tp_size` 写了两次、PP 应为 `--pp_size`，那章里注过。
