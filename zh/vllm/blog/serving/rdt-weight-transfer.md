---
source: https://vllm.ai/blog/2026-08-22-rdt-weight-transfer
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# RDT 分片权重搬运：Kimi K2 7.53 秒

英文对照：[en/vllm/blog/serving/rdt-weight-transfer.md](../../../../en/vllm/blog/serving/rdt-weight-transfer.md)  
原文：https://vllm.ai/blog/2026-08-22-rdt-weight-transfer  
2026-08-22。Aaron Hao、Sumanth Hegde、Gal Meirom、Istvan Haller、Kourosh Hakhamaneshi、Gavin Parnaby、Moein Khazraee、Omri Kahalon。文档：[sharded RDT](https://docs.vllm.ai/en/latest/training/weight_transfer/sharded_rdt/)。端到端：[SkyRL `sharded_rdt`](https://github.com/NovaSky-AI/SkyRL/tree/main/examples/train/megatron/sharded_rdt)。这篇是 **分片怎么搬**。Pause / keep / DPEP：[native-rl](native-rl.md)。更早的 Ray `WorkerExtension`：[openrlhf](openrlhf.md)。Ray 集群怎么起来：[ray-symmetric](ray-symmetric.md)。

Kimi K2 BF16，**48 × (8×H100)**：训练 **32** 节点、推理 **16** 节点，约 **7.9 TB** 用 **7.53 s**，聚合带宽 **1,049 GB/s**。

## Introduction

在线 RL 里，权重要定期对齐，rollout 才是最近一版 checkpoint 吐出来的。模型涨到万亿参数，这次同步的显存和墙钟都会卡住训练环。

这篇在 vLLM 里做了一套原生分片传送，底层是 [Ray Direct Transport (RDT)](https://docs.ray.io/en/latest/ray-core/api/direct-transport.html)。原文贡献四条：

- **原生分片引擎**：dense、MoE（fused 或按 expert 存的 checkpoint）、量化都覆盖，挂在 [Native RL APIs](https://vllm.ai/blog/2026-05-28-native-rl-apis) 的 `WeightTransferEngine` 上。
- **训练框架只描述布局**：怎么切、谁持有哪一块；传送整段归引擎。
- **预处理和传送重叠**：gather、transfer、后处理互相盖住。
- **故障演示**：RDT + NIXL 下推理副本挂了还能接着跑。

本地图（原文版权仍归原站；学习对照用）：

![rdt blog overview](../../../../assets/vllm/blog/serving/rdt-weight-transfer/01-rdt_blog_overview.png)

**总览。** Broadcast 对 RDT（NIXL）分片。NCCL：trainer rank 0 和所有推理 rank 进同一个 collective，把 **整份** HuggingFace 权重 broadcast 出去。分片引擎：**所有** trainer rank 都参与；每个推理 rank **只 pull 自己的分片**。再往下：不跨 PP gather，expert 层也不 gather。

## Background

标准做法是 NCCL broadcast。Trainer 把每个参数 all-gather 成 HuggingFace 布局，再灌给每一个推理 worker。小模型没问题。模型一大，两处开始疼：

1. **每个 worker 都收到整模。** TP8 只留 ⅛，其余扔掉。大 MoE（Kimi K2，常常 wide-EP）更糟：单层完整参数仍可到 **几十 GB**——峰值显存和传送时间一起涨。
2. **Broadcast 是集体通信。** NCCL 要组里每个人都到。掉队的 rank 会卡住 collective；副本一挂，整组可能要重建。

更早的大规模分片：[LMSYS P2P update](https://www.lmsys.org/blog/2026-04-29-p2p-update/)、[Perplexity，两秒以内](https://research.perplexity.ai/articles/weight-transfer-for-rl-post-training-in-under-2-seconds)。这篇盯的是 **通用**：vLLM 能伺候的模型几乎都要能走；别的 RL 框架只要肯描述布局就能接。

## Weight loading in vLLM

### The journey of a weight

HuggingFace 张量进 vLLM worker，不是一次 memcpy。原文列了七拍：

1. **Fuse** —— 比如 attention 里的 Q、K、V。
2. **Relayout** —— 按原 checkpoint 的格式转置 / reshape。
3. **Split / select** —— 切开，或只留一部分（expert parallelism）。
4. **Shard** —— 按 tensor parallelism 切片。
5. **Copy into buffer** —— 拷进按层分配的 staging（「layerwise buffer」）。
6. **Process** —— 可选量化，再加上 kernel 要的 padding / striding。
7. **Copy** —— 写进已经分配好的 live GPU 存储。

1–5 住在 weight loader 里，走 [layerwise reloading](https://docs.vllm.ai/en/latest/training/layerwise/)。CUDA graph 能留，额外显存也有顶。

![layerwise reloading](../../../../assets/vllm/blog/serving/rdt-weight-transfer/02-layerwise_reloading.webp)

Layerwise reloading（[来源](https://docs.vllm.ai/en/latest/training/layerwise/)）。

理想是传 **已经 process 完** 的权重（第 6 步之后），直接写进 live 存储。为了让量化、kernel packing 仍由 vLLM 做，他们传的是 **分片但未 process 的 BF16**——停在第 4 步——5–7 留给引擎。

### Custom weight loading behaviors

把 1–4 搬到训练侧，意味着 trainer 必须知道：每个 worker、每块权重，最后留下哪些字节。从并行配置手算一遍行不通——操作链随 **层、随模型** 变。原文两个例子：

1. **GQA 下的 QKV fusion。** `q_proj`、`k_proj`、`v_proj` 融进一张量。GQA 的 KV head 可以比 TP rank 少，于是两个 worker 可能 **Q 不同、K/V 相同**。标准 MHA 里 Q / K / V 的切法是一致的，这里不是。
2. **Llama-4 的 fused expert。** HuggingFace 里 expert 张量先转置，再拆成 `gate_proj` 和 `up_proj`，然后才按这个 worker 该拿的 expert 去选。

按模型手写 1–4 养不起。通用的办法：运行时把 loader **实际做的事记下来**。

### Solution: a recording-tensor dry run

引擎初始化时，给 loader 一只 **recording tensor**：shape、dtype 都对，**没有数据** 的张量子类。每一次 `view` / `narrow` / `transpose` / `reshape` 都接到一条操作链上。Loader 往 parameter 里 copy 时，记下从哪来、落到哪。这条链就是 **sharding plan**。

Plan 从 vLLM 自己的 loader 里长出来，对那些 loader 在各层、各模型上的行为是构造上正确的。训练侧：回放 1–4，送分片 BF16。推理侧：做 5–7，写进 live 权重。

## A sharded weight-transfer engine with RDT

verl、SkyRL、Slime、NemoRL 一类，调度多半走 [Ray](https://www.ray.io/)；训练 rank 和推理 rank 通常是 Ray actor。[RDT](https://docs.ray.io/en/latest/ray-core/api/direct-transport.html) 让 actor 方法直接返回 GPU 张量，不必先拷下 GPU。调用方拿到 [`ObjectRef`](https://docs.ray.io/en/latest/ray-core/objects.html)；读的时候，字节才经可插拔传输（NIXL、NCCL、Gloo）过去。

他们选 **NIXL**：P2P 灵活（每个 consumer 可以拿不同的权重），长跑也要得起故障。RDT+NIXL 是 **pull**：每个推理 rank 从映射到的若干 trainer rank 上，把需要的分片拉过来。

### At initialization

1. **Trainer 收集 ownership 元数据。** 每个参数：名字、dtype、完整 shape；再加上布局——哪些层（PP）、哪些名字（比如 EP 下一部分 expert）在这个 rank 上。Trainer rank 之间 all-gather。
2. **Rank 0 把 transfer 元数据发给推理 worker：** 参数 + ownership，以及做 RDT 要用的 trainer Ray actor 名。
3. **每个 vLLM worker 用 recording tensor 空跑**，记下 sharding plan。
4. **每个 vLLM worker 映射源 trainer rank。** 同一参数被多人持有时，按 **负载均衡** 选一个。Worker 摊到各 producer 上；**不同副本上同一个 worker rank** 去 **同一个** producer——少一份缓冲，传送也快一点。
5. **两边分配并注册 RDT buffer**，一次性跟 NIXL 登记。

![rdt blog init flow](../../../../assets/vllm/blog/serving/rdt-weight-transfer/03-rdt_blog_init_flow.png)

**初始化。** Trainer all-gather ownership；rank 0 下发 ownership + transfer 元数据；推理 rank 空跑 recording tensor；所有人注册 RDT buffer。

### During weight sync

1. **每个 trainer rank 一次 gather 一个 weight group。** Group 是一块 transformer（attention + MoE）。按层 all-gather，压显存。也可以只 gather 本地张量。他们的集成：**只跨 TP gather**——不跨 PP，EP 下 **也不 gather expert**。分散的 expert：初始化时就把推理 rank 映射到已经拿着那些 expert 的训练 rank。
2. **Worker pull 分片。** 顺着录好的 plan，向映射的 trainer actor 要下一批切片。Trainer 对着 gather 来的权重 **回放** 那些操作，打进已注册的 RDT buffer。Worker 用 RDMA 读进自己的 buffer。
3. **Worker 后台做 process + copy。** 后台线程把切片从 worker 侧 RDT buffer 拷进 layerwise buffer；引擎再 process + copy，变成 kernel 能用的 live 权重。
4. **Worker 释放这个 weight group。** 一组的最后一片拉完，向持有它的 trainer 发信号。所有 worker 都信号过了，trainer 丢掉这组 gather 出来的张量，才能 gather 下一组。
5. **Trainer 在没有 in-flight 之后关掉这次 sync**；worker 做完 layerwise reloading。

![AllScenes](../../../../assets/vllm/blog/serving/rdt-weight-transfer/04-AllScenes.gif)

**Attention 层的权重同步。** 一个 trainer rank、一个推理 rank；Q、K、V。

![ExpertScenes](../../../../assets/vllm/blog/serving/rdt-weight-transfer/05-ExpertScenes.gif)

**MoE 层的权重同步。** 还是这两 rank；expert。

## Performance optimizations

小规模旅程：**Qwen3-235B-A22B**，SkyRL（Megatron + vLLM）。**4 × 8×H100**：两台训练、两台推理。Megatron **TP4 / PP2 / EP8 / ETP1**；vLLM **DP16 / EP16**（wide-EP serving 的形状）。端到端同步延迟：含 all-gather 抽取，多次平均，**去掉第一次冷启动**。

同一套上 SkyRL 的 NCCL broadcast 基线：**64.72 s**。下面几个版本只改训练侧 **怎么 gather、怎么 iterate、怎么传**。映射、recording tensor 空跑，其余不动。

### V1 — a simple iterator (gather across all dims)

按参数走；每个张量跨 **TP、PP、EP** gather；吐出完整 HuggingFace 张量。两处亏：

1. **成千上万的小 collective。** MoE checkpoint 给每个 expert 起名。Qwen3-235B：**94 层 × 128 expert × 若干 projection**，大约 **37,000** 个张量，多半很小。一个一个 gather，开销很大。
2. **每个 rank 都 gather 全部。** 完整张量在每个 trainer rank 上重建：显存重复。

端到端：**25.02 s**。

### V2 — PP-local, EP-local

- **PP-local gather。** 一层的 all-gather 只在同一 pipeline stage 的 rank 之间做。
- **EP-local transfer。** Expert **根本不 gather**。Trainer rank 声明谁持有哪个 expert；推理 rank 去那些 rank 上 pull。

Kimi K2 尤其吃这一套：BF16 下 **一整层 MoE 约 30 GB**。同步时每卡再申请这么大，很容易 OOM。

**25.02 s → 5.61 s。** 元数据缓存一类还有一点小优化；细节在 [SkyRL 例子](https://github.com/NovaSky-AI/SkyRL/tree/main/examples/train/megatron/sharded_rdt)。

### V3 — pipelined execution

V2 里 all-gather、replay、transfer 仍 **串行**。这三步用的资源并不一样。

- **Trainer：按 weight group gather。** 一个 decoder block 是 gather / transfer / 释放的单位。
- **Trainer：gather 和 pull 重叠。** 推理还在拉第 N 组时，已经 gather 第 N+1 组。
- **Trainer：replay 和 transfer 重叠。** 一块的 RDMA 还在落地，下一块已经在 pack、replay。推理侧也可以一边收下一块，一边把当前 RDT block 拷进 layerwise buffer。
- **推理：后台 process。** RDT → layerwise buffer 之后，把 process + copy（第 6–7 步）排到后台，RDT buffer 好去接下层。

![rdt pipelined execution 2x](../../../../assets/vllm/blog/serving/rdt-weight-transfer/06-rdt_pipelined_execution-2x.png)

Trainer 上同时留着多层 all-gather 的结果：抽取、NIXL 传送、推理后处理可以 pipeline。EP/PP-local 抽取把额外显存压下去，这才养得起。

**5.61 s → 3.49 s。**

![rdt qwen weight sync latencies](../../../../assets/vllm/blog/serving/rdt-weight-transfer/07-rdt_qwen_weight_sync_latencies.png)

Qwen3-235B-A22B 端到端同步。4×8×H100，Megatron TP4/PP2/EP8 → vLLM DP16 EP16。

### Final results: Kimi K2 at 48 nodes

NIXL 团队验证：**Kimi K2**，**48 × 8×H100**。

训练：Megatron **TP8 / PP8 / EP32 / ETP1**。推理：vLLM **TP32 / EP32**。

| 指标 | 值 |
| --- | ---: |
| 训练拓扑 | 32 × 8×H100 |
| 推理拓扑 | 16 × 8×H100 |
| 每次同步搬运 | 7.9 TB |
| 权重同步时间 | **7.53 s** |
| 实测聚合带宽 | 1,049 GB/s |

页上的光速估计。绝对 SoL = 权重走上网络的时间。训练占 32 节点；每个推理副本占 **4** 节点。PP **8** 时，每组 4 个节点大约要送 **2 TB / 8 = 0.25 TB** 给 **4** 个副本 → 这 4 个节点大约送出 **1 TB**。每个 4 节点的推理副本要收 **2 TB**。盯一个副本：

- 字节 = **2 TB**
- 聚合带宽：**400 × 4 GB/s = 1600 GB/s**（InfiniBand）
- 绝对 SoL ≈ **1.25 s**

他们现在必须 **按 trainer PP 串行传**：layerwise reloading 每层在 GPU 上另开 buffer，PP 并行往副本灌很容易 OOM。更老实的期望 SoL 因此改看发送侧：**每 PP group 约 0.625 s × PP 8 = 5 s**。测到 **7.53 s**，大约是这期望 SoL 的 **1.5×** 以内。

## Fault tolerance for rollouts

NIXL 相对 broadcast collective 的卖点：一个 rank 挂了，不必把整组掀翻，也不必重建 communicator。

SkyRL 演示：推理引擎挂了，run 降级接着走——router 把流量送到还活着的引擎；下一次 sync，trainer 只跟 **活着的** 引擎说话。副本回来，在 **下一次权重同步边界** 重新入队，拿到新权重，再 serving。

![rdt fault tolerance](../../../../assets/vllm/blog/serving/rdt-weight-transfer/08-rdt_fault_tolerance.png)

Qwen3-32B，Text2SQL，**4 × 8×H100**，**4** 个推理副本。第 **20**、**40** step 杀掉一台引擎，过几步再拉起来。RDT+NIXL 训练照常；页上写收敛不受影响。

Broadcast 集体做不到这一点。这是和 [native-rl](native-rl.md) / [openrlhf](openrlhf.md) 里 NCCL 路径的运维差别。

## Integration with SkyRL

覆盖项：

```shell
generator.inference_engine.weight_sync_backend=sharded_rdt \
trainer.placement.colocate_all=false
```

别家框架在训练侧实现 `WeightSource` iterator：

```python
class WeightSource(ABC):
    def metadata(self) -> list[ParamMeta]: ...        # names, dtypes, full shapes — no transfer
    def __iter__(self): ...                           # yield (name, materialized tensor)

    # Optional, for sharded trainers — declare what THIS rank holds:
    def held_names(self) -> "Collection[str] | None": ... # which params are yielded?
```

`held_names` 才打开 V2 的 PP-local / EP-local。

## Limitations and what's next

还早。原文的坑：

- Loader 必须停在 **可记录** 的操作里。加载时去 **看真实数值** 的 loader，初始化就会失败。
- RDT 目的 buffer **不计入** `gpu_memory_utilization`，要先定 buffer 大小，再选这个比例。
- 当时 **不能和 EPLB** 一起用。
- 传送按 **trainer PP 串行**，免得 layerwise reload OOM。原文提到：让不同 PP group 对着 **不同副本** 并行，或许能解开。
- 当时只有 GPU → GPU 的 RDT。远端 GPU → CPU 已经进 [Ray PR 64815](https://github.com/ray-project/ray/pull/64815)。CPU staging 可以免掉推理 rank 上额外的 RDT GPU buffer，也不必再为了「每副本不另开 GPU buffer」而 **跨副本同步同一个 worker 的 pull**。

## Acknowledgements

和 NIXL 团队合作：Kimi K2 大规模验证，以及一串把传送往上推的建议。Josh Lee、Stephanie Wang 指导 RDT。vLLM 团队，尤其 Ao Shen，帮忙评审。
