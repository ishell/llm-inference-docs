---
source: https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# Elastic Expert Parallelism：MoE 集群不必为了加减卡而重启

英文对照：[en/vllm/blog/serving/elastic-ep.md](../../../../en/vllm/blog/serving/elastic-ep.md)  
原文：https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism  
2026-05-14。作者 **Itay Alroy (NVIDIA), Yongji Wu (Sky Computing), Rui Qiao (Anyscale), Tyler Michael Smith (Red Hat), Moein Khazraee (NVIDIA), Omri Kahalon (NVIDIA), Tzu-Ling Kan (NVIDIA), Ron Tourgeman (NVIDIA)**。[RFC #20323](https://github.com/vllm-project/vllm/issues/20323)，落地 [PR #34861](https://github.com/vllm-project/vllm/pull/34861)；NIXL EP 是 [PR #35627](https://github.com/vllm-project/vllm/pull/35627)。容错方向：[RFC #30112](https://github.com/vllm-project/vllm/issues/30112)。DP Attention + EP 背景：[RFC #16037](https://github.com/vllm-project/vllm/issues/16037)。当时实现范围很窄，旗标以原文为准。Wide-EP 主菜在 [large-scale.md](large-scale.md)。

Expert parallelism（EP）是高吞吐伺候 MoE 的关键刀。WideEP（EP 铺过许多 worker）把 KV 容量做大，高并发或很长上下文才站得住。RL 要又长又快；agent 的多轮对话会把上下文越拉越长。

在 vLLM 里，和许多推理框架一样，EP 一直是**静态**的：启动时几张卡，就几张卡。流量涨了加不进去，闲了减不下来。唯一的办法是换配置**整台重启**——慢，而且会丢掉一大截正在飞的请求。

**Elastic Expert Parallelism**（Elastic EP）改的是这件事：运行时改 worker 个数，MoE 部署可以随需求放大缩小，serving 尽量不断。

它加的、减的是 **data-parallel (DP) worker**。在 vLLM 里，这会改变共享 expert-parallel (EP) 组的大小，以及专家怎么分家——见下文 Background。一次 API 调用：

```bash
curl -X POST http://localhost:8000/scale_elastic_ep \
  -H "Content-Type: application/json" \
  -d '{"new_data_parallel_size": 8}'
```

当前部署会被改到 8 个 DP worker。不必重启 server。

![elastic ep](../../../../assets/vllm/blog/serving/elastic-ep/01-elastic-ep.png)

**图注（原文）。** Elastic EP scale-up：一张新 GPU 加入正在干活的部署，EP 组变宽，已有 worker 不必重启。

这篇写：scale-up / scale-down 流程；重配怎样和正在飞的 forward 和平共处；和 EPLB、EP 通信后端怎么配合；为什么这是容错方向的一块砖。也写 NIXL EP：它的通信模型特别适合弹性重配和容错。

> **给运营的 TL;DR：**
> - Elastic EP 让 vLLM 在运行时改 DP 大小，MoE 部署放大缩小都不必重启 server。
> - 触发：`POST /scale_elastic_ep`；vLLM 改活拓扑，必要时重分专家。
> - 这条运行时重配路径，是容错 serving 的核心积木。
> - NIXL EP 可以明显减少 scale 时的重初始化，并在 EP 侧做失败探测、报告、恢复。

## Background: Expert Parallelism and DP Attention

MoE 里 attention 仍是密的，FFN 大多换成稀疏专家：每个 token 只去被选中的那几位专家家里。谈弹性之前，先把 Elastic EP 踩着的两把刀讲清。

**Data Parallel (DP) Attention** 是请求级并行：每个 engine-core 管自己那一份请求，自己的 KV，自己的 scheduler。MLA 一类架构上尤其重要——纯 TP 会把 KV 在每张卡上复制一份，房子立刻变窄，batch 上不去。

**Expert Parallelism (EP)** 用在专家层。不是把每一位专家切碎摊到多张 GPU，而是整颗整颗分到不同 GPU；token 只被 dispatch 到拥有被选中专家的那些卡。

在 vLLM 里，attention 在每个 DP worker 上各过各的；专家层共用一个 EP 组，组大小是 `DP × TP`。Elastic EP 运行时改 DP 个数，EP 组跟着变，专家要重新分家。

## The Challenge: What State Needs to Change?

运行时改 DP，不是拉起或杀掉几个进程就完了。EP 大小一变，作废的状态有一串：

- **Distributed communication groups。** EP、DP、world 组都把 rank 集合写死了。
- **Expert assignment。** 专家到 rank 的映射随 EP 大小一起变。
- **Model weights。** 新 rank 要权重；老 rank 在重分家之后也可能要更新专家权重。
- **CUDA graphs and compiled state。** CUDA graph capture 和 `torch.compile` 都按旧拓扑特化过。

所以实现把 scaling 当成一台有同步点的状态机。这些同步点还得和正在飞的 model forward 和平共处。

## Scale-Up Flow

从 `DP=N` 到 `DP=M`（`M > N`）比往下缩更复杂：新 rank 要走进一套还在干活的部署。

### 1. Trigger and Request Handling

从 `/scale_elastic_ep` 开始。若设了 `VLLM_ELASTIC_EP_DRAIN_REQUESTS=1`，先等在飞的请求排空，最多 `drain_timeout` 秒（默认 **120**）。否则立刻开始。

### 2. New Engine Core Initialization

新 engine-core 依赖 **Ray DP backend**。Scale-up 时，Ray DP backend 在现有空闲 GPU 上拉起目标 DP 大小还缺的那些 DP worker。新 rank 拿到当前专家映射，用**占位权重**初始化模型，然后等后面的传输和重配，把自己接进现行拓扑。

就绪分两拍：先发一个信号，让老 rank 建 standby 组；再发一个信号，才开始传权重。

### 3. Standby Communication Groups

关键设计：vLLM **不立刻拆掉**正在干活的通信组。老 rank 先用 `StatelessGroupCoordinator` 建一套覆盖**目标** rank 集合的 **standby groups**——独立于 PyTorch 全局 `WORLD`。

这样可以在切换之前把新配置准备好，旧配置在此期间仍可继续 forward。

`nixl_ep` 可以把这件事做成增量：不必拆掉再重建所有 EP 侧连接，用 NIXL EP 的 `connect_ranks()` / `disconnect_ranks()` 加减 rank，已有连接不动。

### 4. Expert Mapping and Weight Transfer

Standby 组建好之后，用它们广播当前专家映射，并把**非专家**权重从老 rank 送到新 rank，尽量在老 rank 之间摊匀。Elastic EP 复用 EPLB 搬专家权重的那条 GPU-to-GPU send/recv，但伸到 attention、norm、embedding 和其他非专家权重；节点内走 NVLink，跨节点走 RDMA。

**专家权重这一步还不搬。** 留给拓扑切过去之后的 EPLB reshuffle。过渡期间普通 EPLB 暂停，免得和重配打架。

### 5. The Switch

切换点：所有 rank 同时离开旧拓扑、进入新拓扑。这一拍 vLLM：

1. 释放 CUDA graph，重置 `torch.compile` 状态。
2. 把 standby 组升成现行 EP、DP、world 组。
3. 拆掉旧组。
4. 按新 EP 大小重配 MoE 模块。
5. 重新 warmup，让 CUDA graph 和编译路径贴上新拓扑。

Engine 协调状态（running flag、wave counter、step counter）在新 DP 组上对齐，每个 rank 从同一点恢复。

此时新 rank 已经进了现行 DP 组，可以参与 forward、跑 attention，但**还没有专家**。专家所有权在接下来的 EPLB reshuffle 里更新。

### 6. EPLB Reshuffle

新拓扑生效之后，EPLB 把专家在全部 `M` 个 rank 上重铺：更新映射，搬走需要搬家的专家权重。Reshuffle 结束，日常 EPLB 恢复。

## Scale-Down Flow

从 `DP=M` 收到 `DP=N`，总节奏和 scale-up 一样，但有一处不能反：**EPLB reshuffle 必须先做**。马上要离开的 rank 可能还握着专家权重。必须先让全部 `M` 个 engine-core 做一次 reshuffle，把专家和需要搬走的权重收到留下的 `N` 个 rank 上，再拆人。

## Coordinating Reconfiguration Steps Across DP Ranks

DP engine-core 是异步的，收到重配通知可能差半拍。有人已经走到 Elastic EP 下一阶段，有人可能又迈进了一次 forward。若早到的 rank 立刻往下走，组会裂成「重配」和「还在 forward」两半——死锁。

Elastic EP 用**两段屏障**。第一段带超时：没到齐，就推断同伴还在跑，自己也回到 engine 循环再走一步，而不是一个人往前冲。下一轮大家站在同一条边界上，第二段屏障不再走超时路径，一起进下一阶段。

## Path to Fault Tolerance

Elastic EP 是容错的核心积木，因为它给出了失败之后需要的运行时重配路径。某个 rank 挂了：先 scale-down 摘掉死人、重分专家，有替补卡再 scale-up，不必把整台部署重启。这是 [RFC #30112](https://github.com/vllm-project/vllm/issues/30112) 那条故障容忍方向的一块地基。

高层恢复流：

1. **Detect** — 健康检查，或后端自己的失败信号。
2. **Scale down** — 摘掉失败的 rank，重分它的专家。
3. **Scale up** — 替补容量到位之后再加回来。

NIXL EP 在这里也相关：它可以在 EP 侧探测、报告、恢复失败，并在容量回来时把替补 rank 接上。

## Next Steps

核心运行时重配路径已经在，当时实现范围仍窄，后续很清楚：

- **Support richer parallel configurations。** 包括 `tensor_parallel_size > 1` 以及更丰富的并行组合。
- **Support more serving features。** 当时 `api_server_count` 上限 **1**；**还不支持 DBO**，也不支持 MoE draft / drafter。
- **Reduce the reconfiguration window。** overlap、warmup 代价、CUDA graph 重捕获、复用已经准备好的状态。
- **Connect Elastic EP to autoscaling policies。** 控制面在这篇里；策略和编排是另一件事（Dynamo、llm-d）。
- **Support additional DP backends。** Scale 操作当时依赖 Ray DP backend。

## Getting Started

### Launch with Elastic EP Enabled

小 MoE 例子：`DeepSeek-V2-Lite-Chat`。当时目标：Ray DP，`tensor_parallel_size=1`，一台 API server，不开 DBO。

```bash
vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --data-parallel-size 2 \
    --data-parallel-backend ray \
    --api-server-count 1 \
    --enable-expert-parallel \
    --enable-elastic-ep \
    --enable-eplb \
    --eplb-config.num_redundant_experts 0 \
    --all2all-backend allgather_reducescatter \
    --gpu-memory-utilization 0.8
```

### Scale Up at Runtime

Ray DP backend 下，加容量可以就是再把一个节点加入 Ray 集群；Ray 看见新 GPU，Elastic EP 就能在运行时把部署铺上去。

例如在新 worker 节点上：

```bash
ray start --address="${HEAD_NODE_IP}:6379"
```

```bash
curl -X POST http://localhost:8000/scale_elastic_ep \
  -H "Content-Type: application/json" \
  -d '{"new_data_parallel_size": 16}'
```

### Scale Down

```bash
curl -X POST http://localhost:8000/scale_elastic_ep \
  -H "Content-Type: application/json" \
  -d '{"new_data_parallel_size": 8}'
```

### Using NIXL EP as the Communication Backend

Elastic EP 配 NIXL EP：

```bash
uv pip install nixl

vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --data-parallel-size 2 \
    --data-parallel-backend ray \
    --api-server-count 1 \
    --enable-expert-parallel \
    --enable-elastic-ep \
    --enable-eplb \
    --all2all-backend nixl_ep
```

安装与传输配置见 [NIXL 仓库](https://github.com/ai-dynamo/nixl)。

## References

- [RFC #20323: Elastic Expert Parallelism](https://github.com/vllm-project/vllm/issues/20323)
- [PR #34861: [1/N] Elastic EP Milestone 2](https://github.com/vllm-project/vllm/pull/34861)
- [PR #35627: [2/N] Elastic EP Milestone 2: Integrating NIXL-EP](https://github.com/vllm-project/vllm/pull/35627)
- [RFC #30112: Fault-Tolerant Expert Parallelism](https://github.com/vllm-project/vllm/issues/30112)
- [RFC #16037: Data Parallel Attention and Expert Parallel MoEs](https://github.com/vllm-project/vllm/issues/16037)

## Acknowledgments

感谢把 Elastic EP 送进 vLLM 的人。

- Sky Computing: Yongji Wu
- NVIDIA: Itay Alroy, Moein Khazraee, Omri Kahalon, Tzu-Ling Kan, Ron Tourgeman
- Red Hat: Tyler Michael Smith
- Anyscale: Rui Qiao
- 更广的 vLLM 社区

必读 serving 线在 Wide-EP 之后接到这里：先学会把专家铺开，再学会**不停机地改变铺开的宽度**。Agent 与 RL 的流量不是一条平线，卡也不该永远按高峰来买。
