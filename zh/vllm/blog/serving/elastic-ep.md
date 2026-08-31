---
source: https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# Elastic Expert Parallelism：MoE 集群不必为了加减卡而重启

英文对照：`en/vllm/blog/serving/elastic-ep.md`  
原文：https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism  
2026-05-14。RFC #20323，落地 PR #34861；NIXL EP 是 PR #35627。容错方向见 RFC #30112。图在原网页。当时实现范围很窄，旗标以原文为准。

[大规模 serving](large-scale.md) 把 Wide-EP 写成了主菜：专家铺开，KV 的房子变大，并发和长上下文才站得住。RL 要又长又快，agent 会把对话越拉越长。可 EP 一直是**静态**的——启动时几张卡，就几张卡。流量涨了加不进去，闲了减不下来。唯一的办法是换配置重启，慢，而且会丢掉一大截正在飞的请求。

Elastic EP 改的是这件事：运行时改 **data-parallel worker 的个数**。在 vLLM 里，attention 在每个 DP worker 上各过各的（自己的 KV、自己的 scheduler）；专家层共用一个 EP 组，组大小是 `DP × TP`。DP 变了，EP 组跟着变，专家要重新分家。

一次 API 调用：

```bash
curl -X POST http://localhost:8000/scale_elastic_ep \
  -H "Content-Type: application/json" \
  -d '{"new_data_parallel_size": 8}'
```

当前部署会被改到 8 个 DP worker。不必重启 server。

## 背景：两把刀为什么要一起用

MoE 里 attention 仍是密的，FFN 大多换成稀疏专家。token 只去被选中的那几位专家家里。

- **DP Attention：** 请求级并行。每个 engine-core 管自己那一份请求、KV、调度。MLA 一类架构上，纯 TP 会把 KV 在每张卡上复制一份，房子立刻变窄。
- **Expert Parallelism：** 专家整颗整颗分到不同 GPU，token 只被 dispatch 到拥有它的那些卡。

Elastic EP 动的是 DP 个数，从而动 EP 组。

## 要换的不只是进程名单

DP 从 N 改到 M，作废的状态有一串：

- 通信组（EP / DP / world）里写死了 rank 集合
- 专家到 rank 的映射
- 新 rank 要权重；老 rank 在重分家之后也可能要更新专家权重
- CUDA graph 和 `torch.compile` 都按旧拓扑特化过

所以这是一台有同步点的状态机，还得和正在飞的 forward 和平共处。

## Scale-up：`DP=N` → `DP=M`（M > N）

1. **触发。** `POST /scale_elastic_ep`。若设了 `VLLM_ELASTIC_EP_DRAIN_REQUESTS=1`，先等在飞的请求排空，超时默认 **120 s**（`drain_timeout`）；否则立刻开始。
2. **新 engine-core。** 依赖 **Ray DP backend**。Ray 在现有空闲 GPU 上拉起新的 DP worker。新 rank 拿到当前专家映射，用占位权重初始化模型，等后面的传输。就绪分两拍：先让老 rank 建 standby 组，再开始传权重。
3. **Standby 通信组。** 不立刻拆掉正在干活的组。老 rank 先用 `StatelessGroupCoordinator` 建一套覆盖目标 rank 集合的待命组——独立于 PyTorch 全局 `WORLD`。旧拓扑仍可继续 forward。`nixl_ep` 可以把这件事做成增量：`connect_ranks()` / `disconnect_ranks()`，已有连接不动。
4. **映射与非专家权重。** 走 standby 组广播专家映射，把 attention、norm、embedding 等非专家权重从老 rank 尽量均匀地送到新 rank。路径复用 EPLB 的 GPU-to-GPU send/recv（节点内 NVLink，跨节点 RDMA）。**专家权重这一步还不搬**——留给拓扑切过去之后的 EPLB reshuffle。过渡期间普通 EPLB 暂停。
5. **切换。** 所有 rank 同一时刻离开旧拓扑：释放 CUDA graph、重置 `torch.compile`；把 standby 组升成现行 EP/DP/world；拆掉旧组；按新 EP 大小重配 MoE 模块；重新 warmup。engine 协调状态（running flag、wave / step counter）在新 DP 组上对齐。此时新 rank 已经能跑 attention，但**还没有专家**。
6. **EPLB reshuffle。** 专家在全部 M 个 rank 上重铺，权重跟着搬家，然后恢复日常 EPLB。

## Scale-down：先搬家，再请人走

从 `DP=M` 收到 `DP=N`，顺序有一处不能反：马上要离开的 rank 可能还握着专家。必须先让全部 M 个 engine-core 做一次 reshuffle，把专家和权重收到留下的 N 个 rank 上，再拆人。

## 两段屏障：别让异步 DP 把自己卡死

DP engine-core 是异步的，收到「该进入下一阶段」的通知可能差半拍。有人已经又迈进了一次 forward，有人停在屏障前——组会裂成两半，死锁。Elastic EP 用**两段屏障**：第一段带超时，没到齐就推断同伴还在跑，自己也回到 engine 循环再走一步；下一轮大家站在同一条边界上，第二段屏障不再走超时路径，一起进下一阶段。

## 通往容错的那块砖

rank 挂了以后，需要的就是同一条运行时改拓扑的路：先 scale-down 摘掉死人、重分专家，有替补卡再 scale-up。这是 RFC #30112 那条故障容忍方向的一块地基。NIXL EP 还能在 EP 侧做失败探测、报告、恢复，以及把替补 rank 接回来。

## 当时还不能做什么

- `tensor_parallel_size > 1` 以及更丰富的并行组合还没有
- `api_server_count` 上限 1；**还不支持 DBO**，也不支持 MoE draft / drafter
- 重配窗口（warmup、CUDA graph 重捕获）仍要收
- autoscaling **策略**不在这篇里——控制面在，策略交给 Dynamo / llm-d
- scale 操作依赖 Ray DP backend

## 启动（小 MoE 例子）

当时目标：Ray DP，`tensor_parallel_size=1`，一台 API server，不开 DBO。

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

新节点先 `ray start --address="${HEAD_NODE_IP}:6379"`，再 `POST /scale_elastic_ep` 把 `new_data_parallel_size` 改成 16 或改回 8。NIXL 路径把 `--all2all-backend` 换成 `nixl_ep`（先 `uv pip install nixl`）。

必读 serving 线在 Wide-EP 之后接到这里：先学会把专家铺开，再学会**不停机地改变铺开的宽度**。Agent 与 RL 的流量不是一条平线，卡也不该永远按高峰来买。
