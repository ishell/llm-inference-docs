---
source: https://vllm.ai/blog/2026-05-18-pegaflow
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# PegaFlow：让 KV 活得比推理进程更长

英文对照：[en/vllm/blog/serving/pegaflow.md](../../../../en/vllm/blog/serving/pegaflow.md)  
原文：https://vllm.ai/blog/2026-05-18-pegaflow  
2026-05-18。署名 **Novita AI and the vLLM Team**。仓库：[novitalabs/pegaflow](https://github.com/novitalabs/pegaflow)。Rust 守护进程 + 外部 KV connector，**不改 vLLM 源码**，**不必养长命 fork**。文中例子用 `vllm>=0.20.0`。和 [Mooncake](mooncake.md)、[KV offload](kv-offload.md) 同一扇门：池子活得比引擎久。页上是偏生产的评估，不是你的 SLA。

**原文 TL;DR：**

- 大约 **500 GiB** 的主机 KV 池已经由外部 cache 服务持有时，vLLM 启动快 **2.15×**。
- 八个 Qwen3-8B 共享一份主机 cache，而不是八个隔离池：吞吐 **+56%**。
- DeepSeek-V3.2 MLA TP8：逻辑 KV **存一份** 而不是每 TP rank 一份，吞吐 **+72%**。
- 内部 RDMA 集群、每节点 **8 × 400 Gbps**：大前缀远端拉取平均 **194 GB/s**。

核心判断：KV cache 该是一份 **长命的 serving 资产**，不是绑在某一个推理进程上的临时状态。对外走现成的 `kv_transfer_config`。

本地图（原文版权仍归原站；学习对照用）：

![architecture](../../../../assets/vllm/blog/serving/pegaflow/01-architecture.png)

![startup time](../../../../assets/vllm/blog/serving/pegaflow/02-startup-time.svg)

![tail latency](../../../../assets/vllm/blog/serving/pegaflow/03-tail-latency.png)

![results overview](../../../../assets/vllm/blog/serving/pegaflow/04-results-overview.svg)

![rdma throughput](../../../../assets/vllm/blog/serving/pegaflow/05-rdma-throughput.svg)

![cache policy comparison](../../../../assets/vllm/blog/serving/pegaflow/06-cache-policy-comparison.png)

## 为什么 KV 需要一道进程边界

生产 serving 里，KV 是最贵的运行时资产之一。每机可以占掉 **几百 GiB**，分配和预热很慢，却常常比当时那波请求 **活得更久**。

绑在进程里：崩溃、滚动升级、换模型，池子一起死。引擎一重启，主机 KV 池就没了。机群换模型，可能要重新申请、预热几百 GiB pinned 内存，才能再接流量。

PegaFlow 把 KV 运行时搬到 **每机一台独立 daemon**。Server 管：主机 KV 池、SSD cache、拓扑元数据、RDMA 资源、索引、后台任务。vLLM worker 用 **CUDA IPC**（数据）和 **gRPC**（本机控制）连上去。

**Figure 1。** PegaFlow 坐在 vLLM 旁边。本机 CUDA IPC + gRPC；PegaFlow 管 pinned 内存、SSD、RDMA，以及可选的跨节点索引（经 **MetaServer**）。

页上的生产要求：**一台 cache 伺候同一宿主机上的多引擎、多模型**。不同模型、TP 布局、引擎版本用 **namespace** 隔离，共享同一份内存池、SSD、跨节点带宽。

故障域切开：vLLM 可以崩、可以升级、可以换模型，cache 还在。Cache 层的问题也不必把推理进程一起带走。

## 外部持有池子，重启更快

只测启动路径：**8 × RTX 5090**，**Qwen3-8B TP8**，**dummy 权重**，**eager**——拿掉加载权重和编译，只看大约 **500 GiB** 的主机 KV 池。

- 内嵌 / 进程内：vLLM **71.4 s** 才 ready。
- 池子预先进 PegaFlow：server ready 之后 vLLM **33.2 s** → vLLM 启动快 **2.15×**，因为长命的主机分配不再跟推理进程绑死。

**Figure 2。** 就是这两根柱。

## Rust 数据面和尾延迟

把进程拆出去，首先是为了生命周期、共享、CPU 隔离。用 **Rust** 实现，顺手买到延迟稳定性。

数据面躲开 Python 解释器、**GIL** 争用、停顿式 GC。生产 cache 不只在关键路径搬字节：统计、索引上传、预取、健康检查、指标、驱逐、SSD 管理。这些任务住在同一份独立的 Rust 服务里，**不和 vLLM 共用一个解释器**。

**Figure 3。** Baseline 和 GIL-load 下的尾延迟 / 平均延迟。**Rust Tokio** 被后台负载带偏的程度，远小于 **Python uvloop** 和 **Python ZMQ**。

## 把 cache 在实例之间、节点之间攒成池

同一份逻辑 KV 常常被存很多遍，因为进程、模型、节点把 cache 彼此挡住。原文点名的几种：

- **同一宿主机上多个小模型实例。** 八卡上八个 Qwen3-8B，可以把 **同一条 system prompt 存八次**。
- **宽专家并行。** 同一机器上多个 DP replica 各养一份 prefix cache。
- **MLA + tensor parallelism。** DeepSeek-V3.2：逻辑上的 latent KV 本可存一份；进程内 TP8 可能 **每个 rank 存一份**。
- **跨节点调度。** Node A 命中，可 A 过载，请求被派到 Node B → prefix 从头再算。

PegaFlow 把这些碎片收成共享池。

单机上，本地实例都连同一台 PegaFlow，共享一份 CPU KV 池。相同 block 可以 **物理上只存一次**，给多个引擎复用（小模型多实例、WideEP 的 DP replica、TP worker）。

跨机时，**PegaFlow MetaServer** 维持一份 **近似全局索引**。节点用 **one-sided RDMA READ** 拉远端 KV；连接建好之后远端 **不再占 CPU**。远端命中可以更像本地命中，躲开昂贵的 Prefill 重算。

## 成绩

Cache **预算** 固定；变的只是这份 cache 在进程 / rank / 节点之间 **看得见看不见**。

### 单机多实例共享

八个 Qwen3-8B，一机，**500 GiB** 预算。

| 方案 | Cache 布局 | 吞吐 | Mean TTFT | 请求命中率 |
| --- | --- | ---: | ---: | ---: |
| PegaFlow | 500 GiB 共享池 | 11.97 req/s | 5.26 s | 52.35% |
| 进程内 | 8 × 62.5 GiB 隔离池 | 7.68 req/s | 8.22 s | 11.77% |

不是多用了内存——还是 500 GiB，只是一份池而不是八座岛。吞吐 **+56%**，Mean TTFT **−36%**，请求命中大约 **4.4×**。

### MLA 逻辑 KV 去重

DeepSeek-V3.2 MLA，TP8，**500 GiB** 预算。

| 方案 | Cache 布局 | 吞吐 | Mean TTFT | 请求命中率 |
| --- | --- | ---: | ---: | ---: |
| PegaFlow | 逻辑 KV 存一份 | 1.81 req/s | 35.66 s | 97.23% |
| 进程内 | 每个 TP rank 存一份 | 1.05 req/s | 60.88 s | 65.18% |

逻辑 KV 不再按 rank 重复存，等于把 **可用** 容量撑开。吞吐 **+72%**，Mean TTFT **−41%**，命中率贴近这条 trace 的实际上限。

**Figure 4。** 两次固定预算的本地共享实验。有效容量变大，因为同一份 KV 预算穿过了隔离边界。

### 跨节点 RDMA 共享

内部生产集群：每节点 **8 × 400 Gbps** RDMA 网卡。抽样最近几千次在线远端读。前缀拉取 **≥ 1 GiB**：

- 平均有效吞吐 **194 GB/s**
- **P99 250 GB/s**
- 峰值 **261.6 GB/s**

这个速率下，**24 GiB** 的 KV 段从远端拉过来大约 **100 ms**——换掉本来要吃掉 **数秒** GPU 时间的 Prefill。远端命中不只是「比 miss 好」；它可以快到坐进 serving 路径。

**Figure 5。** 这些大块远端读的有效吞吐；按测到的平均，24 GiB ≈ 100 ms。

## 三层 cache

池化让容量更值钱；主机内存仍然有限。复用距离很长的 prefix 会被赶走；简单 LRU 会被 **扫描型** 流量打乱（大量一次性 block 穿过系统）。

| 层 | 介质 | 访问路径 | 典型角色 |
| --- | --- | --- | --- |
| L1 | 本地 pinned DRAM | 本地内存 | 快的本地 KV 复用 |
| L2 | 远端 DRAM | RDMA READ | 跨节点共享 |
| L3 | 本地 SSD | `io_uring` | 大容量溢出 |

SSD cache 是 Rust，底下 `io_uring`。内部测：单盘峰值读大约 **6.9 GB/s**。在线稳态压在每盘大约 **6.5–6.6 GB/s**——大约让出 **5%** 峰值带宽，换更稳的尾延迟。多盘 **RAID0**：吞吐 **近似线性**。

扫描偏重、或预算更小的机器，可以打开 **TinyLFU** 准入：只让「像会再被用到」的 block 进 cache，免得一次性流量把池子灌满。**默认关**——最好的准入策略跟负载形状走。若干内部 trace 上，cache 小或扫描压力大时，它明显好过 LRU。

**Figure 6。** 小 cache 下的策略对比。扫描型 trace 会让只认 recency 的策略失效；带准入的策略（TinyLFU）能挡住一次性 block。

## 离理论上限还有多远

只看在线命中率会骗人。负载几乎没有复用时，**3%** 可能已经很好；理论上限高很多时，**90%** 仍可能有空档。运维该问的是：**离这份负载合理能达到的最好命中，我们还有多远？**

PegaFlow 用 **HyperLogLog** 在线估上限：

```
r* = (N − U) / N
```

`N` 是窗口里的 block 请求数，`U` 是第一次见到的独特 block。**24 小时** 窗口占用 **< 1 MiB**，误差大约 **0.8%**。

滚动 HLL 窗口，默认：**15 分钟**、**1 小时**、**24 小时**。把实测命中和 `r*` 放在同一块仪表上：

- 已经贴近上限 → 再加容量可能没用。
- 远低于上限 → 容量、准入、预取、跨节点发现还有戏。
- 上限本身很低 → 负载本来就没什么可复用，瓶颈多半不在 cache 实现。

## 走外部 connector 接入

许多外部 KV 系统要动 scheduler、block manager、attention kernel。PegaFlow 走 vLLM 的 **external KV connector**。配 `kv_transfer_config`；用 `kv_connector_module_path` 动态加载包。运行时接管关键 KV 操作；**不改 vLLM 源码**，**不必 fork**。

在 vLLM 看来，PegaFlow 不是另一台推理引擎，只是 KV transfer 接口上的外部 cache 后端。调度、执行、batch、OpenAI 兼容的 serving 路径仍归 vLLM。这道边界让 PegaFlow 可以按自己的钟迭代 Rust 数据面 / SSD / RDMA / 索引 / connector。

## 快速开始

```bash
uv pip install pegaflow-llm        # CUDA 12
uv pip install pegaflow-llm-cu13   # CUDA 13
```

单节点 server，pinned 主机内存 + SSD：

```bash
pegaflow-server \
  --pool-size 30gb \
  --ssd-cache-path <ssd-cache-file-path> \
  --ssd-cache-capacity 512gb
```

在线部署建议加 **`--use-hugepages`**。Huge page 要 **事先预留**。它们加快 CPU pinned 内存分配，并在注册和传输时降低地址翻译开销，减轻 RDMA **MTT** 压力。

多机：先起 **MetaServer**，再在每个节点带 RDMA 起 PegaFlow。开了 P2P 时，每台 server 的 **`--addr` 必须是对端能路由到的 IP**，不能是 `0.0.0.0` 或 `127.0.0.1`——对端拿它做 gRPC 握手和 block 查询。

```bash
pegaflow-metaserver --addr 0.0.0.0:50056
```

```bash
pegaflow-server \
  --addr this-node:50055 \
  --pool-size 30gb \
  --ssd-cache-path <ssd-cache-file-path> \
  --nics mlx5_0 mlx5_1 \
  --metaserver-addr http://metaserver-host:50056
```

接 vLLM（文中例子 `vllm>=0.20.0`）：

```bash
vllm serve <model> \
  --kv-transfer-config '{
    "kv_connector": "PegaKVConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "pegaflow.connector"
  }'
```

`PEGAFLOW_HOST` 和 `PEGAFLOW_PORT` 把 connector 指到服务。默认：`http://127.0.0.1` 和 `50055`。

仓库里还有安装、server 配置、P2P RDMA、指标、connector 例子。

## 公开参考 bench

仓库里的 KV cache bench：**H800**，**Llama-3.1-8B**，**8** 条 prompt，**10K-token** Prefill，**1-token** Decode，**4.0 req/s**。热缓存：Mean TTFT **572.5 ms → 61.5 ms**；P99 TTFT **1113.7 ms → 77.0 ms**。

## 致谢

Novita AI 团队把 PegaFlow 做出来并落到生产。vLLM 维护者和更广的社区提供了讨论、评审，以及让这次接入成为可能的 connector 基础设施。
