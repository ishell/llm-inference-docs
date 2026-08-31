---
source: https://vllm.ai/blog/2026-05-06-mooncake-store
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# vLLM × Mooncake：agent 的前缀，不该每回合重读一遍

英文对照：`en/vllm/blog/serving/mooncake.md`  
原文：https://vllm.ai/blog/2026-05-06-mooncake-store  
2026-05-06。Mooncake 仓库与 KVConnector 实现见原文链接。图在原网页。数字是 Codex / SWE-bench Pro 轨迹上的演示。

Agent 来了以后，推理负载换了一种脾气。不再是一问一答的短会话，而是长地平线上的循环：想一想，调用工具，把工具吐回来的东西接进上下文，再想。Jensen 在 GTC 2026 里说的「从聊天机器人走向会自己做事的系统」，落到 serving 上就是一件很具体的事——**同一段前缀被反复看见**。

## 一张 agent 轨迹的解剖

他们收集了 Codex 与 GPT-5.4 在 SWE-bench Pro 上的轨迹，并开源了数据集。610 条，中位约 **33** 回合。到第 30 回合，上下文中位大约 **80K** token，最长可以超过 **180K**。可每一回合真正新进来的，常常只有几百到几千 token。其余全是已经读过的：系统提示、skills / memory、先前各轮。

整份数据上：

- 若能缓存前缀，命中率大约 **94.2%**
- 平均 ISL : OSL 大约 **131 : 1**
- 每回合上下文大约再长 **2,242** token
- 中位上下文从约 12K 长到约 80K
- 回合之间的等待：中位 **5.2 s**，P99 **81.4 s**

缓存住前缀，prefill 里那一大截几乎免费。每回合真正该付钱的，只是 delta。

## 为什么「本机卸到 CPU」不够

vLLM 早就能把 KV 卸到本机 DRAM 或盘上。对 agent 有两道墙：

1. **容量与驱逐。** 100K token 的上下文可以占掉数 GB（文中例子：Kimi-2.5 的 FP8 KV 大约 **3.8 GB**）。一台正忙着伺候许多长会话的实例，前缀很快把本地池子撑满，然后被赶走。
2. **跨实例 miss。** 路由器为了摊负载，下一回合未必还落在同一台 vLLM 上。新实例从没见过这段前缀，只好从头算。

所以不能再把推理服务当成一排互不相识的 replica。Agent 需要一块**集群级的 KV 池**：容量是大家的，命中也可以跨实例。

## Mooncake Store 怎么坐进 vLLM

Mooncake 本来就是开源的 KV 传输与分布式存储库。vLLM 已经用它的 transfer engine 做 Prefill/Decode 分离（`MooncakeConnector`）。这一步把 **Mooncake Store** 做成分布式 KV 池。

集群里：一台 master 管 KV block 的哈希、大小、发现、死人清理；GPU 节点上的 client 管本地 DRAM / SSD，彼此用 RDMA 传块。多台 vLLM 各自嵌一个 client，共用这一池。

接入走现成的 **KVConnector**——和 P/D 分离是同一扇门：

- **Scheduler 侧：** 新请求来了，把 prompt 按 block 哈希，去问 Mooncake master 有没有现成的块，用结果帮调度做决定。
- **Worker 侧：** 每个 GPU worker 里嵌 client，后台线程搬数据。GPU 上的 KV 登记成 RDMA buffer，**GPUDirect RDMA** 直接在 HBM 和远端 DRAM 之间读写，不走 SM，也不在 CPU 上再垫一层。

## 设计上三件不肯省的事

**不占 SM、零拷贝。** `cudaMemcpyAsync` 对大量小块不一定痛快；专门起拷贝 kernel 又会跟正在跑的 attention 抢 SM。第三条路：NIC + GPUDirect RDMA。Mooncake Transfer Engine 还能把节点上多块 RNIC 池在一起、按拓扑选路。

**传输对主路径完全异步。** RDMA 本身异步，可准备 descriptor、发出读写仍要吃 CPU；序列越长，块越多。这些活全部丢给专用 I/O 线程，免得主 CPU 路径卡住、推迟 GPU kernel launch。

**P/D 分离可以和池子叠在一起。** `MultiConnector` 把若干子 connector 串起来，彼此不依赖。Prefill 实例既给 P/D connector 准备块，也把块写入分布式池；命中时可以向所有 connector 询问，从 Store 把前缀捞回来。Decode 写入池子立刻对 prefill 可见。当时 decode **还不从池子读**——vLLM 会把每个请求同时排到一台 prefill 和一台 decode，前缀由 prefill 从池子加载，再经 P/D connector 转给 decode。他们还在做从 prefill 与池子双路径同时加载，好把带宽吃满。

## 成绩（演示）

Kimi-2.5 NVFP4，GB200，P/D 分离：prefill **TP4**，decode **DP8 + EP**。他们认为这是当时延迟–吞吐最好的折中。

**真实 Codex 轨迹，1P1D，一共 12 张 GB200：** 相对基线，吞吐大约 **3.8×**，P50 TTFT 大约 **46×** 更低，端到端大约 **8.6×** 更低。驱动数字是命中率：从大约 **1.7%**（几乎只缓存了系统提示）到大约 **92.2%**（前缀几乎都在）。

**扩到多机。** 合成负载贴着 Codex 的形状：2 万公共 token、首轮 1 万、之后每回合输入 2048、输出 900、共 30 回合；会话数随 GPU 从 75 涨到 375。为了故意制造跨节点流量，路由用 **round-robin**——下一回合经常落在另一台机器上。没有分布式池，这就是大规模 miss。有 Mooncake Store：12→60 张 GB200，命中率始终 **>95%**，吞吐接近线性。

## 下一步（文中当时）

分布式盘卸载（NVMe、分布式文件系统）；混合注意力模型的分层缓存策略；**cache-aware routing**（先送到已经握着前缀的实例，池子当退路）；NVLink 多节点 + RDMA 的多路径传输。实现受 vLLM-Ascend 启发。

读完 [Router](router.md) 和 [大规模 serving](large-scale.md) 再读这一篇：认得 KV 的路由器解决「下一句话去哪」；分布式 KV 池解决「去了另一台也不必重读」。Agent 把 ISL/OSL 拉到 131:1 以后，这两件事是同一句话的上下半句。
