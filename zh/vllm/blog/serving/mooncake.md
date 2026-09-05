---
source: https://vllm.ai/blog/2026-05-06-mooncake-store
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# vLLM × Mooncake：agent 的前缀，不该每回合重读一遍

英文对照：[en/vllm/blog/serving/mooncake.md](../../../../en/vllm/blog/serving/mooncake.md)  
原文：https://vllm.ai/blog/2026-05-06-mooncake-store  
2026-05-06。作者 **Yifan Qiao, Trong Dao Le, Ao Shen, Zhewen Li, Bowen Wang**。Mooncake 仓库：[kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)。vLLM 已用 [`MooncakeConnector`](https://docs.vllm.ai/en/stable/features/mooncake_connector_usage/) 做 Prefill/Decode 分离；这一篇把 **Mooncake Store** 做成集群级 KV 池。落地 [PR #40900](https://github.com/vllm-project/vllm/pull/40900)；bench 脚本在原文链到的 artifact 树。数字是 Codex / SWE-bench Pro 轨迹上的演示，不是你的 SLA。

**TL;DR。** Agent 负载会生出巨大的共享前缀，回合之间却常被重算。把 Mooncake 的分布式 KV store 接进 vLLM：真实 agent 轨迹上吞吐约 **3.8×**，TTFT 约 **46×** 更低，端到端约 **8.6×** 更低；扩到 **60** 张 GB200 仍接近线性。

本地图（原文版权仍归原站；学习对照用）：

![hero vllm mooncake](../../../../assets/vllm/blog/serving/mooncake/01-hero_vllm_mooncake.svg)

认得 KV 的路由器见 [router.md](router.md)；本机 CPU 卸 KV 见 [kv-offload.md](kv-offload.md)。这一篇是**跨实例的池子**。

## Agentic workloads are reshaping LLM serving

Claude Code、OpenClaw 这类 agent 起来以后，推理负载换了一种脾气。Jensen 在 GTC 2026 [keynote](https://www.nvidia.com/gtc/keynote/) 里说：LLM 正从聊天机器人走向会自己做事、活得很久的系统。落到 serving 上，结构才是关键。

典型循环是长地平线上的多回合：*reasoning* 一步（读上下文、吐中间想法），再 *action* 一步（发工具调用、接外部输出）。不是一问一答的短会话。

他们收集了 Codex 与 GPT-5.4 在 SWE-bench Pro 上的轨迹，并开源为 [Inferact/codex_swebenchpro_traces](https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces)，方便社区研究 agent serving。

Figure 1 是这份语料的解剖，以及一条代表性会话。

![agentic trace](../../../../assets/vllm/blog/serving/mooncake/02-agentic_trace.svg)

**图注（原文 Figure 1）。** Codex / SWE-bench Pro 轨迹的解剖。每一行是一次 LLM 调用；每回合尺寸用 **610** 条轨迹的中位数。缓存住的前缀（系统提示、skills / memory、先前各轮历史）回合之间复用；每回合真正在动的，只有新的工具输出和模型的 Decode。

到第 30 回合，上下文大约长到 **80K** token，最长可以超过 **180K**。可每一回合真正新进来的，常常只有几百到几千 token。其余全是模型已经见过的前缀。整份数据上，平均 ISL : OSL 大约 **131 : 1**。

若能缓存这些前缀，cached 那一段的 Prefill 几乎免费。每回合真正该付钱的，只是 delta。

610 条轨迹，每条中位约 **33** 回合，他们看到：

- 若能缓存前缀，命中率大约 **94.2%**
- 平均 ISL : OSL 大约 **131 : 1**
- 每回合上下文大约再长 **2,242** token
- 中位上下文从约 **12K** 长到约 **80K**
- 回合之间的等待：中位 **5.2 s**，P99 **81.4 s**

vLLM 早就能把 KV 卸到本机 DRAM 或盘上。对 agent 有两道墙：

- **Limited capacity and eviction。** 100K token 的上下文可以占掉数 GB（文中例子：Kimi-2.5 的 FP8 KV 大约 **3.8 GB**）。一台正忙着伺候许多长会话的实例，这些大前缀很快把本地池子撑满，然后被赶走。
- **Cross-instance misses。** 路由器为了摊负载，下一回合未必还落在同一台 vLLM 上。新实例从没见过这段前缀，只好从头算。

**Takeaway：** 不能再把推理服务当成一排互不相识的 replica。Agent 需要一块**集群级的 KV 池**：容量是大家的，命中也可以跨实例。

## Distributed KV cache pool with Mooncake Store

[Mooncake](https://github.com/kvcache-ai/Mooncake) 是开源的高性能 KV 传输与分布式存储库。vLLM 已经用它的 transfer engine、经 [`MooncakeConnector`](https://docs.vllm.ai/en/stable/features/mooncake_connector_usage/) 做 Prefill/Decode 分离。这一步把 **Mooncake Store** 做成分布式 KV 池。

![overall design option C](../../../../assets/vllm/blog/serving/mooncake/03-overall_design_option_C.svg)

**图注（原文 Figure 2）。** vLLM 分布式 KV 池的总图。多台 vLLM 各自嵌 Mooncake client，共用集群级 Mooncake Store。Mooncake master 管 KV-block 元数据、服务发现、client 健康；worker 用 RDMA 在 GPU HBM 和分布式 DRAM / SSD 池之间搬 KV block。

高层：Mooncake Store 是一台 master 加一群 client。Master 跑在集群范围，管元数据（KV block 哈希、大小等），也盯 client 健康与可用性，做服务发现和死人清理。

Client 跑在 GPU 节点上，管本地 CPU / DRAM / SSD。Client 之间用 RDMA 传 KV。合在一起，就是分布式 KV 池。

接入走现成的 [`KVConnector`](https://github.com/vllm-project/vllm/blob/db9a84e0cd0e17ab693467ff4a71103abd4b77bf/vllm/distributed/kv_transfer/kv_connector/v1/base.py)——和 P/D 分离是同一扇门。Connector 有两个角色：

**Scheduler 侧。** 新请求来了，vLLM 把 prompt 的 token block 做哈希，去问 Mooncake master 有没有匹配的 KV block，用结果帮调度做决定。

**Worker 侧。** 每个 GPU worker 里嵌一个 Mooncake client，后台线程搬数据。GPU 上的 KV 登记成 RDMA buffer，经 Mooncake client 做 **GPUDirect RDMA** 读写：不占 SM，也不在 CPU 上再垫一层。

## Design highlights

### SM-free and zero-copy KV transfer with GPUDirect RDMA

GPU→CPU 传统上两条路。`cudaMemcpyAsync` 走 copy engine，但对大量小块不一定痛快。专门起拷贝 kernel 用 SM 搬，小块多时吞吐可以，却会跟正在跑的 attention 抢 SM。

第三条路：RDMA NIC + GPUDirect RDMA，KV block 直接在 GPU HBM 和 CPU 内存之间走。不要 staging buffer，不吃 SM，大量小块也合适。

Mooncake Transfer Engine 还能把节点上多块 RNIC 池在一起、按拓扑选路，把 KV 传输的带宽在多 NIC 上聚起来。

### Fully asynchronous transfer

RDMA 操作本身是异步的，可准备 descriptor、发出读写仍要吃 CPU。序列越长，KV block 越多，这笔开销越大。

为了不堵住主 CPU 路径（堵住就会推迟 GPU kernel launch），所有 RDMA 丢给专用后台 I/O 线程。从 vLLM 看，传输路径是完全异步的。

### Enabling PD + distributed KV cache pool with MultiConnector

同一套接入也自然叠到 P/D 分离，走 [`MultiConnector`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py)。Figure 3：`MultiConnector` 是把若干子 connector 串起来的包装。每个 connector 独立工作，彼此不依赖。

![animation](../../../../assets/vllm/blog/serving/mooncake/04-animation.gif)

**图注（原文 Figure 3）。** 经 `MultiConnector`，P/D 分离和分布式 KV 池叠在一起。

**Prefill。** Prefill 实例既给 P/D connector 准备 KV block，也经 store connector 写入分布式池。命中时 vLLM 向所有 connector 询问，可以从 Mooncake Store connector 把匹配前缀捞回来。

**Decode。** Decode 实例把 KV block 写入分布式池，立刻对 Prefill 实例可见。Decode **当时还不从池子读**：vLLM 会把每个请求同时排到一台 Prefill 和一台 Decode；Prefill 从池子加载前缀 KV，再经 P/D connector 转给 Decode。

他们还在做从 Prefill 实例与分布式池**双路径**同时加载，好把可用网络带宽吃满。后来的方向指向类似 [DualPath](https://arxiv.org/abs/2602.21548) 的同时加载。

## Performance

当时实现：[PR #40900](https://github.com/vllm-project/vllm/pull/40900)。Bench 脚本在 artifact 仓库：[ivanium/vllm `scripts/mooncake/artifacts`](https://github.com/ivanium/vllm/tree/feat/mooncake-store-int/scripts/mooncake/artifacts)。文中亮两张成绩。

Kimi-2.5 **NVFP4**，GB200，P/D 分离：Prefill 用 **TP4**，Decode 用 **DP8 + EP**。他们认为这是当时延迟–吞吐最好的折中。

### Speeding up real agentic traces

先用上文 Codex 轨迹做真实场景。部署 **1P1D**，一共 **12** 张 GPU。

![pd compare mooncake vs nixl](../../../../assets/vllm/blog/serving/mooncake/05-pd_compare_mooncake_vs_nixl.png)

**图注（原文 Figure 4）。** 真实 Codex agent 轨迹上，vLLM + Mooncake Store vs 基线（1P1D，12 张 GB200）。分布式 KV 池把吞吐抬约 **3.8×**，P50 TTFT 降约 **46×**，E2E 降约 **8.6×**；驱动是命中率从 **1.7%** 升到 **92.2%**。

分布式 KV 池把吞吐抬约 **3.8×**，P50 TTFT 与 E2E 分别降约 **46×** 和 **8.6×**。驱动数字是命中率：从大约 **1.7%**（几乎只缓存了系统提示）到大约 **92.2%**（前缀几乎都在）。

### Scaling out to multiple nodes

可扩展性测试再加节点，用从 Codex 负载派生的合成数据做可控扩容。

实验设定：

- **20K** 公共 token（系统指令）
- 首轮输入 **10K** token
- 之后每回合输入 **2,048** token
- 输出 **900** token
- 一共 **30** 回合
- 会话数随 GPU 涨：75 → 150 → 225 → 300 → 375
- 参数大致贴着原始 Codex，总 output/input 比大约 **1.3%**

![pd scaling](../../../../assets/vllm/blog/serving/mooncake/06-pd_scaling.png)

**图注（原文 Figure 5）。** Round-robin 路由下，Mooncake Store 从 12 张扩到 60 张 GB200。各规模命中率都 **>95%**，吞吐接近线性。

为了故意制造跨节点流量，路由用 **round-robin**。结果：回合之间请求经常落在另一台机器上，常常要从前一节点把 KV 捞过来。

没有分布式 KV 池，这种路由就是大规模 miss、吞吐塌掉。有 Mooncake Store：命中率始终 **>95%**，系统扩到 **60** 张 GPU 仍接近线性。

这张图说的是：分布式 KV 池既把命中率抬上去，也在集群变大时把 datapath 保住。

## What's next?

当时还在做：

- **Distributed disk offloading。** 存储层级从 CPU DRAM 再往 NVMe SSD 和分布式文件系统伸，容量再大一档。
- **KV cache offloading for hybrid models。** 混合注意力架构，层与层的缓存策略可能不一样。
- **Cache-aware routing。** 路由器和 KV 池一起设计：下一回合先送到已经握着前缀的实例，本地命中优先，分布式池当退路。
- **Further datapath optimization。** RDMA 之外再用 NVIDIA 多节点 NVLink，多路径传 KV。也在探类似 DualPath 的从 Prefill 与 Decode 同时加载，把聚合带宽吃满。

## Acknowledgements

vLLM × Mooncake Store 的接入，很大程度受 [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend) 启发。特别感谢 Ant Group 的 Chao Lei 做初版，以及 Inferact 的 Zijing Liu 做 agent 轨迹与分析。

也感谢 Approaching.AI 的 Jiahao Lu、Zuoyuan Zhang、Zihan Tang、Ke Yang；Huawei 的 Pengbo Zhao、Fuqiao Duan、Tianyu Xu；Alibaba Cloud Computing 的 Tianchen Ding、Xuchun Shang、Xingrui Yi、Teng Ma；Ant Group 的 Yunxiao Ning、Dejiang Zhu、Shoujian Zheng；9#AISoft 的 Feng Ren。更广的 vLLM 与 Mooncake 社区，以及全程协作的 Inferact 团队。

读完 [Router](router.md) 和 [大规模 serving](large-scale.md) 再读这一篇：认得 KV 的路由器解决「下一句话去哪」；分布式 KV 池解决「去了另一台也不必重读」。Agent 把 ISL/OSL 拉到 131:1 以后，这两件事是同一句话的上下半句。
