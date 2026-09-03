---
source: https://vllm.ai/blog/2026-08-17-distributed-layerwise-offload
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Distributed Layerwise Offload：124 GB 模型挤进 64 GB HBM

英文对照：[en/vllm/blog/serving/omni-layerwise-offload.md](../../../../en/vllm/blog/serving/omni-layerwise-offload.md)  
原文：https://vllm.ai/blog/2026-08-17-distributed-layerwise-offload  
vLLM 0.27.0 + Omni `v0.27.0rc1`。Cosmos3-Super 64B / 124 GB。

`--enable-distributed-layerwise-offload`。mmap 权重走 OS page cache，冷启动 cgroup 峰值 Cosmos3-Nano DP4 **178 GB → 47 GB**（约 −73%）。每 rank 只留 1/dp 权重，运行时 AllGather 与计算重叠。设备上固定 **双缓冲两层**，与总层数无关。DP 多并发相对单请求 HSDP 约 **3.3×**（理想 4× 的 83%）。8×B300 上 AllGather vs rank-local 看拓扑。v0.26.0 的 Cosmos3 DLO+DP 会拒请求（`supports_request_batch`）；#5864 之后每 rank 走单请求前向。`--dlo-no-use-allgather` 当时还开着 #5911。

本地图（原文版权仍归原站；学习对照用）：

![dlo problem overview](../../../../assets/vllm/blog/serving/omni-layerwise-offload/01-dlo-problem-overview.svg)

![mmap loading memory](../../../../assets/vllm/blog/serving/omni-layerwise-offload/02-mmap-loading-memory.svg)

![weight sharding allgather](../../../../assets/vllm/blog/serving/omni-layerwise-offload/03-weight-sharding-allgather.svg)

![dlo pipeline last frame](../../../../assets/vllm/blog/serving/omni-layerwise-offload/04-dlo_pipeline_last_frame.png)

![dlo pipeline](../../../../assets/vllm/blog/serving/omni-layerwise-offload/05-dlo_pipeline.gif)

![hbm nano vs super](../../../../assets/vllm/blog/serving/omni-layerwise-offload/06-hbm-nano-vs-super.svg)

![dp multi concurrency](../../../../assets/vllm/blog/serving/omni-layerwise-offload/07-dp-multi-concurrency.svg)

![ascend memory accounting](../../../../assets/vllm/blog/serving/omni-layerwise-offload/08-ascend-memory-accounting.svg)

![minimax h3 topology policy](../../../../assets/vllm/blog/serving/omni-layerwise-offload/09-minimax-h3-topology-policy.svg)

![minimax h3 multimodal frontiers](../../../../assets/vllm/blog/serving/omni-layerwise-offload/10-minimax-h3-multimodal-frontiers.png)
