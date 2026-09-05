# LLM 推理笔记

个人学习库，**不是官方译本**。对照 NVIDIA NIM、TensorRT-LLM、AIPerf 与 vLLM 的公开文档，写成学习笔记：指标名、CLI、公式保留英文。vLLM 带日期的博客按原文分节写完；NVIDIA 一侧多为压缩对照。

侧栏按主题排。建议先走这四步：

1. [基本概念](/zh/nvidia/benchmarking/blog-01-fundamental-concepts.md) — TTFT / ITL / TPS；压测和性能测试不是同一把尺子
2. [用 AIPerf 打一轮](/zh/nvidia/benchmarking/nim-04-aiperf.md) — 画出 latency–throughput 曲线
3. [vLLM 调优顺序](/zh/vllm/optimization/optimization.md) — CPU 核、`-O*`、batch、并行、cache
4. [必读博客](/zh/vllm/blog/MUST-READ.md) — 主线机制，不必按 CATALOG 逐篇读

完整对照表在 [总目录](/README.md)。仓库在收集什么、压测和调优怎么分开、按目的选哪条路，写在 [怎么读](/zh/GUIDE.md)。
