---
source: https://vllm.ai/blog/2025-09-11-semantic-router
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Semantic Router：按意图决定走不走推理

英文对照：[en/vllm/blog/serving/semantic-router.md](../../../../en/vllm/blog/serving/semantic-router.md)  
原文：https://vllm.ai/blog/2025-09-11-semantic-router  
2025-09 立项文。v0.1 架构翻新见 [Iris](semantic-router-iris.md)。

vLLM 会把 GPU 喂饱，但不会问「这句要不要 CoT」。全开推理贵，全关复杂题掉点。Semantic Router 用语义分类把简单查询送快路径、复杂查询送推理模型。当时四根柱：ModernBERT 分类、快/慢路由、Rust + Candle、K8s / Envoy `ext_proc`。试验数字约 **+10%** 准确、**~50%** 延迟、**~50%** token；商科场景准确可再高一截——当演示。

坑：推理预算没有动态闸门会把 TTFT/p95 打爆；工具目录膨胀会伤准确，路由侧要先滤工具。Classifier 当时跑在路由进程里，还不是 vLLM embedding。这是 **控制面**，不是替换 [Router](router.md) 那只 Rust P/D 负载均衡器——名字都叫 router，职责不同。

本地图（原文版权仍归原站；学习对照用）：

![request](../../../../assets/vllm/blog/serving/semantic-router/01-request.png)

![architecture](../../../../assets/vllm/blog/serving/semantic-router/02-architecture.png)
