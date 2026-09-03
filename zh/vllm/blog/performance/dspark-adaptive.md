---
source: https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# DSpark 自适应验收：按信心和负载改预算

英文对照：[en/vllm/blog/performance/dspark-adaptive.md](../../../../en/vllm/blog/performance/dspark-adaptive.md)  
原文：https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification  
PR #47808，`enable_adaptive_verification`。演示：DS-V4-Pro TP8、8×B300。

固定 K 的投机在低并发很香，高并发时草稿+验收把 GPU 填满，系统 TPS 反而掉。DSpark 每步用 **草稿信心 × 当前负载** 决定这一步的验收预算：空闲时猜深，挤的时候收短。官方曲线说到 **c=256 仍贴着 Pareto**——不是「永远 K=7」。


本地图（原文版权仍归原站；学习对照用）：

![fig1 policy](../../../../assets/vllm/blog/performance/dspark-adaptive/01-fig1-policy.svg)

![fig2 costcurve](../../../../assets/vllm/blog/performance/dspark-adaptive/02-fig2-costcurve.svg)

![fig3 pareto](../../../../assets/vllm/blog/performance/dspark-adaptive/03-fig3-pareto.svg)

## 引擎条件

要走 **FULL varlen decode graphs**（`AttentionCGSupport.ALWAYS`，SM100 DSV4 那条）。Eager、LoRA、PP、要 output logprobs 的请求当时都不走这套。没有这些图，自适应预算改不动 capture 范围。

和 [并行草稿](parallel-drafting.md)、[投机解码](spec-decode.md) 一起读：改的是 **每步验多少**，不是草稿结构。
