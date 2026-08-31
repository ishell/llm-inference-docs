---
source: https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/
lang: zh
fetched: 2026-08-30
---

# 系列第 4 篇：推理要花多少钱（中文摘译）

原文：https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/

前面测完延迟/吞吐，这篇把它变成 **TCO 和每 token 成本**。

## 步骤

1. **基准测试**（AIPerf/GenAI-Perf）：每个部署单元在负载下的吞吐和延迟。
2. **画延迟–吞吐曲线**，找 **Pareto 前沿**：同样延迟下吞吐最高的点。低并发：延迟低、吞吐低；高并发：batch 效应吞吐升、延迟也升。比较 FP4/FP8/BF16 时，吞吐要按 GPU 数归一化。
3. **定约束**：例如交互式聊天平均 TTFT ≤ 250 ms；以及规划峰值请求/秒（不是同时在线用户数，用户不会人人都在发请求）。
4. 丢掉不满足延迟的点，在剩下的里面选吞吐最高的，得到 **单实例可达 RPS** 和 **每实例 GPU 数**。
5. **最少实例数** = 规划峰值 RPS / 单实例可达 RPS。
6. **服务器数** = 实例数 × 每实例 GPU / 每台服务器 GPU 数。
7. **年成本/台** = 购机价 / 折旧年限 + 年托管 + 年软件许可。
8. **总成本** = 服务器数 × 年成本/台。

再拆成业界常用口径：

- 每 1000 个 prompt 成本 = 年成本/台 /（一台一年能打完的请求数），再按实际上线率打折
- 每 100 万混合 token 成本：用该场景的 ISL+OSL 把 prompt 成本换算成 token
- 输入/输出分开计价：商业 API 通常输出更贵（文中举例 $1 / 1M input vs $3 / 1M output）

文中硬件举例（仅作公式演示，不是报价）：一台服务器 $320,000、8 GPU、折旧 4 年、年托管 $3,000、年软件 $4,500。

动手课：Sizing LLM Inference Systems（NVIDIA 在线课）。
