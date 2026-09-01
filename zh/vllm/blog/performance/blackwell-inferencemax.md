---
source: https://vllm.ai/blog/2025-10-09-blackwell-inferencemax
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# InferenceMAX：Blackwell 相对 Hopper 是整条 Pareto，不是一个点

英文对照：`en/vllm/blog/performance/blackwell-inferencemax.md`  
原文：https://vllm.ai/blog/2025-10-09-blackwell-inferencemax  
2025-10。gpt-oss 120B / Llama 3.3 70B。图在原网页。后续数字见 [gpt-oss-optimizations](gpt-oss-optimizations.md)。

InferenceMAX 每天重跑，1K/1K、1K/8K、8K/1K 三档 ISL/OSL。他们报 Blackwell vs Hopper：gpt-oss 1k/1k 最高约 **4.3×** 吞吐（同 interactivity）；Llama 3.3 70B 1k/8k 最高约 **3.7×**。单点 TPS 会骗人——高吞吐配置通常不是最低每用户延迟。

栈：FlashInfer（FP8 attention/GEMM/MoE，AR+RMSNorm+quant 熔核）；`torch.compile` 扩到 Attention+Output Quant；`--async-scheduling` 把 host 开销叠进 GPU。自动选量化 backend 和 attention；FlashInfer GEMM/MoE 启动时 autotune。后续想靠 EAGLE3 / DEP 再抬集群吞吐。数字跟 SemiAnalysis 当天曲线走，不要当永久铭牌。
