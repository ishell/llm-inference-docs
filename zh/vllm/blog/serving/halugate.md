---
source: https://vllm.ai/blog/2025-12-14-halugate
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# HaluGate：工具已经说对了，模型还在编

英文对照：`en/vllm/blog/serving/halugate.md`  
原文：https://vllm.ai/blog/2025-12-14-halugate  
图在原网页。挂在 [Iris](semantic-router-iris.md) 的插件链上。

工具回了艾菲尔 1887–1889 / 330m，模型仍说 1950 / 500m——外在幻觉。HaluGate **不用 LLM-as-judge**：工具消息当 context，用户句当 question，助手句当要核的 claim。三截：Sentinel（这句要不要核事实；创作/代码常跳过）→ Detector（哪些 token 没接地）→ Explainer（矛盾 vs 中性）。结果走 HTTP header，下游自己定拦还是标。Rust 路径，宣称毫秒级——以原文测量为准。和引擎里的 structured decode 不是一件事。
