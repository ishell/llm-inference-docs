---
source: https://vllm.ai/blog/2025-12-14-halugate
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# HaluGate：工具已经说对了，模型还在编

英文对照：[en/vllm/blog/serving/halugate.md](../../../../en/vllm/blog/serving/halugate.md)  
原文：https://vllm.ai/blog/2025-12-14-halugate  
挂在 [Iris](semantic-router-iris.md) 的插件链上。

工具回了艾菲尔 1887–1889 / 330m，模型仍说 1950 / 500m——外在幻觉。HaluGate **不用 LLM-as-judge**：工具消息当 context，用户句当 question，助手句当要核的 claim。三截：Sentinel（这句要不要核事实；创作/代码常跳过）→ Detector（哪些 token 没接地）→ Explainer（矛盾 vs 中性）。结果走 HTTP header，下游自己定拦还是标。Rust 路径，宣称毫秒级——以原文测量为准。和引擎里的 structured decode 不是一件事。

本地图（原文版权仍归原站；学习对照用）：

![halugate 0](../../../../assets/vllm/blog/serving/halugate/01-halugate-0.png)

![halugate 1](../../../../assets/vllm/blog/serving/halugate/02-halugate-1.png)

![halugate 2](../../../../assets/vllm/blog/serving/halugate/03-halugate-2.png)

![halugate 3](../../../../assets/vllm/blog/serving/halugate/04-halugate-3.png)

![halugate 4](../../../../assets/vllm/blog/serving/halugate/05-halugate-4.png)

![halugate 5](../../../../assets/vllm/blog/serving/halugate/06-halugate-5.png)

![halugate 6](../../../../assets/vllm/blog/serving/halugate/07-halugate-6.png)

![halugate 7](../../../../assets/vllm/blog/serving/halugate/08-halugate-7.png)

![halugate 8](../../../../assets/vllm/blog/serving/halugate/09-halugate-8.png)

![halugate 9](../../../../assets/vllm/blog/serving/halugate/10-halugate-9.png)

![halugate 10](../../../../assets/vllm/blog/serving/halugate/11-halugate-10.png)
