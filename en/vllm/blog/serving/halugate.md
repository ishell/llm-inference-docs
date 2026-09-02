---
source: https://vllm.ai/blog/2025-12-14-halugate
lang: en
fetched: 2026-09-01
---

# HaluGate: the tool was right; the model still lied

Chinese: `../../zh/vllm/blog/serving/halugate.md`  
Plugin on [Iris](semantic-router-iris.md).

Tool says Eiffel 1887–1889 / 330 m; the model says 1950 / 500 m — extrinsic hallucination. HaluGate is **not** LLM-as-judge: tool message = context, user = question, assistant = claims. Three stages: Sentinel (skip creative/code) → Detector (which tokens are ungrounded) → Explainer (contradiction vs neutral). Verdicts ride HTTP headers; downstream blocks or labels. Rust path, claimed millisecond-class — trust their measurement. Not structured decode inside the engine.

Local figures (copyright remains with the original site; study copies):

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
