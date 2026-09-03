---
source: https://vllm.ai/blog/2026-01-05-vllm-sr-iris
lang: en
fetched: 2026-09-01
---

# Semantic Router v0.1 Iris: from 14 classes to a signal chain

Chinese: [zh/vllm/blog/serving/semantic-router-iris.md](../../../../zh/vllm/blog/serving/semantic-router-iris.md)  
Signal/decision internals: later signal-decision post.

Launch-day used one ModernBERT over 14 MMLU domains. Iris is **signal → decision → plugins**: domain / keyword / embedding / factual / feedback / preference, AND/OR with priority. Jailbreak, PII, semantic cache, HaluGate become per-decision plugins. Classification with Candle is **one shared base + many LoRAs**: N full forwards become 1 + N×ε.

HaluGate: Sentinel (does this need facts) → Detector (which tokens are ungrounded) → Explainer (contradiction vs neutral). Tool results are ground truth; verdicts ride HTTP headers.

```
pip install vllm-sr
vllm-sr init
```

K8s: `helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router`. The MoM family is routing-only small models (domain/PII/jailbreak/HaluGate/tool/embedding). Also `/v1/responses` stateful chats and semantic tool filtering. Do not confuse with in-engine [Router](router.md).

Local figures (copyright remains with the original site; study copies):

![iris 0](../../../../assets/vllm/blog/serving/semantic-router-iris/01-iris-0.png)

![iris 1](../../../../assets/vllm/blog/serving/semantic-router-iris/02-iris-1.png)

![iris 2](../../../../assets/vllm/blog/serving/semantic-router-iris/03-iris-2.png)

![iris 3](../../../../assets/vllm/blog/serving/semantic-router-iris/04-iris-3.png)

![iris 4](../../../../assets/vllm/blog/serving/semantic-router-iris/05-iris-4.png)

![iris 7](../../../../assets/vllm/blog/serving/semantic-router-iris/06-iris-7.png)

![iris 6](../../../../assets/vllm/blog/serving/semantic-router-iris/07-iris-6.png)

![iris 5](../../../../assets/vllm/blog/serving/semantic-router-iris/08-iris-5.png)

![iris 8](../../../../assets/vllm/blog/serving/semantic-router-iris/09-iris-8.png)
