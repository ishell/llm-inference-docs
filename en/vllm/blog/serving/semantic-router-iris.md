---
source: https://vllm.ai/blog/2026-01-05-vllm-sr-iris
lang: en
fetched: 2026-09-01
---

# Semantic Router v0.1 Iris: from 14 classes to a signal chain

Chinese: `../../zh/vllm/blog/serving/semantic-router-iris.md`  
Signal/decision internals: later signal-decision post.

Launch-day used one ModernBERT over 14 MMLU domains. Iris is **signal → decision → plugins**: domain / keyword / embedding / factual / feedback / preference, AND/OR with priority. Jailbreak, PII, semantic cache, HaluGate become per-decision plugins. Classification with Candle is **one shared base + many LoRAs**: N full forwards become 1 + N×ε.

HaluGate: Sentinel (does this need facts) → Detector (which tokens are ungrounded) → Explainer (contradiction vs neutral). Tool results are ground truth; verdicts ride HTTP headers.

```
pip install vllm-sr
vllm-sr init
```

K8s: `helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router`. The MoM family is routing-only small models (domain/PII/jailbreak/HaluGate/tool/embedding). Also `/v1/responses` stateful chats and semantic tool filtering. Do not confuse with in-engine [Router](router.md).
