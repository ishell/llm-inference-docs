---
source: https://vllm.ai/blog/2026-07-13-eagle-3-amd-instinct
lang: en
fetched: 2026-09-01
---

# EAGLE 3 on Instinct: Quark MXFP4, Kimi-K2.5 ~1.69–2.00×

Chinese: `../../zh/vllm/blog/performance/eagle3-amd.md`  
CUDA EAGLE: [p-eagle](p-eagle.md) / [eagle31](eagle31.md); AMD spec-decode: [amd-spec-decode](amd-spec-decode.md).

Quark MXFP4. Kimi-K2.5 ~**1.69–2.00×**; MiniMax-M2.5 ~**1.38–1.79×**; MiniMax-M3 acceptance length **2.80**. Draft precision need not match verify — quant hits draft bandwidth. Numbers are their prompt / accept-rate mix; re-measure acceptance length on your model.
