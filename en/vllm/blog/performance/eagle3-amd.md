---
source: https://vllm.ai/blog/2026-07-13-eagle-3-amd-instinct
lang: en
fetched: 2026-09-01
---

# EAGLE 3 on Instinct: Quark MXFP4, Kimi-K2.5 ~1.69–2.00×

Chinese: `../../zh/vllm/blog/performance/eagle3-amd.md`  
CUDA EAGLE: [p-eagle](p-eagle.md) / [eagle31](eagle31.md); AMD spec-decode: [amd-spec-decode](amd-spec-decode.md).

Quark MXFP4. Kimi-K2.5 ~**1.69–2.00×**; MiniMax-M2.5 ~**1.38–1.79×**; MiniMax-M3 acceptance length **2.80**. Draft precision need not match verify — quant hits draft bandwidth. Numbers are their prompt / accept-rate mix; re-measure acceptance length on your model.

Local figures (copyright remains with the original site; study copies):

![figure1](../../../../assets/vllm/blog/performance/eagle3-amd/01-figure1.png)

![figure2](../../../../assets/vllm/blog/performance/eagle3-amd/02-figure2.png)

![figure3](../../../../assets/vllm/blog/performance/eagle3-amd/03-figure3.png)

![figure4](../../../../assets/vllm/blog/performance/eagle3-amd/04-figure4.png)

![figure5](../../../../assets/vllm/blog/performance/eagle3-amd/05-figure5.png)
