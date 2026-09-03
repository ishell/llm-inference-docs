---
source: https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus
lang: en
fetched: 2026-09-01
---

# Speculative decoding on AMD GPUs: five draft paths

Chinese: [zh/vllm/blog/performance/spec-decode-amd.md](../../../../zh/vllm/blog/performance/spec-decode-amd.md)  
MI300X / MI355X, ROCm. Tables on the original page. Demo numbers, not a promise.

Draft proposes, target verifies once: lossless. Five paths:

| Method | How the draft grows | Config |
|---|---|---|
| Native MTP | Model-native aux head, sequential | `"method": "mtp"` — no extra draft path |
| Gemma 4 MTP | Separate checkpoint, still eats target activations | `"method": "mtp"` + `"model"` |
| EAGLE-3 | Three hidden layers, autoregressive | `"method": "eagle3"` |
| DFlash | One forward for a block of positions | `"method": "dflash"` |
| DSpark | DFlash backbone + light causal correction + confidence prefix | `"method": "dspark"` |

`num_speculative_tokens` is not the physical MTP depth: larger N reuses the head. Larger N is not always faster — late positions drop accept rate and you pay draft tax. Gemma-class demos often **2.5–2.8×** output TPS at N=3–5; other models/drafts/concurrency can be flat or negative. Tune N from per-position accepts.

Read with [spec-decode](spec-decode.md), [P-EAGLE](p-eagle.md), [parallel drafting](parallel-drafting.md), [DSpark adaptive](dspark-adaptive.md): this is **how to turn it on and measure on ROCm**, not new accept math.

Local figures (copyright remains with the original site; study copies):

![figure 01](../../../../assets/vllm/blog/performance/spec-decode-amd/01-figure-01.svg)

![figure 02](../../../../assets/vllm/blog/performance/spec-decode-amd/02-figure-02.svg)

![figure method summary](../../../../assets/vllm/blog/performance/spec-decode-amd/03-figure-method-summary.svg)
