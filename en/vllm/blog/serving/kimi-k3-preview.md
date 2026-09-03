---
source: https://vllm.ai/blog/2026-07-22-kimi-k3-preview
lang: en
fetched: 2026-09-01
---

# Kimi K3 preview (before weights)

Chinese: [zh/vllm/blog/serving/kimi-k3-preview.md](../../../../zh/vllm/blog/serving/kimi-k3-preview.md)  
2026-07-22. Weights planned 2026-07-27. Launch numbers: [k3](kimi-k3.md).

2.8T, 1M, native vision, KDA + AttnRes + 896/16 LatentMoE, MXFP4 + SiTU. Not a bigger K2. KDA keeps a recurrent state, not per-token KV. Large physical state blocks used to pin prefix hits to block boundaries — near-identical prompts still missed.

vLLM splits **physical block size**, **scheduler alignment**, and **prefix-match unit**. Fine-grained KDA snapshots inside a large block; copy-on-write before extend. Full-attention and KDA share one `num_computed_tokens`. **Core infra**, not a K3 fork.

Then in flight: FlashKDA prefill; fused NVIDIA decode; AttnRes kernels; hand-fused MLA with separate P/D paths; MXFP4 MoE + SiTU (checked DP16+EP16); AMD FlyDSL A16W4/A8W4. Non-disagg serving worked; Dynamo+Mooncake still closing.

Announce first, open weights later — freeze the checkpoint, then integration. **Not an engine-architecture post.**

Local figures (copyright remains with the original site; study copies):

![kda prefix state](../../../../assets/vllm/blog/serving/kimi-k3-preview/01-kda-prefix-state.png)

![fine grained prefix cache](../../../../assets/vllm/blog/serving/kimi-k3-preview/02-fine-grained-prefix-cache.png)
