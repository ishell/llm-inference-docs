---
source: https://vllm.ai/blog/2026-07-14-vllm-tilert-pd
lang: en
fetched: 2026-09-01
---

# TileRT: decode-side P/D without a vLLM fork

Chinese: `../../zh/vllm/blog/serving/tilert.md`  
TileRT 0.1.5, V1 KV Connector.

Prefill stays stock vLLM; decode is TileRT. The join is a V1 connector — **no vLLM fork**. Demos then: GLM-5 / 5.1, DeepSeek-V3.2. Prefill needs MTP; decode has one in-flight request. Dual pools: MultiConnector + NIXL.

Read with [Mooncake](mooncake.md) and [Router](router.md): the decode engine is swapped, not a new P/D protocol. Numbers, topology, and “is this the production default” follow the original post and that week’s release.

Local figures (copyright remains with the original site; study copies):

![pd arch](../../../../assets/vllm/blog/serving/tilert/01-pd_arch.png)

![glm5 tilert mtp](../../../../assets/vllm/blog/serving/tilert/02-glm5_tilert_mtp.png)
