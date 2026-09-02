---
source: https://vllm.ai/blog/2026-07-23-vllm-afd-plugin
lang: en
fetched: 2026-08-31
---

# vLLM AFD Plugin

2026-07-23. Experimental: https://github.com/vllm-project/afd-plugin via `vllm.general_plugins` + `--additional-config`. Pinned **vLLM 0.19.1**, runner v1 only, full weights on both roles. Study note; not an SLA.

MoE layers mix stateful Attention (scheduler + KV) with routed FFN/experts. One rank topology is the wrong number for both. AFD splits them: requests still hit the Attention OpenAI server; FFN is a connector-driven daemon.

Connectors: GPU `P2pNcclAFDConnector` (sync decode, `FULL_DECODE_ONLY` CUDA graph); NPU `CAMP2pAFDConnector` (sync decode, ACL graph); `CAMAsyncAFDConnector` (async prefill, no graph yet). Wrappers: DeepSeek V2/V3-family, GLM MoE DSA. DBO: exactly two ubatches.

Controlled decode on Ascend 910C, DeepSeek-V3.2 W8A8, forced expert balance (changes outputs). tok/s/die: 16K EP64 **232.6**, 48A16F 220.3 (−5.3%), 64A16F **258.9** (+11.3%); 32K 168.2 / 151.4 / **183.3** (+9.0%). Split ≠ win; Attention:FFN ratio does. Async prefill (2×910C, 10-layer V3.2): median TTFT at 12 rps **15.1 s → 8.0 s**. Path check, not a full-model claim.

EPD splits the ViT; Router splits text P/D; AFD splits Attention vs experts inside the layer.

Local figures (copyright remains with the original site; study copies):

![vllm afd plugin architecture](../../../../assets/vllm/blog/serving/afd/01-vllm-afd-plugin-architecture.svg)

![throughput dsv3 2 16k](../../../../assets/vllm/blog/serving/afd/02-throughput_dsv3-2_16k.png)

![throughput dsv3 2 32k](../../../../assets/vllm/blog/serving/afd/03-throughput_dsv3-2_32k.png)

![text matched dp afd median ttft](../../../../assets/vllm/blog/serving/afd/04-text_matched_dp_afd_median_ttft.png)
