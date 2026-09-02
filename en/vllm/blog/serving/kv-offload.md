---
source: https://vllm.ai/blog/2026-01-08-kv-offloading-connector
lang: en
fetched: 2026-08-31
---

# KV Offloading Connector

2026-01-08. Landed in 0.11.0; 0.12.0 jumped after a contiguous physical-block layout. Same **KVConnector** door as [mooncake.md](mooncake.md): local DRAM vs cluster pool. Study note.

Reuse prefixes → less prefill. No shared prefix → still useful: GPU KV fills, V1 **RECOMPUTE** preemption drops KV; offload to CPU and reload instead of recomputing. CPU RAM is large, nearby, and a staging tier toward disk.

Async connector since 0.9.0 (sync used to stall the engine). Pluggable backend; bundled CPU backend.

```bash
--kv_offloading_backend native --kv_offloading_size <GB>   # ~0.14 / PR #24498
# older:
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks": N}}'
```

Llama-3.1-8B H100: single-prefill TTFT **2–22×** vs recompute (grows with prompt). Async store does not tax miss TTFT. 10k unique 512-token prefills, GPU prefix cache off: up to **~9×** throughput vs hit rate. 0.12.0: up to **~4×** TTFT, **~5×** throughput. 0.14 aimed to reload preempted requests from CPU (#29870) and fix a race (#31341).

DMA (`cudaMemcpyAsync`) leaves SMs for forward; wants large contiguous copies. Default 16-token blocks fragmented per layer and K/V → few KB. New layout: one physical block across layers (~`2×num_layers`): 8B 32KB→**2MB**. Typical **0.5–2 MB**. Hybrids not yet optimized. Worst-case 0.5 MB (Llama-3.2-1B): custom copy kernel slightly wins TTFT; DMA wins concurrency (kernel can be **~6% worse than no offload** at 0% hit — it fights the compute SMs). Llama-3.1-8B: DMA up to **~32%** more throughput, matched TTFT.

Local figures (copyright remains with the original site; study copies):

![figure1](../../../../assets/vllm/blog/serving/kv-offload/01-figure1.png)

![figure2](../../../../assets/vllm/blog/serving/kv-offload/02-figure2.png)

![figure3](../../../../assets/vllm/blog/serving/kv-offload/03-figure3.png)

![figure4](../../../../assets/vllm/blog/serving/kv-offload/04-figure4.png)

![figure5](../../../../assets/vllm/blog/serving/kv-offload/05-figure5.png)

![figure6](../../../../assets/vllm/blog/serving/kv-offload/06-figure6.png)
