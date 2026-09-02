---
source: https://vllm.ai/blog/2026-07-29-optimizing-vllm-on-arm-cpus
lang: en
fetched: 2026-09-01
---

# Optimizing vLLM on Arm CPUs

2026-07-29. Arm / PyTorch / oneDNN / KleidiAI.  Demos vs an October 2025 BF16 baseline on Neoverse. Dense GEMMs were already near hardware efficiency (~80% of runtime); the rest was allocator, OpenMP, layout, attention, quantization. Hardware-out-of-core: [hardware-plugin.md](hardware-plugin.md).

Enablement: wheels/Docker; chunked prefill and prefix caching; INT8 W8A8 / W4A8; GPT-OSS, Whisper, Qwen 3.5 / 3.6.

**mimalloc** as PyTorch default on Arm (glibc `malloc` did not reuse large tensors). Llama 3.1 8B: ~**2.3×** offline, ~**7×** low-concurrency serving. Allocator gains excluded from the heatmaps.

**LSE atomics** in libgomp: high-core paged attention spent ~74% in `gomp_iter_dynamic_next` LL/SC retries. Neoverse V2 has `LDADDAL`. Offline **+9%**, low-concurrency TPOT **−15%**.

**oneDNN prepack** of BF16 weights at warmup. Offline **+16%**, low-concurrency TPOT **−60%**.

**Paged attention:** QK/PV via BFMMLA, vectorized cubic exp. Kernel up to ~**4×**, offline **+12%**; unlocks chunked prefill / prefix cache on CPU. See [paged-attention.md](paged-attention.md).

**INT8:** W8A8 via I8MM `SMMLA`. vs optimized BF16: up to **+88%** throughput, **−45%** TPOT, **−54%** TTFT. W4A8 via KleidiAI: up to **+29%** vs W8A8, largest at low concurrency.

vs Oct 2025 BF16: optimized BF16 up to **2.7×** throughput; W8A8 **4.8×** / TPOT **5.7×**; W4A8 **6.2×** / TPOT **7.8×** / TTFT **2.6×**. Demos.

Local figures (copyright remains with the original site; study copies):

![heatmap bf16 optimized vs bf16 baseline](../../../../assets/vllm/blog/architecture/arm-cpus/01-heatmap_bf16_optimized_vs_bf16_baseline.png)

![heatmap int8 vs bf16 optimized](../../../../assets/vllm/blog/architecture/arm-cpus/02-heatmap_int8_vs_bf16_optimized.png)

![heatmap int4 vs int8](../../../../assets/vllm/blog/architecture/arm-cpus/03-heatmap_int4_vs_int8.png)

![bars all vs bf16 baseline](../../../../assets/vllm/blog/architecture/arm-cpus/04-bars_all_vs_bf16_baseline.png)
