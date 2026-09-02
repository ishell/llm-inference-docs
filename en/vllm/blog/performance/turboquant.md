---
source: https://vllm.ai/blog/2026-05-11-turboquant
lang: en
fetched: 2026-09-01
---

# TurboQuant vs FP8 KV

2026-05-11. vLLM 0.20.2. Read after [fp8-kvcache.md](fp8-kvcache.md). GQA-style attention only then. Study note.

FP8 (`--kv-cache-dtype fp8`) quantizes **attention compute** on Tensor Cores. TurboQuant compresses **storage** to 3–4 bits and dequants to BF16 for attention. Rooms vs dequant tax.

Variants: `k8v4`, `4bit_nc`, `k3v4_nc`, `3bit_nc`. Models: Llama-3.3-70B, Qwen3-30B-A3B Instruct/Thinking, MiniMax-M2.7. Bench: MRCR; AIME25 / GPQA / MATH500 / LiveCodeBench.

**Default remains FP8:** ~2× KV capacity, negligible accuracy hit, matches BF16, wins when memory-tight.

**`k8v4`:** ~2.4× vs 2× capacity — not worth the throughput/latency hit vs FP8.

**`4bit_nc`:** most practical TQ; ~2.3–3.7× capacity, modest accuracy/speed cost. Validate on the target workload (about 1–4 points). Edge / extreme memory.

**`k3v4_nc` / `3bit_nc`:** not a production default. Hard reasoning and very long context drop (up to ~20 points on Thinking; MRCR 256k ~−30% relative). Worse dequant.

Latency +10–68%; TQ throughput strictly below BF16. Llama-70B burst: BF16 P99 TTFT ~17 s (KV saturation); TQ <3.5 s; FP8 ~1.3 s. TQ’s real pitch is “don’t queue”; FP8 already does that without the dequant tax. Short/low-concurrency/ample HBM: stay BF16.

Local figures (copyright remains with the original site; study copies):

![llama 70b pareto](../../../../assets/vllm/blog/performance/turboquant/01-llama_70b_pareto.png)

![qwen3 30b a3b pareto](../../../../assets/vllm/blog/performance/turboquant/02-qwen3_30b_a3b_pareto.png)

![Llama 3.3 70B Instruct openai mrcr 2 needles plot](../../../../assets/vllm/blog/performance/turboquant/03-Llama-3.3-70B-Instruct_openai_mrcr_2_needles_plot.png)

![Qwen3 30B A3B Instruct 2507 openai mrcr 2 needles plot](../../../../assets/vllm/blog/performance/turboquant/04-Qwen3-30B-A3B-Instruct-2507_openai_mrcr_2_needles_plot.png)

![Qwen3 30B A3B Thinking 2507 reasoning plot](../../../../assets/vllm/blog/performance/turboquant/05-Qwen3-30B-A3B-Thinking-2507_reasoning_plot.png)

![MiniMax M2.7 reasoning plot](../../../../assets/vllm/blog/performance/turboquant/06-MiniMax-M2.7_reasoning_plot.png)

![qwen3 30b a3b latency](../../../../assets/vllm/blog/performance/turboquant/07-qwen3_30b_a3b_latency.png)

![llama 70b latency](../../../../assets/vllm/blog/performance/turboquant/08-llama_70b_latency.png)

![qwen3 30b a3b throughput](../../../../assets/vllm/blog/performance/turboquant/09-qwen3_30b_a3b_throughput.png)

![llama 70b throughput](../../../../assets/vllm/blog/performance/turboquant/10-llama_70b_throughput.png)

![qwen3 30b a3b serve](../../../../assets/vllm/blog/performance/turboquant/11-qwen3_30b_a3b_serve.png)

![llama 70b serve](../../../../assets/vllm/blog/performance/turboquant/12-llama_70b_serve.png)

![qwen3 30b a3b ttft](../../../../assets/vllm/blog/performance/turboquant/13-qwen3_30b_a3b_ttft.png)

![llama 70b ttft](../../../../assets/vllm/blog/performance/turboquant/14-llama_70b_ttft.png)
