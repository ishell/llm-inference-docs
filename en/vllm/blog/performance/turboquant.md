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
