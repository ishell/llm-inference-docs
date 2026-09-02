---
source: https://vllm.ai/blog/2026-05-11-turboquant
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# TurboQuant：KV 再压到 3–4 bit 之前，先读完这篇对照

英文对照：`en/vllm/blog/performance/turboquant.md`  
原文：https://vllm.ai/blog/2026-05-11-turboquant  
2026-05-11。vLLM 0.20.2。接在 [FP8 KV](fp8-kvcache.md) 后面读。TurboQuant 当时只支持标准注意力（GQA 一类），滑动窗口 / 混合注意力还没有。

关键差别：FP8（`--kv-cache-dtype fp8`）连 **attention 计算** 也走硬件 FP8 Tensor Core；TurboQuant 只压缩**存储**到 3–4 bit，算的时候再反量化回 BF16。省的是房间，付的是反量化。

变体：`k8v4`（K 8 bit / V 4 bit）、`4bit_nc`（4 bit + norm correction）、`k3v4_nc`、`3bit_nc`。模型：Llama-3.3-70B、Qwen3-30B-A3B Instruct/Thinking、MiniMax-M2.7。基准：MRCR 长上下文检索；AIME25 / GPQA / MATH500 / LiveCodeBench 推理。


本地图（原文版权仍归原站；学习对照用）：

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

## 他们的结论（演示，不是你的 SLA）

**默认仍是 FP8。** 大约 **2×** KV 容量，精度损失可忽略，多数指标打平 BF16，显存紧时还明显更好。

**`k8v4` 几乎没有胜过 FP8 的理由。** 容量大约 2.4× vs 2×，吞吐和延迟却稳定变差。

**`4bit_nc` 是 TurboQuant 里最可能用的。** 容量大约 2.3–3.7×（文中 Pareto：Qwen 上到 3.7×），换中等的精度/延迟/吞吐。边缘、房间极度不够时可以考虑，目标负载上要自己验精度（大约 1–4 分的下降）。

**`k3v4_nc` / `3bit_nc` 不要当生产默认。** 推理和超长上下文掉得狠（Thinking 模型上平均可掉约 20 分；MRCR 256k 相对 BF16 大约 −30% 相对）。反量化更复杂，延迟/吞吐更差。

延迟：Qwen 上 TQ 大约 +10–60%；70B 上 +10–68%，而且随 batch **变大**——反量化量和访问的 KV 一起涨。吞吐：TQ 全在 BF16 之下（Qwen 大约 80%→73%；70B 75%→66%）。serving：70B burst 时 BF16 的 P99 TTFT 冲到约 17 s（KV 饱和排队）；TQ 压在 3.5 s 内，FP8 约 1.3 s。所以 TQ 的真正卖点是 **用速度换「请求不会在门口排队」**；两边都要的话，FP8 已经把这件事做了。

```bash
vllm serve MiniMaxAI/MiniMax-M2.7 --kv-cache-dtype fp8
vllm serve MiniMaxAI/MiniMax-M2.7 --kv-cache-dtype turboquant_4bit_nc
```

短上下文、低并发、显存充足：留 BF16。
