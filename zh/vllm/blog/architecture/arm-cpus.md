---
source: https://vllm.ai/blog/2026-07-29-optimizing-vllm-on-arm-cpus
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Arm CPU 上的 vLLM：瓶颈不在 GEMM

英文对照：[en/vllm/blog/architecture/arm-cpus.md](../../../../en/vllm/blog/architecture/arm-cpus.md)  
原文：https://vllm.ai/blog/2026-07-29-optimizing-vllm-on-arm-cpus  
2026-07-29。Arm / PyTorch / oneDNN / KleidiAI。数字是 Neoverse 上相对 2025-10 BF16 基线的演示。

CPU 能便宜铺开。密集层大约 80% 时间已经在优化过的 BF16 GEMM 里，kernel 本身贴着硬件效率。剩下是 allocator、OpenMP 同步、weight layout、attention、量化。另一条「硬件进主干」的路见 [hardware-plugin](hardware-plugin.md)。


本地图（原文版权仍归原站；学习对照用）：

![heatmap bf16 optimized vs bf16 baseline](../../../../assets/vllm/blog/architecture/arm-cpus/01-heatmap_bf16_optimized_vs_bf16_baseline.png)

![heatmap int8 vs bf16 optimized](../../../../assets/vllm/blog/architecture/arm-cpus/02-heatmap_int8_vs_bf16_optimized.png)

![heatmap int4 vs int8](../../../../assets/vllm/blog/architecture/arm-cpus/03-heatmap_int4_vs_int8.png)

![bars all vs bf16 baseline](../../../../assets/vllm/blog/architecture/arm-cpus/04-bars_all_vs_bf16_baseline.png)

## 能用

预编译 wheel 与 Docker；chunked prefill、prefix caching；INT8 W8A8 / W4A8；GPT-OSS、Whisper、Qwen 3.5 / 3.6。

## 五刀

**mimalloc。** glibc `malloc` 大块不复用、多线程抢。PyTorch 在 Arm 默认改 mimalloc。Llama 3.1 8B 开箱离线吞吐约 **2.3×**，低并发 serving 约 **7×**。文中热力图把 allocator 贡献拿掉了，免得盖住别的。

**LSE atomics。** 高核数时 paged attention 里约 74% 耗在 `gomp_iter_dynamic_next` 的 LL/SC 重试。Neoverse V2 有 LSE（`LDADDAL`），PyTorch 那份 libgomp 没用。换成 LSE 后离线 **+9%**，低并发 TPOT **−15%**。

**oneDNN 预打包。** 低并发每次 GEMM 都付 layout 转换。warmup 把 BF16 权重打成 kernel 格式。离线 **+16%**，低并发 TPOT **−60%**。

**Paged attention。** QK / PV 走 BFMMLA，softmax exp 用三次多项式。kernel 最多约 **4×**，离线 **+12%**；prefill 也能走 paged，chunked prefill 和 prefix caching 才开得了。Paged 本身见 [paged-attention](paged-attention.md)。

**INT8。** W8A8 走 I8MM `SMMLA`（相对 BF16 理论 matmul 吞吐 2×）。相对优化后的 BF16：吞吐最多 **+88%**，TPOT **−45%**，TTFT **−54%**。W4A8 再走 KleidiAI INT4：相对 W8A8 吞吐最多 **+29%**，低并发更吃带宽。

相对 2025-10 BF16：优化 BF16 吞吐最多 **2.7×**；W8A8 吞吐 **4.8×**、TPOT **5.7×**；W4A8 吞吐 **6.2×**、TPOT **7.8×**、TTFT **2.6×**。演示。
