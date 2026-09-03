---
source: https://vllm.ai/blog/2026-02-13-gb300-deepseek
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# DeepSeek-V3.2 on GB300：验证部署，不是刷峰值

英文对照：[en/vllm/blog/performance/gb300-deepseek.md](../../../../en/vllm/blog/performance/gb300-deepseek.md)  
原文：https://vllm.ai/blog/2026-02-13-gb300-deepseek  
vLLM 0.14.1，CUDA 13.0。GB300 / B300 288GB。文内强调 **可复现基线**，不是调到顶。

`VLLM_USE_FLASHINFER_MOE_FP4=1`。两卡装得下 NVFP4。V3.2 TP2 prefill-only 单卡 **7360 TGS**；ISL=2k/OSL=1k 输出 **2816 TGS**。R1 NVFP4+EP2 两卡 prefill **22476 TGS**（ISL=2k/OSL=1/batch=256）；混上下文 **3072 TGS**。相对 Hopper：prefill 约 **8×**，混上下文约 **10–20×**。NVFP4+TP2 比 FP8 更吃得开——TP4 把每卡活摊薄，Tensor Core 吃不满。R1 上 EP 更适合作 prefiller；P+D 一体、ISL 大 OSL 小时反而 TP2 更好，免得 attention 把 decode 时间挤掉。

MTP（`num_speculative_tokens 1`）在中低并发、上下文不太长时抬 decode；高并发或 decode 占比极低时开销摊不掉。V3.2 prefill 约为 R1 的 **1/3**：Indexer/Sparse MLA 每层 DSA 约 **2.7×** MLA kernel 时间。DSA 的 TPOT 优势大约在 **10k–20k** 上下文才翻过来。1P1D / 3P1D 相对一体机吞吐更好、TPOT 更稳；ISL 从 2k 到 8k 时 1P1D 会饿 decode，要加 P。v0.14.1 的 P/D 当时还要手动补 #32698。

本地图（原文版权仍归原站；学习对照用）：

![dsv32 fp4 vs fp8 throughput](../../../../assets/vllm/blog/performance/gb300-deepseek/01-dsv32-fp4-vs-fp8-throughput.png)

![dsr1 h200 b300 gb300 throughput](../../../../assets/vllm/blog/performance/gb300-deepseek/02-dsr1-h200-b300-gb300-throughput.png)

![dsr1 ep2 tp2 throughput prefill only](../../../../assets/vllm/blog/performance/gb300-deepseek/03-dsr1-ep2-tp2-throughput-prefill-only.png)

![dsr1 ep2 tp2 ttft prefill only](../../../../assets/vllm/blog/performance/gb300-deepseek/04-dsr1-ep2-tp2-ttft-prefill-only.png)

![dsr1 ep2 tp2 pd throughput](../../../../assets/vllm/blog/performance/gb300-deepseek/05-dsr1-ep2-tp2-pd-throughput.png)

![dsr1 ep2 tp2 pd ttft](../../../../assets/vllm/blog/performance/gb300-deepseek/06-dsr1-ep2-tp2-pd-ttft.png)

![dsr1 ep2 tp2 pd tpot](../../../../assets/vllm/blog/performance/gb300-deepseek/07-dsr1-ep2-tp2-pd-tpot.png)

![dsr1 mtp throughput](../../../../assets/vllm/blog/performance/gb300-deepseek/08-dsr1-mtp-throughput.png)

![dsr1 mtp ttft](../../../../assets/vllm/blog/performance/gb300-deepseek/09-dsr1-mtp-ttft.png)

![dsr1 mtp peak output throughput](../../../../assets/vllm/blog/performance/gb300-deepseek/10-dsr1-mtp-peak-output-throughput.png)

![dsr1 mtp tpot](../../../../assets/vllm/blog/performance/gb300-deepseek/11-dsr1-mtp-tpot.png)

![dsr1 vs v32 throughput](../../../../assets/vllm/blog/performance/gb300-deepseek/12-dsr1-vs-v32-throughput.png)

![dsr1 vs v32 ttft](../../../../assets/vllm/blog/performance/gb300-deepseek/13-dsr1-vs-v32-ttft.png)

![dsv32 pd disagg throughput](../../../../assets/vllm/blog/performance/gb300-deepseek/14-dsv32-pd-disagg-throughput.png)

![dsv32 pd disagg tpot](../../../../assets/vllm/blog/performance/gb300-deepseek/15-dsv32-pd-disagg-tpot.png)

![dsv32 pd disagg throughput isl8k](../../../../assets/vllm/blog/performance/gb300-deepseek/16-dsv32-pd-disagg-throughput-isl8k.png)
