---
source: https://vllm.ai/blog/2026-02-13-gb300-deepseek
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# DeepSeek-V3.2 on GB300：验证部署，不是刷峰值

英文对照：`en/vllm/blog/performance/gb300-deepseek.md`  
原文：https://vllm.ai/blog/2026-02-13-gb300-deepseek  
vLLM 0.14.1，CUDA 13.0。GB300 / B300 288GB。图在原网页。文内强调 **可复现基线**，不是调到顶。

`VLLM_USE_FLASHINFER_MOE_FP4=1`。两卡装得下 NVFP4。V3.2 TP2 prefill-only 单卡 **7360 TGS**；ISL=2k/OSL=1k 输出 **2816 TGS**。R1 NVFP4+EP2 两卡 prefill **22476 TGS**（ISL=2k/OSL=1/batch=256）；混上下文 **3072 TGS**。相对 Hopper：prefill 约 **8×**，混上下文约 **10–20×**。NVFP4+TP2 比 FP8 更吃得开——TP4 把每卡活摊薄，Tensor Core 吃不满。R1 上 EP 更适合作 prefiller；P+D 一体、ISL 大 OSL 小时反而 TP2 更好，免得 attention 把 decode 时间挤掉。

MTP（`num_speculative_tokens 1`）在中低并发、上下文不太长时抬 decode；高并发或 decode 占比极低时开销摊不掉。V3.2 prefill 约为 R1 的 **1/3**：Indexer/Sparse MLA 每层 DSA 约 **2.7×** MLA kernel 时间。DSA 的 TPOT 优势大约在 **10k–20k** 上下文才翻过来。1P1D / 3P1D 相对一体机吞吐更好、TPOT 更稳；ISL 从 2k 到 8k 时 1P1D 会饿 decode，要加 P。v0.14.1 的 P/D 当时还要手动补 #32698。
