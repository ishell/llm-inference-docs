---
source: https://vllm.ai/blog/2026-02-01-gpt-oss-optimizations
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# gpt-oss on Blackwell：推的是 Pareto，不是单点 TPS

英文对照：`en/vllm/blog/performance/gpt-oss-optimizations.md`  
原文：https://vllm.ai/blog/2026-02-01-gpt-oss-optimizations  
gpt-oss-120b MXFP4 MoE，B200/GB200。图在原网页。续篇见 [blackwell-inferencemax](blackwell-inferencemax.md)。

max-throughput 约 **+38%**，min-latency 约 **+13%**——同时动曲线两端。FlashInfer：`trtllm-gen` / CUTLASS MoE，FP8 KV attention。`torch.compile` 做 AR+RMSNorm fusion，不是手写死融合。Pad+Quant / Finalize+Slice 当时还在滚，预期约 6%。GPU 太快时 CPU 调度跟不上：async scheduling（新版本默认）约 **10%**；`--stream-interval` 把后续 token 缓冲（首 token 仍立刻发）。gpt-oss-20b、1024 并发上他们报端到端约 **57%**——那是输出队列瓶颈被松开，不是 kernel 变 57%。

菜谱：`--cuda-graph-capture-size 2048`；高并发 `--api-server-count 20` 或 `--stream-interval 20`；`VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1`。当时 DEP2 投影比 TP 好，实测更差（MoE kernel 选错）。Issue #30758 跟后续。系统 TPS ≠ 每用户 TPS。
