---
source: https://vllm.ai/blog/2026-02-01-gpt-oss-optimizations
lang: en
fetched: 2026-09-01
---

# gpt-oss on Blackwell: push the Pareto, not a single TPS point

Chinese: `../../zh/vllm/blog/performance/gpt-oss-optimizations.md`  
gpt-oss-120b MXFP4 MoE, B200/GB200. Sibling: [blackwell-inferencemax](blackwell-inferencemax.md).

Max-throughput ~**+38%**, min-latency ~**+13%** — both ends of the curve. FlashInfer: `trtllm-gen` / CUTLASS MoE, FP8 KV attention. `torch.compile` fuses AR+RMSNorm instead of hardcoded fusion. Pad+Quant / Finalize+Slice still rolling, ~6% expected. GPU too fast, CPU cannot dispatch: async scheduling (default in later releases) ~**10%**; `--stream-interval` buffers later tokens (first token still immediate). On gpt-oss-20b @ 1024 concurrency they quote ~**57%** e2e — output-queue bottleneck, not a 57% kernel.

Recipe: `--cuda-graph-capture-size 2048`; high cc `--api-server-count 20` or `--stream-interval 20`; `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1`. DEP2 projected better than TP, measured worse (wrong MoE kernel). Track Issue #30758. System TPS ≠ per-user TPS.
