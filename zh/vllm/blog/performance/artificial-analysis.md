---
source: https://vllm.ai/blog/2026-05-11-vllm-tops-artificial-analysis
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Artificial Analysis 榜：三只模型三只瓶颈，融合和 draft 都在 main

英文对照：`en/vllm/blog/performance/artificial-analysis.md`  
原文：https://vllm.ai/blog/2026-05-11-vllm-tops-artificial-analysis  
2026-05 那一周 DigitalOcean 测的。图在原网页。数字跟榜单当天走。

DeepSeek V3.2：低 batch 被 launch 钉死，attention 路径 ~33 kernel → ~10，batch=1 约 **1.28×**（85.8→109.3 tok/s，4×GB200，无 MTP）。单 8×B300 cc=1：无 MTP TP8 **125**；MTP=1 **234**（接受率 ~90%）；P/D TP4+TP4+MTP=3 **262**。router GEMM 再约 6%；indexer TopK 单 graph，128K decode 最高约 **17%** TPOT。MiniMax-M2.5：TorchSpec 训 EAGLE3 + `fuse_minimax_qk_norm`；天花板实验（100% 接受）TP4 **326 tok/s**。Qwen 3.5 397B：漏掉的 `allreduce_rms` 让 decode 一半时间耗在未融合跨卡 reduce；修完 + post-conv fusion + dual-stream，TEP=8 cc=1 **163 tok/s**，cc=256 **6.69→7.33 req/s**。系统 TPS ≠ 每用户 TPS。接 [gpt-oss-optimizations](gpt-oss-optimizations.md) / [qwen35-25k-tps](../serving/qwen35-25k-tps.md)。
