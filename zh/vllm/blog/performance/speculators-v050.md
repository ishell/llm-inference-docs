---
source: https://vllm.ai/blog/2026-05-28-speculators-v050
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Speculators v0.5.0：DFlash 与在线 hidden

英文对照：`en/vllm/blog/performance/speculators-v050.md`  
原文：https://vllm.ai/blog/2026-05-28-speculators-v050  
vLLM ≥0.20.0（PR#38300）。

EAGLE-3 多步自回归；DFlash **一次前向一块**，块内非因果。训练若每个位置都开预测块，attention mask 会炸。他们只在 loss 位置随机抽 **anchor**，块数与序列长度脱钩。训练参数：`--speculator-type dflash`、`--block-size`、`--max-anchors`。Gemma 4 31B 的 DFlash 在推理/代码任务接受率更好；ITL 优于 EAGLE-3 和单独 FP8 verifier，再叠 FP8 verifier 更短。

```
vllm serve -tp 2 RedHatAI/gemma-4-31B-it-speculator.dflash
```

Hidden 不再钩 vLLM 内部 Python API，改走 [extract-hidden-states](../architecture/extract-hidden-states.md) 的 HTTP 路径：在线边训边抽，离线先缓存。两种模式同一数据格式，可以混用（先离线一部分，缺的在线补）。和 [并行草稿](parallel-drafting.md) 一起读。

本地图（原文版权仍归原站；学习对照用）：

![gemma4 dflash acceptance rates](../../../../assets/vllm/blog/performance/speculators-v050/01-gemma4-dflash-acceptance-rates.png)

![gemma4 dflash latency](../../../../assets/vllm/blog/performance/speculators-v050/02-gemma4-dflash-latency.png)
