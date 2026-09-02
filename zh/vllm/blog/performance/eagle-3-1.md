---
source: https://vllm.ai/blog/2026-05-26-eagle-3-1
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# EAGLE 3.1：压住 attention drift

英文对照：`en/vllm/blog/performance/eagle-3-1.md`  
原文：https://vllm.ai/blog/2026-05-26-eagle-3-1  
vLLM nightly / 当时即将 v0.22.0。Kimi K2.6 NVFP4、TP4、GB200、非分离、SPEED-Bench coding：**c=1 约 2.03×** 每用户输出 TPS，c=4 约 1.71×，c=16 约 1.66×。

聊天模板、长上下文、OOD system prompt 会让 EAGLE-3 接受长度掉。原因叫 **attention drift**：猜得越深，草稿注意力离开 sink token、盯住自己刚吐的字。两处：融合输入里高层 hidden 越来越霸道；未归一化 residual 让 hidden 幅度跨步膨胀。

3.1：每路 target hidden 进 FC **之前** 做 FC normalization；下一步吃 **post-norm** hidden，更像递归调草稿而不是往 target 后面叠层。长上下文接受长度相对 EAGLE-3 最多约 **2×**。仍走 `method: eagle3`，旧 checkpoint 能用。

```
--speculative-config '{"model":"lightseekorg/kimi-k2.6-eagle3.1-mla","method":"eagle3","num_speculative_tokens":3}'
```

训练侧 TorchSpec。和 [spec-decode](spec-decode.md)、[P-EAGLE](p-eagle.md) 一起读。

本地图（原文版权仍归原站；学习对照用）：

![pre norm vs post norm](../../../../assets/vllm/blog/performance/eagle-3-1/01-pre-norm-vs-post-norm.png)

![tpot baseline vs eagle31](../../../../assets/vllm/blog/performance/eagle-3-1/02-tpot_baseline_vs_eagle31.png)
