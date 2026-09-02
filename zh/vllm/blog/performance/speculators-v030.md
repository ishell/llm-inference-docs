---
source: https://vllm.ai/blog/2025-12-13-speculators-v030
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Speculators v0.3.0：把 EAGLE-3 草稿训出来

英文对照：`en/vllm/blog/performance/speculators-v030.md`  
原文：https://vllm.ai/blog/2025-12-13-speculators-v030  
后来的 DFlash / 在线训练见 [v0.5.0](speculators-v050.md)；hidden 导出见 [extract-hidden-states](../architecture/extract-hidden-states.md)。

投机解码要 **每只 verifier 一只草稿**。v0.3 把离线数据 → 训练 → `vllm serve` 串起来。数据：三层 hidden、token id、只在 assistant span 上的 loss mask、verifier 分布。自定义 worker 在 prefill 截 hidden，`.pt` 异步落盘；`token_freq.pt` 做缩小的 draft vocab（t2d/d2t）。训练用 Eagle3 的 train-time-testing + FlexAttention 稀疏 mask，序列沿 seq 维拼接而不是狂 padding。

产物 `config.json` 带 `speculators_config`：verifier 路径、算法、默认 N。短命令：

```
vllm serve RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3
```

长命令可换量化 verifier、改 `num_speculative_tokens`。当时训/服：Llama 3.x、Qwen3 dense/MoE、GPT-OSS；Llama 4 视觉只 serving。官方说草稿对齐时低负载大约 **1.5–3×** 延迟——不是承诺。

本地图（原文版权仍归原站；学习对照用）：

![data generation](../../../../assets/vllm/blog/performance/speculators-v030/01-data_generation.png)

![hidden state generator](../../../../assets/vllm/blog/performance/speculators-v030/02-hidden_state_generator.png)

![flex attention](../../../../assets/vllm/blog/performance/speculators-v030/03-flex_attention.png)

![EAGLE3](../../../../assets/vllm/blog/performance/speculators-v030/04-EAGLE3.png)
