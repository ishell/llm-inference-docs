---
source: https://vllm.ai/blog/2025-09-05-beyond-text-generation
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# 文本之外：pooling 模型与 IO Processor

英文对照：`en/vllm/blog/architecture/beyond-text.md`  
原文：https://vllm.ai/blog/2025-09-05-beyond-text-generation  
图在原网页。和 [Omni](../serving/vllm-omni.md) 分清：这篇是 **非自回归、一次前向出多模态结果**（地学分割一类），不是 Thinker-Talker 流水线。

vLLM 从纯文本 LLM 走到 LLaVA 式「多模态进、文本出」。再往后：卷积 / ViT 一次推理出图或结构，不需要 detokenize。他们把这类模型当 **pooling**（identity pooler 吐 hidden），TerraTorch 当 generic backend（NASA/ESA 地学模型）。为此：无 attention 模型、可跳过 tokenizer、原始张量进、扩展 serving API。

张量进张量出还不够。Transformers processor 不懂 GeoTIFF。**IO Processor 插件**：进程外实现预/后处理，entry point 组 `vllm.io_processor_plugins`，启动 `--io-processor-plugin <name>`，作用在 `/pooling`。当时一只实例一只插件。Prithvi 洪水检测示例：`--model-impl terratorch --task embed --skip-tokenizer-init --io-processor-plugin prithvi_to_tiff`，URL 进 GeoTIFF、base64 出。这是 **输出形态** 的门，不是 PagedAttention 改写。
