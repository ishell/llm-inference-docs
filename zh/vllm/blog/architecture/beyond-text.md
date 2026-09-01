---
source: https://vllm.ai/blog/2025-09-05-beyond-text-generation
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# 文本之外：pooling 模型把图吐回来

英文对照：`en/vllm/blog/architecture/beyond-text.md`  
原文：https://vllm.ai/blog/2025-09-05-beyond-text-generation  
2025-09-05。IBM / TerraTorch。图在原网页。后来的多模态流水线见 [vllm-omni](../serving/vllm-omni.md)；插件面见 [plugin-system](plugin-system.md)。

vLLM 先会文本进文本出，再会 LLaVA 式多模态进、文本出。这篇走第三步：**非自回归、一次前向吐出多模态输出**——推理形态像 pooling，但输入输出要自己处理。落地是地理空间基础模型（多光谱 / 雷达 + 元数据），TerraTorch 整族经 generic backend 进 vLLM。

为了让这些模型站住：无 attention 的模型、不需要 tokenizer、原始输入而不是默认 multimodal embedding、serving API 加长。Identity pooler 把 hidden 原样交出来。

## IO Processor

光 pooling 只做到 tensor↔tensor。GeoTIFF 一类 transformers processor 不会。**IO Processor 插件**：引擎外实现接口，`vllm.io_processor_plugins` entry point，`--io-processor-plugin <name>`。当时每个实例一只插件，挂在 `/pooling`。

Prithvi 洪水检测示例：下载 GeoTIFF → 切成 512×512 补丁（6 波段 + GPS/日期）→ 多条 prompt → 推理 → 按元数据拼回 GeoTIFF。

```bash
vllm serve ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
  --model-impl terratorch --task embed --skip-tokenizer-init --enforce-eager \
  --io-processor-plugin prithvi_to_tiff
```

请求打 `http://localhost:8000/pooling`，`softmax: false` 才能拿到生输出。当时要装最新主干（尚未进 v0.10.1.1）。
