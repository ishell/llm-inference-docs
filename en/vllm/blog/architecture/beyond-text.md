---
source: https://vllm.ai/blog/2025-09-05-beyond-text-generation
lang: en
fetched: 2026-09-01
---

# Beyond text: pooling models and IO Processors

Chinese: `../../zh/vllm/blog/architecture/beyond-text.md`  
Not [Omni](../serving/vllm-omni.md): this is **non-autoregressive, one-pass multimodal output** (geospatial segmentation), not a Thinker-Talker pipeline.

vLLM went from text LLMs to LLaVA-style multimodal-in, text-out. Next: conv / ViT models that emit an image or structure in one pass — no detokenize. They land as **pooling** (identity pooler returns hidden) with a TerraTorch generic backend (NASA/ESA geospatial models). Needed: attention-free models, skip tokenizer, raw tensors in, serving API extensions.

Tensor-in/tensor-out is not enough. Transformers processors do not speak GeoTIFF. **IO Processor plugins** live out-of-tree, register in `vllm.io_processor_plugins`, start with `--io-processor-plugin <name>`, wrap `/pooling`. One plugin per instance then. Prithvi flood example: `--model-impl terratorch --task embed --skip-tokenizer-init --io-processor-plugin prithvi_to_tiff`, URL in, base64 GeoTIFF out. This is the **output-shape** door, not a PagedAttention rewrite.
