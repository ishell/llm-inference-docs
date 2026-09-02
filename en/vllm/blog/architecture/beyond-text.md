---
source: https://vllm.ai/blog/2025-09-05-beyond-text-generation
lang: en
fetched: 2026-09-01
---

# Beyond text generation

2025-09-05. IBM / TerraTorch.  Later multimodal pipeline: [vllm-omni.md](../serving/vllm-omni.md). Plugins: [plugin-system.md](plugin-system.md).

vLLM grew from text→text to LLaVA-style multimodal-in / text-out. This is the third step: **non-autoregressive models that emit multimodal output in one pass** — pooling-shaped inference with custom I/O. First landing: geospatial foundation models (multispectral/radar + metadata) via a generic TerraTorch backend.

Needed: attention-free models, no tokenizer, raw tensors instead of default multimodal embeddings, serving API extensions. Identity pooler returns hidden states unchanged.

Tensor↔tensor is not enough (GeoTIFF). **IO Processor** plugins live outside the tree, register `vllm.io_processor_plugins`, `--io-processor-plugin <name>`. Then one plugin per instance, applied on `/pooling`.

Prithvi flood example: fetch GeoTIFF → 512×512 patches (6 bands + GPS/date) → prompts → stitch with metadata.

```bash
vllm serve ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
  --model-impl terratorch --task embed --skip-tokenizer-init --enforce-eager \
  --io-processor-plugin prithvi_to_tiff
```

POST `/pooling` with `softmax: false` for raw output. Needed trunk newer than v0.10.1.1 at the time.

Local figures (copyright remains with the original site; study copies):

![models diff](../../../../assets/vllm/blog/architecture/beyond-text/01-models-diff.png)

![io plugins flow](../../../../assets/vllm/blog/architecture/beyond-text/02-io-plugins-flow.png)

![prithvi prediction](../../../../assets/vllm/blog/architecture/beyond-text/03-prithvi-prediction.png)
