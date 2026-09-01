---
source: https://vllm.ai/blog/2025-08-19-glm45-vllm
lang: en
fetched: 2026-09-01
---

# GLM-4.5 / 4.5V: hybrid thinking, glm45 parsers, V0 not supported then

Chinese: `../../zh/vllm/blog/serving/glm45.md`  
355B/32B and Air 106B/12B. FP8/BF16 same serve. Later 5.2 production: [glm52-b300](glm52-b300.md).

`--tool-call-parser glm45` `--reasoning-parser glm45`. Disable thinking: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`. 4.5V needs `--allowed-local-media-path` and video `num_frames: -1`. Full 4.5 on 8×H100 then often `--cpu-offload-gb 16`. FlashInfer issues: `VLLM_ATTENTION_BACKEND=XFORMERS`. V0 unsupported. 4.5V grounding: `<|begin_of_box|>` boxes, xy normalized by W/H then ×1000.
