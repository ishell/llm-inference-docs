---
source: https://vllm.ai/blog/2025-08-19-glm45-vllm
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# GLM-4.5 / 4.5V：hybrid thinking，parser 叫 glm45，当时不要 V0

英文对照：[en/vllm/blog/serving/glm45.md](../../../../en/vllm/blog/serving/glm45.md)  
原文：https://vllm.ai/blog/2025-08-19-glm45-vllm  
355B/32B 与 Air 106B/12B。FP8/BF16 同一条 serve。后续 5.2 生产见 [glm52-b300](glm52-b300.md)。

`--tool-call-parser glm45` `--reasoning-parser glm45`。关思考：`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`。4.5V 要 `--allowed-local-media-path` 和 video `num_frames: -1`。8×H100 跑满血 4.5 当时可能 `--cpu-offload-gb 16`。FlashInfer 不顺就 `VLLM_ATTENTION_BACKEND=XFORMERS`。V0 不支持。4.5V grounding：`<|begin_of_box|>` 框坐标，xy 按宽高归一再 ×1000。
