---
source: https://vllm.ai/blog/2025-11-30-vllm-omni
lang: en
fetched: 2026-09-04
---

# Announcing vLLM-Omni: Omni-Modality Model Serving

Chinese: [zh/vllm/blog/serving/vllm-omni.md](../../../../zh/vllm/blog/serving/vllm-omni.md)

2025-11-30. **vLLM-Omni Team**. Repo: [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni). Docs: [vllm-omni.readthedocs.io](https://vllm-omni.readthedocs.io/en/latest/). First cut on **vLLM v0.11.0** (package **v0.11.0rc**). Not “one LLM does everything” — **heterogeneous stages** are the building. Throughput vs Hugging Face Transformers is in their figure, not your SLA.

Later stage-splitting notes: [omni-diffusion-cache.md](omni-diffusion-cache.md), [omni-tts.md](omni-tts.md), [omni-layerwise-offload.md](omni-layerwise-offload.md). Same directory: [omni-autoround.md](omni-autoround.md), [qwen3-omni.md](qwen3-omni.md).

## Why vLLM-Omni?

vLLM started as high-throughput, memory-efficient **LLM** serving. The terrain moved: not just text-in, text-out. SOTA models reason across text, images, audio, and video, and emit heterogeneous outputs from diverse architectures.

**vLLM-Omni** is framed as one of the first open-source frameworks for omni-modality serving: extend vLLM’s performance to multimodal and non-autoregressive inference.

Local figures (copyright remains with the original site; study copies):

![omni modality model architecture](../../../../assets/vllm/blog/serving/vllm-omni/01-omni-modality-model-architecture.png)

Traditional serving engines were optimized for text Autoregressive (AR) work. As models become omni agents — seeing, hearing, speaking — the serving stack has to move. Three architectural shifts on the page:

1. **True omni-modality.** Process and generate text, image, video, and audio.
2. **Beyond autoregression.** Extend vLLM’s memory management to **Diffusion Transformers (DiT)** and other parallel generators.
3. **Heterogeneous pipeline.** One request can invoke several heterogeneous components: multimodal encoding, AR reasoning, diffusion-based multimodal generation, and so on.

## Inside the architecture

Not a wrapper. Data flow inside and beyond vLLM is re-thought: a fully disaggregated pipeline, with dynamic resource allocation across generation stages. The figure unifies three phases:

- **Modality Encoders:** multimodal inputs (ViT, Whisper, …)
- **LLM Core:** vLLM for autoregressive text and hidden states, one or more language models
- **Modality Generators:** high-performance serving for DiT and other decoding heads, rich media out

Those stages are the detachable **OmniStage**s later posts keep using.

### Key features

![vllm omni user interface](../../../../assets/vllm/blog/serving/vllm-omni/02-vllm-omni-user-interface.png)

- **Simplicity.** If you know vLLM, you know Omni. Hugging Face models; OpenAI-compatible API.
- **Flexibility.** The `OmniStage` abstraction folds Qwen-Omni, Qwen-Image, and other then-SOTA models into one stage story.
- **Performance.** Pipelined stages overlap compute: while one stage runs, others need not sit idle.

![vllm omni pipeline async stage](../../../../assets/vllm/blog/serving/vllm-omni/03-vllm-omni-pipeline-async-stage.png)

They benchmarked vLLM-Omni against Hugging Face Transformers for omni-modal serving efficiency. Numbers live in the figure; the body does not add a table.

![vllm omni vs hf](../../../../assets/vllm/blog/serving/vllm-omni/04-vllm-omni-vs-hf.png)

## Roadmap

Then: more models, further efficient inference, and a stable framework for omni-modality research.

- **Expanded model support:** more open-source omni models and diffusion transformers as they appear.
- **Adaptive framework:** evolve with new omni models and execution patterns — one foundation for production and research.
- **Deeper vLLM integration:** merge core omni features upstream so multimodality is first-class in the whole vLLM ecosystem.
- **Diffusion acceleration:** parallel inference (DP / TP / SP / USP…), cache (TeaCache / DBCache…), compute (quantization / sparse attention…). Later landing: [omni-diffusion-cache.md](omni-diffusion-cache.md).
- **Full disaggregation:** via OmniStage, split encoder / prefill / decode / generation for throughput and latency.
- **Hardware:** follow the [hardware plugin](../architecture/hardware-plugin.md) so Omni is not glued to one vendor.

## Getting started: install and serve

Initial **vllm-omni v0.11.0rc**, built on **vLLM v0.11.0**. The post does **not** inline a `pip install`; install steps live in the docs:

- [Installation](https://vllm-omni.readthedocs.io/en/latest/getting_started/installation/)

Serving is not one universal CLI in this post either. The repo `examples` directory has scripts for image, audio, and video generation:

- [examples](https://github.com/vllm-project/vllm-omni/tree/main/examples)

Gradio is there for a gentler UX. The demo on the page is serving **Qwen-Image**:

![vllm omni gradio serving demo](../../../../assets/vllm/blog/serving/vllm-omni/05-vllm-omni-gradio-serving-demo.png)

CLI flags and `Omni(...)` knobs belong to the then-current docs and examples. Later TTS / cache / offload notes grow staged commands — do not back-port those into this announcement’s “first command”.

## Join the community

The start of omni-modality serving. The page asks the community to help pick the next architectures.

- Code and docs: [GitHub](https://github.com/vllm-project/vllm-omni) · [Documentation](https://vllm-omni.readthedocs.io/en/latest/)
- Slack: `#sig-omni` at [slack.vllm.ai](https://slack.vllm.ai)
- Weekly meeting: Tuesdays **19:30 PDT**. [Join](https://tinyurl.com/vllm-omni-meeting)
