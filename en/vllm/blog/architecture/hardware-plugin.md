---
source: https://vllm.ai/blog/2025-05-12-hardware-plugin
lang: en
fetched: 2026-09-01
---

# Hardware Plugin

2025-05-12. Hardware Pluggable RFC with Ascend since Dec 2024. Proof: `vllm-ascend`, `vllm-spyre`. Study note.

More backends scattered Executor/Worker/Runner/Attention through core. Plugins keep **Platform** in a separate package. Base RFCs: general plugins; device-agnostic `platform` (+ #6080). Then Platform as plugin; refactor Executor, Worker, ModelRunner, AttentionBackend, Communicator.

Dev: `platform.py` from `vllm.platforms.Platform`; optional Worker/ModelRunner/AttentionBackend/Communicator bases; `setup.py` `vllm.platform_plugins` entry point whose register returns the Platform path. User: `pip install vllm vllm-ascend`. Log: `plugin ascend loaded`.

Then: V1/VLM, scheduler/graph/custom-op plugin surfaces. Slack `#sig-extensible-hardware`. [torch-compile.md](torch-compile.md) takes optimization out of model files; this takes hardware out of core.
