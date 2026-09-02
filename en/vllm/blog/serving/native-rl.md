---
source: https://vllm.ai/blog/2026-05-28-native-rl-apis
lang: en
fetched: 2026-09-01
---

# Native RL APIs

2026-05-28. Docs: weight transfer, async RL. `examples/rl`. HTTP needs `VLLM_SERVER_DEV_MODE=1`. Study note.

Two recurring pains: every RL framework patched vLLM workers to sync weights; async RL deadlocked on P/D and DPEP.

**Weight transfer** (`WeightTransferEngine`): init → start_update → update (chunkable) → finish (e.g. quant). Backends: NCCL broadcast, CUDA IPC. Packed tensors optional. Register your own engine. Etha-style sharded transfer as a prototype; later RDT is the large-scale cousin.

**Pause `keep`:** besides abort (client retries) and wait (no overlap). Keep freezes the scheduler, preserves in-flight requests, still async. `POST /pause` / `/resume`.

**DPEP deadlock:** pause lived in `AsyncLLM` while DP waves lived in EngineCore. Fix: pause in EngineCore; two-phase — local pause still honors `START_DP_WAVE` to finish the in-flight forward; periodic all-reduce (every 32 steps) enters global pause together, then weights move.

SkyRL: Qwen3-1.7B DAPO over HTTP. Prime-RL: GLM-5.1-FP8, 16×8×H200, 2×(4P+4D) DPEP32, 1 TB CPU KV/node, vllm-router sticky, 100+ stable steps. Sleep Mode swaps models without killing the process; this swaps **new weights of the same model**.

Local figures (copyright remains with the original site; study copies):

![rl system overview](../../../../assets/vllm/blog/serving/native-rl/01-rl_system_overview.png)

![weight transfer nccl](../../../../assets/vllm/blog/serving/native-rl/02-weight_transfer_nccl.svg)

![async rl](../../../../assets/vllm/blog/serving/native-rl/03-async_rl.svg)

![dp generate](../../../../assets/vllm/blog/serving/native-rl/04-dp_generate.svg)

![vllm deadlock](../../../../assets/vllm/blog/serving/native-rl/05-vllm_deadlock.svg)

![vllm no deadlock](../../../../assets/vllm/blog/serving/native-rl/06-vllm_no_deadlock.svg)

![skyrl validation](../../../../assets/vllm/blog/serving/native-rl/07-skyrl_validation.svg)

![prime rl](../../../../assets/vllm/blog/serving/native-rl/08-prime_rl.svg)
