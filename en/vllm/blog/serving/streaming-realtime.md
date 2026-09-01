---
source: https://vllm.ai/blog/2026-01-31-streaming-realtime
lang: en
fetched: 2026-09-01
---

# Streaming input and `/v1/realtime`

Chinese: `../../zh/vllm/blog/serving/streaming-realtime.md`  
PR #28973.

Outputs can already SSE. Inputs often wait for the full prompt — speech, simultaneous translation, realtime agents cannot. This post wires **streamable inputs** and an OpenAI-style **`/v1/realtime` WebSocket** into vLLM.

## The model must be causal

Bidirectional attention cannot stream: future frames are missing but positions already see each other. Causal + sliding window can increment; the model still needs streaming-aligned training. Voxtral-class models are built this way. Dumping an arbitrary causal LLM into the realtime socket does not make it a streaming ASR.

A full prompt can still use **chunked prefill**: chunks enter the engine and TTFT is often shorter than “wait then one prefill”. That is scheduler chunking, not “tokens arriving on the wire”.

## API

`StreamingInput`: `prompt` plus optional `sampling_params`. Do not pass a complete string to `AsyncLLM.generate()` — pass an `AsyncGenerator` that yields `StreamingInput`. Empty list ends the input. Examples use `max_tokens=1` to emit as chunks arrive; production sampling follows the model.

HTTP: `/v1/realtime` WebSocket. Event names and OpenAI Realtime deltas: original post and docs of that week — the surface is still growing.

Read with [Anatomy](anatomy.md) prefill/decode and the [V1](v1.md) scheduler: this changes **how requests enter the engine**, not the attention formula.
