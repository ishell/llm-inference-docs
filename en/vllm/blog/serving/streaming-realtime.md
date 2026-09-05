---
source: https://vllm.ai/blog/2026-01-31-streaming-realtime
lang: en
fetched: 2026-09-05
---

# Streaming Requests & Realtime API in vLLM

Chinese: [zh/vllm/blog/serving/streaming-realtime.md](../../../../zh/vllm/blog/serving/streaming-realtime.md)

2026-01-31. **Meta, Mistral AI as well as the vLLM team**. Streaming input: [PR #28973](https://github.com/vllm-project/vllm/pull/28973). Realtime WebSocket API: [PR #33187](https://github.com/vllm-project/vllm/pull/33187). How-to pointers on the page: [streaming input tests](https://github.com/vllm-project/vllm/tree/main/tests/v1/streaming_input), [Realtime API docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/?h=realtime+api#realtime-api). Read with [Anatomy](../architecture/anatomy.md) Prefill/Decode and the [V1](../architecture/v1-alpha.md) scheduler: this changes **how requests enter the engine**, not the attention formula. No local figures (the post’s cover was not fetched).

Traditional LLM inference takes a **complete** prompt, runs the model, then returns a response (streamed or at once). That fits text chatbots and batch jobs. It does not fit realtime audio, video, robotics, or anything that must start work before the last input byte arrives. vLLM added **streamable inputs** in the engine and a **Realtime WebSocket API** on top, exposing `/v1/realtime`.

Output streaming (SSE / token emit) is old. Input was fixed: the entire request had to be present before inference began. These two features unlock incremental input.

## Why realtime is needed

### The traditional batch paradigm

The user submits a full request, e.g. via a [`ChatCompletionRequest`](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create), waits for the model to process it entirely, then gets the complete response. vLLM has long supported *output* streaming. The *input* side stayed complete-prompt-only.

That is enough for chatbots, document summarization, and code generation. A growing class of apps cannot wait for complete input.

### Why streaming matters

Voice assistant that drives a computer or phone: microphone audio streams into an LLM that must process continuously and emit actions in realtime. Latency here is [TTFT](https://www.emergentmind.com/topics/time-to-first-token-ttft). The post’s bar: the user does not want to wait more than a second to open an app or type into search. Natural duplex voice needs listen and speak **at the same time**.

Can you fake streaming with a non-streaming LLM by buffering chunks? In principle: segment audio, process each segment, concatenate outputs. In practice:

- Sub-second TTFT needs highly performant **chunk detection** — where to cut so no relevant information is lost.
- Bad cuts raise TTFT or hurt quality by fragmenting temporal context.
- Chunk-then-respond is still turn-based: each chunk must finish before a reply. No true listen-and-speak overlap.

The same gap shows up in:

- **Voice assistants** — sub-second response to feel natural.
- **Live transcription** — text as speech is recognized.
- **Robotics / embodied AI** — cameras, microphones, LIDAR; control actions with minimal delay.

Batch waiting is the wrong delay. Infrastructure must process input incrementally and start generating **before all input has arrived**.

Even for apps that still need the full input before the first output token, streaming the prompt as it becomes available can still help TTFT. Default vLLM uses [chunked prefill](https://docs.vllm.ai/en/stable/cli/serve/?h=max+num+b#-enable-chunked-prefill-no-enable-chunked-prefill): an input of \(N\) tokens is processed in \(N \div M\) forward passes, \(M\) = [`max_num_batched_tokens`](https://docs.vllm.ai/en/stable/cli/serve/?h=max+num+b#-max-num-batched-tokens). When \(N \div M > 1\), feeding tokens as they arrive lets the **first** Prefill forward pass schedule earlier.

That is **not** the same knob as streaming input:

| | Chunked prefill | Streaming input |
| --- | --- | --- |
| What is split | A **complete** prompt the scheduler already holds | Tokens / audio **arriving on the wire** |
| Who splits | Scheduler (`max_num_batched_tokens`) | Client / `AsyncGenerator` yielding `StreamingInput` |
| Typical win | Interleave Prefill with Decode; bound a giant Prefill | Start Prefill before the last chunk exists; duplex |

### Requirements for streaming

Not every model can do true streaming. Two requirements: the right attention pattern, and training for incremental processing.

#### Attention patterns

- **Causal attention** (uni-directional mask): position \(t\) attends only to \(j \le t\). Future tokens are excluded, so the output at \(t\) is final once token \(t\) arrives. True streaming is possible; earlier outputs never need revision because of future input.
- **Bidirectional attention** (full mask): every position attends to past **and** future. Output at \(t\) is conditioned on tokens that may not have arrived. Until the full sequence is known, no position has a stable output. Incompatible with streaming / online processing.

For long-running or **infinite** streams, plain causal attention is not enough: attending to the entire past grows compute and memory without bound. Practice truncates past context. Common fix: **sliding-window attention** — a fixed window of recent tokens, bounded compute and memory. Causal + sliding window is the usual architecture for modern streaming models.

#### Training for streaming inputs

A streamable architecture is not enough. The model must be **trained** for true streaming input.

Let \(X = (x_0, \ldots, x_T)\) be the input sequence and \(Y = (y_0, \ldots, y_{T'})\) the output. In streaming, \(y_t\) should correspond to \(x_t\) at step \(t\) with as little latency as possible (e.g. transcription of an audio frame streamed at time \(t\)).

Standard next-token training conditions on the **entire** input:

\[
P(y_i \mid y_{i-1}, \ldots, y_0, x_T, x_{T-1}, \ldots, x_0).
\]

That cannot run in realtime: generating \(y_i\) requires full \(X\).

A streaming model must predict from past inputs and, optionally, a small lookahead \(\delta\):

\[
P(y_i \mid y_{i-1}, \ldots, y_0, x_{i+\delta}, \ldots, x_i, \ldots, x_0),
\]

\(\delta\) as small as possible. In theory \(\delta = 0\); in practice a small delay is usually needed for quality.

Training therefore needs:

- **i)** align input and output so \(T' = T\), and each \(y_i\) is the correct output for \(x_i\);
- **ii)** an architecture that can process new \(x_{i+1}\) while \(x_i, \ldots, x_0\) have already been processed.

[Delayed Streams Modeling](https://arxiv.org/abs/2509.08753) (picked up by Voxtral-Realtime; the post left that second link as `TODO`) sum-pools input embeddings (e.g. speech) and output embeddings (e.g. text) into one sequence, then predicts

\[
P(y_i \mid y'_{i-1}, \ldots, y'_0), \qquad y'_k = y_k + x_{k+\delta}.
\]

You cannot take an arbitrary causal LLM, hang it on `/v1/realtime`, and expect streaming ASR quality. It has to be trained with that alignment and those architectural constraints.

### Why architecture matters for serving

vLLM can serve any model. **True** streaming needs architecturally causal models. [Voxtral](https://mistral.ai/news/voxtral) is designed for streaming with causal attention that supports incremental processing.

Serving must also accept incremental input. A streaming-capable model behind a “complete prompt first” server loses the latency win. That is why vLLM now streams **input** as well as output.

### Further reading on streaming architecture

- [Transformer Transducer](https://arxiv.org/abs/2002.02562) — streamable speech recognition.
- [Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling](https://arxiv.org/abs/2509.08753) (Kyutai) — the architecture sketched above.
- [Streaming Simultaneous Speech Translation with Augmented Memory Transformer](https://arxiv.org/abs/2011.00033) — translation is less “monotonic” than speech, so performant streaming is harder.
- [Voxtral](https://mistral.ai/news/voxtral) / Voxtral-Realtime — massively pretrained, open-sourced, competitive with most offline speech recognition models.

## Streaming input in vLLM

[PR #28973](https://github.com/vllm-project/vllm/pull/28973) adds streaming input: chunks arrive over time; output is generated continuously.

### The `StreamingInput` interface

```python
from dataclasses import dataclass
from vllm.inputs import PromptType
from vllm.sampling_params import SamplingParams

@dataclass
class StreamingInput:
    prompt: PromptType
    sampling_params: SamplingParams | None = None
```

Do not pass a fixed prompt to `AsyncLLM.generate()`. Pass an `AsyncGenerator` that yields `StreamingInput` objects. Each one is the next chunk appended to a cumulative prompt.

```python
import asyncio
from vllm.inputs.data import StreamingInput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams

async def streaming_input_example():
    async_llm = AsyncLLM.from_engine_args(...)

    # Input queue can consume inputs in a separate async task
    input_queue = asyncio.Queue[list[int]]()

    async def input_generator():
        # Loop until empty list encountered => input finished
        while new_tokens := input_queue.get():
            yield StreamingInput(prompt=new_tokens)

    output_generator = async_llm.generate(
        prompt=input_generator(),
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
    )

    async for output in output_generator:
        # ...

asyncio.run(streaming_input_example())
```

You may wait until the output for the last input finishes before sending the next chunk; you do not have to — chunks are queued internally. End the input stream by exiting the async generator or closing it with `aclose`. The **output** generator does not complete until every received input has been processed **and** the input generator has completed.

Examples use `max_tokens=1` so each resumable request mostly computes KV for `prompt_token_ids` and leaves the single `output_token_ids` token to the application. Production sampling follows the model.

### How it works

Each chunk is treated as a separate request with a **cumulative** prompt. As new chunks arrive, the engine:

1. Extends `prompt_token_ids` with `max_tokens - 1` generated `output_tokens` **and** the new incoming `prompt_token_ids`.
2. Reuses all cached KV.
3. Generates output tokens from the current cumulative prompt and the indicated `max_tokens`.
4. Optionally discards output when new input arrives.

Tokens emitted **between** input chunks may be revised as more context arrives. The final output reflects the complete input.

Implementation is a **sticky session**. The first chunk creates an **anchor request** that lasts the session. Later chunks with the same internal request ID are queued and processed in order.

### The Anchor Request pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STREAMING SESSION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User's AsyncGenerator              Scheduler                              │
│   ═══════════════════               ═════════                               │
│                                                                             │
│   ┌──────────────┐                                                          │
│   │   Chunk 1    │ ──────────────►  Add ANCHOR REQUEST                      │
│   │   [A, B, C]  │                  ┌────────────────────────────────┐      │
│   └──────────────┘                  │  Request (id="session_1")      │      │
│                                     │  ├── resumable: true           │      │
│                                     │  ├── max_tokens: 2             │      │
│                                     │  ├── streaming_queue: deque()  │      │
│                                     │  ├── status: RUNNING           │      │
│                                     │  └── prompt_token_ids: [A,B,C] │      │
│                                     └────────────────────────────────┘      │
│                                              │                              │
│                                              ▼                              │
│   ┌──────────────┐                  ┌────────────────┐                      │
│   │   Chunk 2    │                  │    ENGINE      │  Generating...       │
│   │   [D, E]     │ ─────┐           │  Processing    │  ──► Output: [X, Y]  │
│   └──────────────┘      │           └────────────────┘                      │
│                         │                                                   │
│                         ▼           Anchor busy? Queue it!                  │
│   ┌──────────────┐      │           ┌────────────────────────────────┐      │
│   │   Chunk 3    │      └────────►  │  streaming_queue:              │      │
│   │   [F, G]     │ ─────────────►   │  ┌───────┐ ┌───────┐           │      │
│   └──────────────┘                  │  │[D, E] │→│[F, G] │→ ...      │      │
│                                     │  └───────┘ └───────┘           │      │
│                                     └────────────────────────────────┘      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                     WHEN ANCHOR FINISHES CURRENT CHUNK                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Engine signals: chunk complete (stopped = True)                           │
│          │                                                                  │
│          ▼                                                                  │
│   ┌────────────────────────────────────────────────────────────────┐        │
│   │  _handle_stopped_request() pops first item from queue          │        │
│   │                                                                │        │
│   │  streaming_queue: [[D,E], [F,G]]  ──►  [[F,G]]                 │        │
│   │                      ▲                                         │        │
│   │                      │                                         │        │
│   │                    pop!                                        │        │
│   └──────────────────────┬─────────────────────────────────────────┘        │
│                          │                                                  │
│                          ▼                                                  │
│   ┌────────────────────────────────────────────────────────────────┐        │
│   │  _update_request_as_session(anchor, update=[D, E])             │        │
│   │                                                                │        │
│   │  BEFORE:                       AFTER:                          │        │
│   │  ┌───────────────────────┐     ┌───────────────────────────┐   │        │
│   │  │ prompt_token_ids:     │     │ prompt_token_ids:         │   │        │
│   │  │   [A, B, C]           │     │   [A, B, C, X, D, E]      │   │        │
│   │  │ _output_token_ids:    │ ──► │ _output_token_ids:        │   │        │
│   │  │   [X, Y]              │     │   []                      │   │        │
│   │  │ _all_token_ids:       │     │ _all_token_ids:           │   │        │
│   │  │   [A, B, C, X, Y]     │     │   [A, B, C, X, D, E]      │   │        │
│   │  │ num_computed_tokens: 4│     │ num_computed_tokens: 4    │   │        │
│   │  │ status: RUNNING       │     │ status: WAITING           │   │        │
│   │  └───────────────────────┘     └───────────────────────────┘   │        │
│   │                                                                │        │
│   │  Note: Y is DISCARDED (last sampled token, not yet computed)   │        │
│   │        Only X is kept (num_computed_tokens = 4, so [A,B,C,X])  │        │
│   └────────────────────────────────────────────────────────────────┘        │
│                          │                                                  │
│                          ▼                                                  │
│   ┌────────────────────────────────────────────────────────────────┐        │
│   │  Anchor returns to waiting queue → scheduled again → ENGINE    │        │
│   └────────────────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why discard the last token (Y) on the next `prompt_token_ids`?**

On a resumable request, the engine needs KV for all `prompt_token_ids` plus `max_tokens - 1` generated tokens. The user marks those `max_tokens - 1` generated tokens as **final** and reusable. The last entry of `output_token_ids` is only the result of the most recent forward pass: it does **not** yet have KV. The application can still look at it; it is dropped from the updated anchor’s `prompt_token_ids`. Discarding it is “free”: no cached state is invalidated, and it would have to be recomputed anyway if kept.

The [Realtime API](https://github.com/vllm-project/vllm/blob/a2443de5fa4a0605607f6c3d9219022c7f6ac480/vllm/entrypoints/openai/realtime/connection.py#L209) (and most apps) set `max_tokens=1`: each resumable request computes KV only for `prompt_token_ids`; the single generated token is the app’s to use. For [Voxtral Realtime](https://mistral.ai/news/voxtral), that token is combined with the next audio chunk to form the next resumable request.

**Caveat:** some models emit special stop tokens that must be present to continue generation. Scheduling then needs **+1 token** to recompute the stop token before the new input chunk.

### Example flow

Multiple resumable requests can stream with **different** `max_tokens`. The post’s illustration:

```
Input chunks: ([A1, B1, C1], max_tokens=1), ([A2, B2], max_tokens=2), ([A3], max_tokens=2)

1. First chunk [A1, B1, C1]
   -> Model generates [D1]

2. Second chunk [A2, B2]
   -> Cumulative prompt: [A1, B1, C1, A2, B2] (D1 discarded)
   -> Model generates [C2, D2, E2]

3. Third chunk [A3]
   -> Cumulative prompt: [A1, B1, C1, A2, B2, C2, D2, A3] (E2 discarded)
   -> Model generates [C3, D3]

Output stream: D1, C2, D2, E2, C3, D3
```

## Realtime API with WebSockets

Streaming input is the engine capability. Production apps want a socket. [PR #33187](https://github.com/vllm-project/vllm/pull/33187) adds a WebSocket Realtime API inspired by [OpenAI’s Realtime API](https://platform.openai.com/docs/guides/realtime). Endpoint: `/v1/realtime`.

### Architecture

Bidirectional streaming between clients and the vLLM server. Clients send audio; the server responds with transcribed text and model outputs. Four pieces:

1. **WebSocket Client** — capture microphone (or file) audio, send chunks.
2. **Realtime Handler** — receive WebSocket messages, convert to `StreamingInput`.
3. **AsyncLLM** — process streaming input, generate output.
4. **Response Stream** — send generated tokens back on the same WebSocket.

### Server setup

```bash
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --enforce-eager
```

WebSocket: `ws://localhost:8000/v1/realtime`.

### Client example

Stream an audio file, print transcription. Expected audio: **PCM16 @ 16 kHz**. Chunks: **4 KB**. The snippet on the page uses an undefined `model` in `session.update` — copied as published.

```python
import asyncio
import base64
import json
import librosa
import numpy as np
import websockets

def load_audio_as_pcm16(audio_path: str) -> bytes:
    """Load audio file and convert to PCM16 @ 16kHz."""
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    return (audio * 32767).astype(np.int16).tobytes()

async def stream_audio_file(audio_path: str, server_url: str = "ws://localhost:8000/v1/realtime"):
    async with websockets.connect(server_url) as ws:
        response = json.loads(await ws.recv())

        pcm_audio = load_audio_as_pcm16(audio_path)

        await ws.send(json.dumps({"type": "session.update", "model": model}))

        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        for i in range(0, len(pcm_audio), 4096):
            chunk = pcm_audio[i:i + 4096]
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode()
            }))

        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        async for message in ws:
            data = json.loads(message)
            if data["type"] == "transcription.delta":
                print(data["delta"], end="", flush=True)
            elif data["type"] == "transcription.done":
                break

asyncio.run(stream_audio_file("audio.wav"))
```

Workflow on the page:

- Load / convert audio to PCM16 @ 16 kHz (the format the API expects).
- Connect to `/v1/realtime`; send `session.update` to validate the model.
- Stream audio in 4 KB chunks as `input_audio_buffer.append`; `input_audio_buffer.commit` marks start and end (`final: True` on the closing commit).
- Print `transcription.delta` until `transcription.done`.
- **Realtime caveat of this sample:** it sends **all** audio before listening. The protocol itself is fully asynchronous — send chunks and receive transcription concurrently. A production service would start transcribing as the first chunk arrives.

### Message types

Every type **named** in the post (the example and the catalog disagree in places; both lists are below).

**Client → server** (catalog):

- `session.create` — initialize a new session
- `input_audio_buffer.append` — send audio data
- `input_audio_buffer.commit` — signal end of audio input (example also uses it at stream **start**, and with `"final": True` at the end)
- `response.create` — request model response

**Server → client** (catalog):

- `session.created` — session initialization confirmed
- `response.text.delta` — incremental text (TTS / generation path)
- `response.audio.delta` — incremental audio (TTS models)
- `response.done` — response complete
- `error` — error occurred

**Also used in the client sample** (not repeated in the catalog):

- `session.update` — validate / set model after connect
- `transcription.delta` — partial transcription text
- `transcription.done` — transcription finished

The surface was still growing when the post shipped; event names vs OpenAI Realtime can drift — check that week’s docs.

### Example scripts

In the vLLM tree:

- [`examples/online_serving/openai_realtime_client.py`](https://docs.vllm.ai/en/latest/examples/online_serving/openai_realtime_client/?h=realtime#openai-realtime-client) — basic WebSocket client
- [`examples/online_serving/openai_realtime_microphone_client.py`](https://docs.vllm.ai/en/latest/examples/online_serving/openai_realtime_microphone_client/#openai-realtime-microphone-client) — microphone integration

Those examples capture system microphone audio and stream it to vLLM in real time.

### Performance considerations

A dedicated `AsyncGenerator` session keeps the session KV **as-is**. That is better than relying on automatic prefix caching because:

- Session cache blocks will not be evicted while waiting for the next input chunk.
- Prefix caching is **block-level** (typically **16** tokens), so a small remainder would be recomputed on every new chunk.

Cost: an open session **pins** that memory. Other requests cannot use it; capacity / throughput suffer if sessions linger. **At the time of the post, vLLM would not preempt idle streaming-input sessions.** The authors said that would improve in a later update.

### Future directions

More fully streamable open weights that match this input-streaming design → a larger realtime app ecosystem. The capability was still novel in LLM serving, so they expected to adapt it: tighter audio/video encoder integration, tuning the anchor-request pattern for different latency budgets, multimodal streaming.

### Get involved

Try input streaming and the Realtime API. Feedback, issues, and patches: [vLLM GitHub](https://github.com/vllm-project/vllm).

## Acknowledgements

Streaming input and the Realtime API were a multi-team effort:

**Meta:** Joshua Deng, Jiatong Zhou, Zhuohan Li, Yu Luo, Jeremy Teboul

**Mistral AI:** Patrick von Platen, Andy Lo

**vLLM Team:** Nick Hill, Roger Wang, Cyrus Leung, Nicolò Lucchesi, Woosuk Kwon

Other streaming-input work in vLLM they name: Tao He (Alibaba Qwen), Edward Wibowo (Brown University), Deepti Raghavan (Brown University), Luis Gaspar Schroeder (UC Berkeley).
