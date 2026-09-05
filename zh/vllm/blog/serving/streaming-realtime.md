---
source: https://vllm.ai/blog/2026-01-31-streaming-realtime
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# 输入不必一次到齐：StreamingInput 与 `/v1/realtime`

英文对照：[en/vllm/blog/serving/streaming-realtime.md](../../../../en/vllm/blog/serving/streaming-realtime.md)  
原文：https://vllm.ai/blog/2026-01-31-streaming-realtime  
2026-01-31。署名 **Meta, Mistral AI as well as the vLLM team**。流式输入：[PR #28973](https://github.com/vllm-project/vllm/pull/28973)。Realtime WebSocket：[PR #33187](https://github.com/vllm-project/vllm/pull/33187)。当时的用法入口：[streaming input 测试](https://github.com/vllm-project/vllm/tree/main/tests/v1/streaming_input)、[Realtime API 文档](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/?h=realtime+api#realtime-api)。和 [Anatomy](../architecture/anatomy.md) 的 Prefill/Decode、[V1](../architecture/v1-alpha.md) 的 scheduler 一起读：改的是 **请求怎么进引擎**，不是 attention 公式。原文封面图未收录。

适用：语音助手、同传、现场转写、机器人传感器流——输入还在来，输出已经要开始。不适合：把任意因果 LLM 塞进 `/v1/realtime` 就当 ASR；也不要把 **chunked prefill**（调度切完整 prompt）当成「token 从网线一块块到」。

传统推理很干净：用户交一整段 prompt，模型算完，再一次或 SSE 吐回答。文字聊天、批处理够用。音频、视频、机器人等不起。vLLM 在引擎里加了 **streamable inputs**，上面再叠一套 **Realtime WebSocket API**，开口是 `/v1/realtime`。

输出流早就有。输入却一直要等齐。这两件事补的是后一半。

## 为什么要 realtime

### 传统 batch：prompt 必须先齐

请求走完整的 [`ChatCompletionRequest`](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)，等模型整段处理完，再拿整段回答。输出可以一块块吐；输入侧仍是「先交齐」。

聊天、摘要、写代码，这个模型很自然。有一类应用不能等输入结束才开工。

### 流式为什么要紧

用声音开电脑、开手机：麦克风把音频流进 LLM，模型要连续处理、实时吐动作。这里的延迟更精确地说是 [TTFT](https://www.emergentmind.com/topics/time-to-first-token-ttft)。原文的尺子：打开应用、往搜索栏打字，人不想等超过一秒。要像人一样边听边说，模型必须 **同时** 处理音频流和生成动作。

能不能把非流式模型切成块来假装流式？原则上：音频切段、每段独立算、输出再拼。实际上几道墙：

- 亚秒 TTFT 需要很强的 **chunk detection**——切在哪里才不丢关键信息。
- 切坏了：TTFT 变长，或把时间上下文撕碎，模型变差。
- 每块必须算完才能答，仍是回合制，听和说叠不起来。

同一缺口在很多地方出现：

- **语音助手** — 要亚秒才像人。
- **现场转写** — 话还在说，字已经上屏。
- **机器人 / embodied AI** — 相机、麦克风、LIDAR；控制动作拖不起。

整段等齐，是这类负载付不起的税。基础设施得增量处理输入，并在输入尚未到齐时就开始生成。

即便是「必须读完全部输入才能吐第一个输出 token」的老应用，输入一到就往引擎送，对 TTFT 仍可能有好处。vLLM 默认开 [chunked prefill](https://docs.vllm.ai/en/stable/cli/serve/?h=max+num+b#-enable-chunked-prefill-no-enable-chunked-prefill)：\(N\) 个 token 要走 \(N \div M\) 次前向，\(M\) 是 [`max_num_batched_tokens`](https://docs.vllm.ai/en/stable/cli/serve/?h=max+num+b#-max-num-batched-tokens)。若 \(N \div M > 1\)，token 随到随送，**第一次** Prefill 前向可以更早排上。

这两件事不要并成一句：

| | Chunked prefill | 流式输入 |
| --- | --- | --- |
| 切开的是什么 | 调度器手里 **已经齐** 的整段 prompt | **网线上陆续到达** 的 token / 音频 |
| 谁切 | Scheduler（`max_num_batched_tokens`） | 客户端 / yield `StreamingInput` 的 `AsyncGenerator` |
| 赢在哪 | Prefill 和 Decode 交错；胖 Prefill 有上限 | 最后一块还没到就能开 Prefill；真正的双工 |

### 流式的两道门槛

不是所有模型都能真流式。两件事都要：对的 attention 形态，以及按增量处理训过。

#### Attention 形态

- **Causal attention**（单向 mask）：位置 \(t\) 只看 \(j \le t\)。未来 token 进不来，\(t\) 的输出在 token \(t\) 到达时就是终态。真流式成立；早先的输出不必因为后面的输入再改。
- **Bidirectional attention**（全 mask）：每个位置看过去也看未来。\(t\) 的输出绑在可能还没到的 token 上。整段未知，任何位置都没有稳的输出。和流式 / 在线处理不相容。

长跑、甚至无穷流，光因果还不够：每个 token 都看全部过去，算力和显存没有上界。实践里要截断过去。常见做法：**sliding-window attention**——只看固定近窗，算力和显存有界。现代流式模型常常是因果加滑窗。

#### 训练必须按流式对齐

架构能流，仍不够。模型还得为 **true streaming input** 训过。

记输入 \(X = (x_0, \ldots, x_T)\)，输出 \(Y = (y_0, \ldots, y_{T'})\)。流式里，\(y_t\) 应对着时刻 \(t\) 的 \(x_t\)，延迟尽量小——比如一帧音频 \(x_t\) 的转写。

常规 next-token 目标把下一个 token 的分布条件在 **整段** 输入上：

\[
P(y_i \mid y_{i-1}, \ldots, y_0, x_T, x_{T-1}, \ldots, x_0).
\]

实时用不上：生成 \(y_i\) 需要完整的 \(X\)。

流式模型只能用已经看到的过去，外加一点点 lookahead \(\delta\)：

\[
P(y_i \mid y_{i-1}, \ldots, y_0, x_{i+\delta}, \ldots, x_i, \ldots, x_0),
\]

\(\delta\) 越小越好。理论上可以是 0；实践里通常要留一点延迟，质量才站得住。

于是训练要两件事：

- **i)** 把输入输出对齐成 \(T' = T\)，每个 \(y_i\) 就是 \(x_i\) 该有的输出；
- **ii)** 架构能在已经处理完 \(x_i, \ldots, x_0\) 之后，再吃进新的 \(x_{i+1}\)。

[Delayed Streams Modeling](https://arxiv.org/abs/2509.08753)（Voxtral-Realtime 沿这条路；原文第二条链接写成 `TODO`）把输入 embedding（如语音）和输出 embedding（如文本）sum-pool 成一条，再预测

\[
P(y_i \mid y'_{i-1}, \ldots, y'_0), \qquad y'_k = y_k + x_{k+\delta}.
\]

部署上这一点很硬：任意因果模型丢进 realtime 口，不会自动变成流式 ASR。要对齐方式、架构约束都按上面训过。

### 架构决定 serving 能不能流

vLLM 能伺候许多模型。**真**流式要架构上就是因果的。[Voxtral](https://mistral.ai/news/voxtral) 从训练起就为流式设计，用支持增量的因果 attention。

Serving 也得接增量输入。模型能流、服务器却要等齐 prompt，延迟优势就没了。所以 vLLM 在已有的输出流之外，把输入也做成流。

### 流式架构延伸阅读

- [Transformer Transducer](https://arxiv.org/abs/2002.02562) — 可流式语音识别里很成功的一条。
- [Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling](https://arxiv.org/abs/2509.08753)（Kyutai）— 上面那套架构的展开。
- [Streaming Simultaneous Speech Translation with Augmented Memory Transformer](https://arxiv.org/abs/2011.00033) — 翻译不如语音「单调」，流式更难做好。
- [Voxtral](https://mistral.ai/news/voxtral) / Voxtral-Realtime — 大规模预训练、开源，跟多数离线语音识别可打。

## vLLM 里的流式输入

[PR #28973](https://github.com/vllm-project/vllm/pull/28973) 把流式输入接进推理：输入随时间到，输出连续生成。

### `StreamingInput`

```python
from dataclasses import dataclass
from vllm.inputs import PromptType
from vllm.sampling_params import SamplingParams

@dataclass
class StreamingInput:
    prompt: PromptType
    sampling_params: SamplingParams | None = None
```

不要把定死的 prompt 交给 `AsyncLLM.generate()`。传一个随时间 yield `StreamingInput` 的 `AsyncGenerator`。每一块接到累积 prompt 后面。

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

可以等上一块对应的输出结束再送下一块，但不是必须——块会在内部排队。输入结束：从 async generator 退出，或 `aclose`。**输出** generator 要等到收到的输入都处理完、**并且** 输入 generator 也结束，才完。

例子里 `max_tokens=1`：每次 resumable 请求主要给 `prompt_token_ids` 算 KV，那一个 `output_token_ids` 交给应用。生产采样跟模型走。空 list 在示例循环里表示输入结束。

### 内部怎么接

每一块被当成一次独立请求，prompt 却是累积的。新块进来，引擎会：

1. 用 `max_tokens - 1` 个已生成的 `output_tokens`，加上新到的 `prompt_token_ids`，伸长 `prompt_token_ids`。
2. 复用全部已缓存的 KV。
3. 按当前累积 prompt 和这次的 `max_tokens` 再生成。
4. 新输入到了，可以选择丢掉尚未落稳的输出。

块与块之间吐出的 token，后面上下文变了可能要改。最终输出对着完整输入。

实现是 **sticky session**。第一块造一只活过整段会话的 **anchor request**。之后同一内部 request ID 的块排队、按序处理。

### Anchor Request

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

**下一轮 `prompt_token_ids` 为什么丢掉最后一个 token（Y）？**

resumable 请求真正要算的，是全部 `prompt_token_ids` 的 KV，外加 `max_tokens - 1` 个已生成 token 的 KV。用户用 `max_tokens` 声明：前一次请求里那 `max_tokens - 1` 个生成 token **已经是终态**，可以复用。`output_token_ids` 的最后一个，只是最近一次前向的采样结果，**还没有对应的 KV**。应用仍可以看它；更新后的 anchor 不会把它写进 `prompt_token_ids`。丢掉几乎免费：没有缓存状态被作废，留下来也得重算。

[Realtime API](https://github.com/vllm-project/vllm/blob/a2443de5fa4a0605607f6c3d9219022c7f6ac480/vllm/entrypoints/openai/realtime/connection.py#L209) 和多数应用把 `max_tokens` 设成 `1`：每次 resumable 请求只给 `prompt_token_ids` 填 KV，那一个生成 token 交给用户。[Voxtral Realtime](https://mistral.ai/news/voxtral) 会把这一个 token 和下一块音频拼成下一次 resumable 请求。

**原文的 caveat：** 有的模型会吐特殊 stop token，后续生成还依赖它。调度就要多留 **+1 token**，先把 stop token 重算出来，再处理新输入块。

### 一段例子

不同块可以带不同的 `max_tokens`。原文的示意：

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

## Realtime API（WebSocket）

流式输入是引擎能力。生产还要一口方便的双向通道。[PR #33187](https://github.com/vllm-project/vllm/pull/33187) 做了 WebSocket Realtime API，灵感来自 [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)。开口：`/v1/realtime`。

### 架构

客户端和 vLLM 服务器双向流。客户端送音频，服务器回话写文本和模型输出。四块：

1. **WebSocket Client** — 从麦克风（或文件）抓音频，分块送出。
2. **Realtime Handler** — 收 WebSocket 消息，转成 `StreamingInput`。
3. **AsyncLLM** — 处理流式输入，生成输出。
4. **Response Stream** — 生成的 token 从同一条 WebSocket 送回。

### 起服务

```bash
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --enforce-eager
```

WebSocket：`ws://localhost:8000/v1/realtime`。

### 客户端例子

把音频文件流过去，打转写。期望格式：**PCM16 @ 16 kHz**。块大小：**4 KB**。原文片段里 `session.update` 用了未定义的 `model`——按发表稿照抄。

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

页上的流程：

- 音频转成 PCM16 @ 16 kHz（接口期望的输入）。
- 连 `/v1/realtime`，发 `session.update` 校验模型。
- 按 4 KB 切，用 `input_audio_buffer.append` 送；`input_audio_buffer.commit` 标开始和结束（收尾那次带 `"final": True`）。
- 打 `transcription.delta`，直到 `transcription.done`。
- **这个示例的 realtime 保留：** 它先把音频全部送完再听转写。协议本身完全异步——可以边送块边收转写。生产服务应在第一块到达时就开始转写，发送和接收叠在一起。

### 消息类型

原文点名的类型都列在这里（示例和后文目录不完全重合，两份都收）。

**客户端 → 服务器**（目录）：

- `session.create` — 开新 session
- `input_audio_buffer.append` — 送音频
- `input_audio_buffer.commit` — 标音频输入结束（示例里也用来标流的 **开始**，结束时再带 `"final": True`）
- `response.create` — 请求模型回复

**服务器 → 客户端**（目录）：

- `session.created` — session 建好
- `response.text.delta` — 增量文本（生成 / TTS 路径）
- `response.audio.delta` — 增量音频（TTS 模型）
- `response.done` — 这一轮完
- `error` — 出错

**示例里还用到、目录没重复写的：**

- `session.update` — 连上之后校验 / 设置模型
- `transcription.delta` — 部分转写
- `transcription.done` — 转写结束

发文时这层表面还在长；和 OpenAI Realtime 的事件名可能有差——以当时文档为准。

### 示例脚本

仓库里现成的客户端：

- [`examples/online_serving/openai_realtime_client.py`](https://docs.vllm.ai/en/latest/examples/online_serving/openai_realtime_client/?h=realtime#openai-realtime-client) — 基本 WebSocket 客户端
- [`examples/online_serving/openai_realtime_microphone_client.py`](https://docs.vllm.ai/en/latest/examples/online_serving/openai_realtime_microphone_client/#openai-realtime-microphone-client) — 接麦克风

演示从系统麦克风抓音频，实时流进 vLLM。

### 性能上要小心的

专用的 `AsyncGenerator` session 把这路会话的 KV **原样留住**。这比指望自动 prefix caching 更稳，因为：

- 等下一块输入时，对应的 cache block **不会被赶走**。
- Prefix caching 是 **block 级**（通常 **16** token），余数每来一块都可能重算一小截。

代价：开着的 session **占住** 那份内存。别的请求用不上，会话挂太久会伤容量 / 吞吐。**发文时 vLLM 还不会抢占闲着的流式输入 session。** 作者说后续会改。

### 以后想做的

更多和这套输入流设计对得上的、开源的全流式权重出来，realtime 应用生态才会明显变大。LLM serving 里这还是新能力，他们准备接着改：音频 / 视频 encoder 接得更紧、按不同延迟预算调 anchor request、把多模态流式铺开。

### 一起做

去试输入流和 Realtime API。反馈、issue、补丁：[vLLM GitHub](https://github.com/vllm-project/vllm)。

## 致谢

流式输入和 Realtime API 是多队合作：

**Meta：** Joshua Deng、Jiatong Zhou、Zhuohan Li、Yu Luo、Jeremy Teboul

**Mistral AI：** Patrick von Platen、Andy Lo

**vLLM Team：** Nick Hill、Roger Wang、Cyrus Leung、Nicolò Lucchesi、Woosuk Kwon

文中还点名了其他在 vLLM 里做过流式输入的人：Tao He（Alibaba Qwen）、Edward Wibowo（Brown University）、Deepti Raghavan（Brown University）、Luis Gaspar Schroeder（UC Berkeley）。
