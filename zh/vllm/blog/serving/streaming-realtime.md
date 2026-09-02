---
source: https://vllm.ai/blog/2026-01-31-streaming-realtime
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# 流式输入与 `/v1/realtime`

英文对照：`en/vllm/blog/serving/streaming-realtime.md`  
原文：https://vllm.ai/blog/2026-01-31-streaming-realtime  
PR #28973。

输出可以 SSE 一块块吐，输入却常常要等整段 prompt 到齐——语音、同传、实时 agent 等不起。这篇把 **streamable inputs** 和 OpenAI 风格的 **`/v1/realtime` WebSocket** 接到 vLLM。

## 模型先得因果

双向 attention 不能流：未来帧还没到，位置已经互相看见。因果 + 滑窗可以增量；模型还得按流式对齐训过。Voxtral 一类从训练起就是这条路。任意因果 LLM 丢进 realtime 口，不一定好听。

整段 prompt 仍可 **chunked prefill**：切块进引擎，TTFT 往往比「等齐再一次 prefill」短。这和「token 一个个从网线来」是两件事——前者是调度切块，后者是输入接口。

## API

`StreamingInput`：`prompt` + 可选 `sampling_params`。不要把整段 str 交给 `AsyncLLM.generate()`，传一个 yield `StreamingInput` 的 `AsyncGenerator`。空 list 表示输入结束。示例里 `max_tokens=1` 是为了边收边吐；生产参数跟模型走。

HTTP 侧：`/v1/realtime` WebSocket。细节、事件名、与 OpenAI Realtime 的差异以原文和当时文档为准——接口还在长。

和 [Anatomy](../architecture/anatomy.md) 的 prefill/decode、[V1](../architecture/v1-alpha.md) 的 scheduler 一起读：流式输入改的是 **请求怎么进引擎**，不是 attention 公式。
