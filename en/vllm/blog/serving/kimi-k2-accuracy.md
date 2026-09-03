---
source: https://vllm.ai/blog/2025-10-28-kimi-k2-accuracy
lang: en
fetched: 2026-09-01
---

# Kimi K2 tool-calling: the handshake is the chat template, not the MoE kernel

Chinese: [zh/vllm/blog/serving/kimi-k2-accuracy.md](../../../../zh/vllm/blog/serving/kimi-k2-accuracy.md)  
v0.11.0, K2-Vendor-Verifier. Templates after Kimi-K2-0905 `94a4053` / Kimi-K2 `0102674`.

Official API schema errors **0**; vLLM initially 218 successful tool calls / 1200+ (<20%). Three cuts: ① `add_generation_prompt` hid in tokenizer `**kwargs`; vLLM dropped it for security (PR #25794); prompt missed `<|im_assistant|>…<|im_middle|>`, model continued as prose. ② Empty `content: ''` promoted to list-of-dicts; Jinja dumped the literal into the prompt. ③ IDs must be `functions.func_name:idx`; history `search:2` blew `split('.')[1]`. After Hub templates: 218→~1000 parsed, F1 **83.57%**, schema still **76%** — model hallucinates tools from history not in this turn. Moonshot has an Enforcer (constrained decoding); vLLM did not. Debug via `/completions` and token ids, not only `/chat/completions`.

Local figures (copyright remains with the original site; study copies):

![k2 vendor verifier](../../../../assets/vllm/blog/serving/kimi-k2-accuracy/01-k2-vendor-verifier.jpeg)
