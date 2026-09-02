---
source: https://vllm.ai/blog/2025-10-22-agent-lightning
lang: en
fetched: 2026-09-01
---

# `return_token_ids`: stop retokenizing agent RL

Chinese: `../../zh/vllm/blog/serving/agent-lightning.md`  
vLLM ≥0.10.2.

Single-turn RL can take tokens from `generate`. Agent stacks call OpenAI `chat.completions`, which used to return strings only. Trainers retokenize: **Retokenization Drift**. Their plot: two text-and-retokenize runs wobble; the run that keeps engine tokens does not.

Three usual forks:

1. **HAVING**: sampled as `H`+`AVING`, retokenized as `HAV`+`ING` — same text, different IDs.
2. **Tool-call**: parser objectifies `<tool_call>{...}</tool_call>` and re-renders; whitespace/JSON fixes hide real model errors.
3. **Chat template**: one space between vLLM and HuggingFace templates drifts the whole ID sequence.

That off-policy gap is not token-level IS.

`"return_token_ids": true` on `/v1/chat/completions` or `/v1/completions` adds `prompt_token_ids` and `token_ids`. Agent Lightning treats each model call as its own sample instead of stitching a trajectory. v0.1 monkey-patched the OpenAI server; now the field is first-class. Read with [Native RL](native-rl.md) and [bitwise RL](bitwise-rl.md): token IDs are the policy; kernels are the numerics.

Local figures (copyright remains with the original site; study copies):

![1 rewards](../../../../assets/vllm/blog/serving/agent-lightning/01-1_rewards.png)

![2 having](../../../../assets/vllm/blog/serving/agent-lightning/02-2_having.png)

![3 agl](../../../../assets/vllm/blog/serving/agent-lightning/03-3_agl.png)

![4 tasks spans loop](../../../../assets/vllm/blog/serving/agent-lightning/04-4_tasks-spans-loop.svg)
