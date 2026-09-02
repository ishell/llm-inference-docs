---
source: https://vllm.ai/blog/2025-10-22-agent-lightning
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# `return_token_ids`：Agent RL 别再二次分词

英文对照：`en/vllm/blog/serving/agent-lightning.md`  
原文：https://vllm.ai/blog/2025-10-22-agent-lightning  
vLLM ≥0.10.2。

单轮 RL 可以直接拿 `generate` 的 token。Agent 框架走 OpenAI `chat.completions`，以前只回字符串。训练时再 tokenize：**Retokenization Drift**。原文曲线：存文本再分词的两条 run 不稳，直接用推理 token 的那条稳。

三处常见分叉：

1. **HAVING**：生成时 `H`+`AVING`，训练时 `HAV`+`ING`，字面一样 ID 不同。
2. **Tool-call**：parser 把 `<tool_call>{...}</tool_call>` 收成对象再渲染，空白/JSON 修正会掩盖模型真错误。
3. **Chat template**：vLLM 与 HuggingFace 模板差一个空格，整段 ID 就漂。

这种 off-policy 不在 token 级 IS 能修的尺度上。

请求里 `"return_token_ids": true`（`/v1/chat/completions` 或 `/v1/completions`），响应带 `prompt_token_ids` 和 `token_ids`。Agent Lightning 把每次模型调用当独立 sample，不再把轨迹缝成一条。v0.1 还 monkey-patch 过 OpenAI server；现在自动加这个字段。和 [Native RL](native-rl.md)、[bitwise RL](bitwise-rl.md) 一起读：token ID 对齐是政策，kernel 对齐是数值。

本地图（原文版权仍归原站；学习对照用）：

![1 rewards](../../../../assets/vllm/blog/serving/agent-lightning/01-1_rewards.png)

![2 having](../../../../assets/vllm/blog/serving/agent-lightning/02-2_having.png)

![3 agl](../../../../assets/vllm/blog/serving/agent-lightning/03-3_agl.png)

![4 tasks spans loop](../../../../assets/vllm/blog/serving/agent-lightning/04-4_tasks-spans-loop.svg)
