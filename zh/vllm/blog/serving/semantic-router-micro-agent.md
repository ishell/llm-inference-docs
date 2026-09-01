---
source: https://vllm.ai/blog/2026-06-29-micro-agent-frontier-models
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Micro-agent：一个 model 名后面的有界协作

英文对照：`en/vllm/blog/serving/semantic-router-micro-agent.md`  
原文：https://vllm.ai/blog/2026-06-29-micro-agent-frontier-models  
图在原网页。分数是他们 closed/hybrid 配方的 scorecard，不是「每个请求都该上全套闭源模型」。

用户仍调 `vllm-sr/auto`。Looper 在路由里跑：Confidence（便宜先试，低分再升级）、Ratings（`max_concurrent` 封顶的并行加权）、ReMoM（广度采样 + quorum + 合成）、Fusion（面板–法官）、Workflows（有预算的角色）。任务整形：GPQA 保 `ANSWER: X`，LiveCodeBench 看隐藏测试风险，HLE 看分歧。他们表上 VSR Closed：LiveCodeBench 92.6、GPQA-Diamond 96.0、HLE 50.0——对照行在原文。协作是 serving 原语，不是应用里再搭一套 agent 图。
