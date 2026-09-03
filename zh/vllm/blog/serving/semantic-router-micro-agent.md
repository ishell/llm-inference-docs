---
source: https://vllm.ai/blog/2026-06-29-micro-agent-frontier-models
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Micro-agent：一个 model 名后面的有界协作

英文对照：[en/vllm/blog/serving/semantic-router-micro-agent.md](../../../../en/vllm/blog/serving/semantic-router-micro-agent.md)  
原文：https://vllm.ai/blog/2026-06-29-micro-agent-frontier-models  
分数是他们 closed/hybrid 配方的 scorecard，不是「每个请求都该上全套闭源模型」。

用户仍调 `vllm-sr/auto`。Looper 在路由里跑：Confidence（便宜先试，低分再升级）、Ratings（`max_concurrent` 封顶的并行加权）、ReMoM（广度采样 + quorum + 合成）、Fusion（面板–法官）、Workflows（有预算的角色）。任务整形：GPQA 保 `ANSWER: X`，LiveCodeBench 看隐藏测试风险，HLE 看分歧。他们表上 VSR Closed：LiveCodeBench 92.6、GPQA-Diamond 96.0、HLE 50.0——对照行在原文。协作是 serving 原语，不是应用里再搭一套 agent 图。

本地图（原文版权仍归原站；学习对照用）：

![router capability layer](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/01-router-capability-layer.png)

![looper micro agents](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/02-looper-micro-agents.png)

![confidence loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/03-confidence-loop.png)

![ratings loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/04-ratings-loop.png)

![remom loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/05-remom-loop.png)

![fusion loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/06-fusion-loop.png)

![workflows loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/07-workflows-loop.png)

![auto recipe loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/08-auto-recipe-loop.png)

![benchmark shaped recipes](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/09-benchmark-shaped-recipes.png)

![three eval scorecard](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/10-three-eval-scorecard.png)
