---
source: https://vllm.ai/blog/2026-06-05-v0.3-vllm-sr-themis-release
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Themis v0.3：能问「刚才为什么走这条」

英文对照：`en/vllm/blog/serving/semantic-router-themis.md`  
原文：https://vllm.ai/blog/2026-06-05-v0.3-vllm-sr-themis-release  
相对 v0.2 约 350+ commit。

合同收成一条：信号 → **projection** → 决策 → 算法 → 模型。CLI、dashboard、DSL、Helm 要说同一种话。运营能回答：哪些信号响了、哪条决策、哪个选模算法、安全/replay 插件有没有改道、哪版 config。有状态、可 replay、协议对齐、会话连续。Athena 的野心还在，边界更硬。和引擎 [Router](router.md) 仍不是同一个词。

本地图（原文版权仍归原站；学习对照用）：

![hero v2](../../../../assets/vllm/blog/serving/semantic-router-themis/01-hero-v2.png)

![release value map](../../../../assets/vllm/blog/serving/semantic-router-themis/02-release-value-map.png)

![config contract](../../../../assets/vllm/blog/serving/semantic-router-themis/03-config-contract.png)

![routing contract](../../../../assets/vllm/blog/serving/semantic-router-themis/04-routing-contract.png)

![session aware routing](../../../../assets/vllm/blog/serving/semantic-router-themis/05-session-aware-routing.png)

![projection layer](../../../../assets/vllm/blog/serving/semantic-router-themis/06-projection-layer.png)

![operator console](../../../../assets/vllm/blog/serving/semantic-router-themis/07-operator-console.png)

![long context binding](../../../../assets/vllm/blog/serving/semantic-router-themis/08-long-context-binding.png)

![hardware backend paths](../../../../assets/vllm/blog/serving/semantic-router-themis/09-hardware-backend-paths.png)

![amd validation path](../../../../assets/vllm/blog/serving/semantic-router-themis/10-amd-validation-path.png)

![routerarena leaderboard vllm sr](../../../../assets/vllm/blog/serving/semantic-router-themis/11-routerarena-leaderboard-vllm-sr.png)

![hermes roadmap](../../../../assets/vllm/blog/serving/semantic-router-themis/12-hermes-roadmap.png)
