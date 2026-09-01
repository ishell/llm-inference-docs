---
source: https://vllm.ai/blog/2026-06-05-v0.3-vllm-sr-themis-release
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Themis v0.3：能问「刚才为什么走这条」

英文对照：`en/vllm/blog/serving/semantic-router-themis.md`  
原文：https://vllm.ai/blog/2026-06-05-v0.3-vllm-sr-themis-release  
相对 v0.2 约 350+ commit。图在原网页。

合同收成一条：信号 → **projection** → 决策 → 算法 → 模型。CLI、dashboard、DSL、Helm 要说同一种话。运营能回答：哪些信号响了、哪条决策、哪个选模算法、安全/replay 插件有没有改道、哪版 config。有状态、可 replay、协议对齐、会话连续。Athena 的野心还在，边界更硬。和引擎 [Router](router.md) 仍不是同一个词。
