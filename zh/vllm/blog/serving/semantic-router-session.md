---
source: https://vllm.ai/blog/2026-06-02-session-aware-agentic-routing
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# SAAR：长程 agent 问的是「现在能不能换模」

英文对照：`en/vllm/blog/serving/semantic-router-session.md`  
原文：https://vllm.ai/blog/2026-06-02-session-aware-agentic-routing  
2026-06-02。图在原网页。数字是确定性矩阵 + AMD ROCm 现场跑，当演示。产品化写进 [Themis](semantic-router-themis.md)。

单轮路由看见 tool 结果像便宜短句，可能甩给小模型——工具环就断了。SAAR 仍走信号→决策→选模，外面加一层会话控制。客户端带稳定 `x-session-id`。

五件：

- **Router memory：** 上次物理模型、决策、phase、切换次数、idle、cache 证据、replay id。不是对话记忆。
- **Hard lock：** 工具环、不可移植的 provider continuation。不安全就不许买便宜。
- **Reset：** idle timeout、决策漂移才重选。
- **Switch economics：** prefix-cache checkout 差价；暖的前沿会话更难丢掉。
- **Replay：** 留下为何 stay / switch / locked stay。

`algorithm.type: session_aware`，`base_method: hybrid` 一类。这是选模策略，不是 endpoint 负载均衡——后者仍是 [Router](router.md) / Envoy。

## 数字（演示）

21,600 确定性 turn：相对单轮，切换 **−79.29%**，不安全切换 3,836→0，估算物理模型成本 **−78.71%**。粘会话几乎不切（340 次）但质量 δ **−0.1433**；Full SAAR 质量 δ **−0.0453**。消融：去掉 tool lock 不安全切换回到 760。

现场 AMD ROCm 2,896 请求、0 次连续性违规。balanced-32×64：2,048 req、p95 overhead **6.181 ms**；stateful-16×48：**26.805 ms**；idle 带真实 sleep 的 p95 **283.463 ms** 要分开读。注入 503 后 32/32、24/24 session 恢复。任务迹 18/18，96/96 routed turn 有 replay header。
