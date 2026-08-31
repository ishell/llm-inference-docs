---
source: https://vllm.ai/blog/2025-12-13-vllm-router-release
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# vLLM Router：认得 KV、认得 Prefill/Decode 的负载均衡

英文对照：`en/vllm/blog/serving/router.md`  
原文：https://vllm.ai/blog/2025-12-13-vllm-router-release  
2025-12-13。仓库：Github 上的 vllm-router。Rust，尽量轻。坐在客户端和一队 vLLM worker 之间，K8s 或裸金属都行。从 **SGLang model gateway** fork 再简化；他们说可能并进 vLLM 主仓，大规模 gateway 功能也可能再和 SGLang 对齐。图在原网页。

普通负载均衡把 LLM 当无状态。KV 是有状态的；P/D 分离更不是「所有 pod 长得一样」。Router 要办的就是这两件事。

## 负载策略

对话要把同一用户后续请求粘到**还握着他 KV** 的那台 worker。

- **Consistent hashing**：同一 routing key（session / user）粘住，prefix cache 才有意义。性能主策略。
- **Power of Two**：低开销随机，分布仍好。
- Round robin / random：无状态兜底。

## Prefill/Decode 分离

compute-bound 的 prefill 和 memory-bound 的 decode 分成两组 worker。Router：新请求进 prefill 组 → 完成后把状态交给 decode 组。发现与路由支持 **NIXL** 以及 **NCCL + ZMQ discovery**。

## 生产

K8s label selector 自动发现 pod。重试（指数退避 + jitter）+ 熔断；健康检查失败立刻踢出池子。`/metrics` Prometheus：量、延迟、错误、每台 worker 健康。

## 当时的基准（演示）

排除了 vLLM 自带 DP/EP coordinator（当时吞吐只有别人的 1/8，已知问题）。对手：llm-d（默认 queue-aware）、K8s 原生 round-robin（**不认 P/D**）。

Llama 3.1 8B，8 prefill + 8 decode pod：Router 的 req/s 比 llm-d 高约 **25%**，比 K8s 原生高约 **100%**；TTFT 接近原生，比 llm-d 快约 **1200 ms**。

DeepSeek V3，1 prefill TP8 + 1 decode TP8：req/s 接近 llm-d，比原生高约 **100%**；TTFT 比两者快约 **2000 ms**。

读完 production-stack / AIBrix 再读这一篇：集群里真正决定「下一句话去哪张卡」的，是这只认得记忆的路由器。P/D 的编排从这里开始；下一篇 EPD 把**视觉编码器**也拆出去。
