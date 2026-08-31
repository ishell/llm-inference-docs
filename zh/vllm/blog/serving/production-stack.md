---
source: https://vllm.ai/blog/2025-01-21-stack-release
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# vLLM production-stack：从单机引擎到 K8s 上的一叠盘子

英文对照：`en/vllm/blog/serving/production-stack.md`  
原文：https://vllm.ai/blog/2025-01-21-stack-release  
2025-01-21。仓库：https://github.com/vllm-project/production-stack。LMCache 团队 + vLLM。图在原网页。数字是发布时的演示：相对「裸 vLLM + 别的架子」，他们自称延迟 **3–10×** 更低、吞吐 **2–5×**（文里又写 10×——以原文图为准）。这是 **单机引擎上面的集群参考实现**，不是替换 PagedAttention。

四块补丁：

- **KV 共享与外置存储**（LMCache）：上下文被复用时少算。
- **Prefix-aware routing**：把请求送到**已经握着这段 KV** 的那台实例。
- **可观测性**：引擎状态 + 查询级 TTFT / TBT / 吞吐。
- **Autoscaling** 与故障。

流量：应用 → 路由看哪台池子里已有这段 context → 转发；集群管理看负载起新节点；看板盯 TTFT/TBT/吞吐。Helm 一条：

```bash
sudo helm repo add llmstack-repo https://lmcache.github.io/helm/ && \
  sudo helm install llmstack llmstack-repo/vllm-stack
```

基准是多轮问答，对手包括 vLLM+KServe 和某个商业 endpoint。TTFT / ITL 图在原文。监控面板：延迟分布、请求随时间、KV hit rate。

下一篇 AIBrix 会直接问：这和 ByteDance 那套控制面有何不同。答案写在那篇 FAQ 里——一个从生产里长出来，一个从社区积木长出来；短期内 production-stack 还打算借用 AIBrix 的零件。
