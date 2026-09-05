---
source: https://vllm.ai/blog/2025-01-21-stack-release
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# vLLM production-stack：从单机引擎到 K8s 上的一叠盘子

英文对照：[en/vllm/blog/serving/production-stack.md](../../../../en/vllm/blog/serving/production-stack.md)  
原文：https://vllm.ai/blog/2025-01-21-stack-release  
2025-01-21。仓库：https://github.com/vllm-project/production-stack。LMCache 团队 + vLLM。数字是发布时的演示，不是你集群上的 SLA。原文有架构 / TTFT / ITL / Grafana 图，本仓库没有那些原图的副本。

## TL;DR

vLLM 已经是开源社区里最热闹的**单机**推理引擎。这篇问的是：怎样把它变成一套**集群 serving 系统**。

他们交出 **vLLM production-stack**，自称两件好处：

- **大约 10× 更好的性能**（正文拆开写：**延迟 3–10×** 更低、**吞吐 2–5×**），靠 prefix-aware routing 和 KV 共享。标题里的 10× 以原文图为准。
- **集群好部署**：容错、autoscaling、可观测，走 Helm。

从第一天就是开源。

## 背景

他们引用的那句：AI 军备竞赛已经不只是谁的模型最好，而是谁的 **LLM serving 系统**最好。硬件和模型覆盖、贡献者密度都已经有了；部署却仍多半是**单机**。LMCache 和 vLLM 一起搭这块参考实现，好让一家组织能在 GPU 集群上指着看板说话：可靠、吞吐、延迟。

它不替换 PagedAttention。它坐在单机引擎**上面**。

## 四块补丁

对着**一簇 GPU 节点**的参考实现：

1. **KV 共享与外置存储**（[LMCache](https://github.com/LMCache/LMCache)）：同一段 context 被复用时，少算一遍。长上下文、Prefill 重的负载，是他们给自己标的长处。
2. **Prefix-aware routing**：把请求送到**已经握着这段 KV** 的那台实例。无状态 round-robin 会把前缀命中率踩碎。
3. **可观测性**：引擎状态 + 查询级 **TTFT** / **TBT** / 吞吐。没有这三项，autoscaling 只是在猜。
4. **Autoscaling**：负载涨了起新节点。

原文有一张对照表，把 production-stack 和当时能找到的邻近方案（含 KServe）并排放——表在网页里，本地不搬。

流量的故事很短：应用把请求送来 → 路由看池子里谁已经算过这段 context → 转发；集群管理看整体负载起新的 vLLM 节点；看板盯 TTFT / TBT / 吞吐和 KV hit rate。后来 [Router](router.md) 把「认得 KV」收成 Rust 网关；[Mooncake](mooncake.md) 再把「没握着也能从池子里捞」补上。

## 部署

Helm 一条（原文带 `sudo`，按你集群的习惯取舍）：

```bash
sudo helm repo add llmstack-repo https://lmcache.github.io/helm/ &&\
  sudo helm install llmstack llmstack-repo/vllm-stack
```

仓库 README 和 [tutorials](https://github.com/vllm-project/production-stack/tree/main/tutorials) 写搭 K8s、改 Helm values。这是参考实现，不是「装完就等于生产」。

## 成绩与看板（演示）

基准是**多轮问答**——正好是 prefix cache 该发光的形状。对手包括 **vLLM + KServe** 和某个商业 endpoint。他们报 TTFT 和 ITL（字与字之间）。多轮问答上好看，不代表短问短答、或完全没有共享前缀的负载也会好看。旋钮的意义随负载变，NVIDIA 那组压测文说的是同一句。

监控他们点名的量：延迟分布、请求随时间、**KV cache hit rate**。

## 收场

他们把这一天写成：从最好的单机引擎，走向能铺开的 serving 系统。页面上的行动：clone、试用、[兴趣表](https://forms.gle/mQfQDUXbKfp2St1z7)。联系：[vLLM Slack](https://slack.vllm.ai/)、[LMCache Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ)。

一个月后 ByteDance 交出 [AIBrix](aibrix.md)。FAQ 写得很干脆：一个从生产里长出来，一个从社区积木长出来；production-stack 的长处是 KV 向的传输、blending、路由；短期内还打算借用 AIBrix 的零件。读这两篇不是为了选边，是为了看清 2025 年初社区在补的那一层——引擎已经很快，缺的是**认得出记忆的控制面**。
