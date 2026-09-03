---
source: https://vllm.ai/blog/2025-01-21-stack-release
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# vLLM production-stack：从单机引擎到 K8s 上的一叠盘子

英文对照：[en/vllm/blog/serving/production-stack.md](../../../../en/vllm/blog/serving/production-stack.md)  
原文：https://vllm.ai/blog/2025-01-21-stack-release  
2025-01-21。仓库：https://github.com/vllm-project/production-stack。LMCache 团队 + vLLM。数字是发布时的演示。

到这一天，vLLM 已经是开源社区里最热闹的单机推理引擎。模型多、硬件多、人多。可「最好的单机引擎」和「一家公司能放心铺开的 serving 系统」之间，还差一叠盘子：请求去哪台、KV 能不能共用、挂了谁来补、看板看什么。LMCache 团队和 vLLM 一起交出的参考实现，名字就叫 production-stack。它不替换 PagedAttention，它坐在单机引擎**上面**。

他们自称相对「裸 vLLM + 别的架子」，延迟 **3–10×** 更低、吞吐 **2–5×**（标题里又写 10×——以原文图为准）。这是发布日的演示，不是你集群上的 SLA。

## 四块补丁

- **KV 共享与外置存储**（LMCache）：同一段 context 被复用时，少算一遍。长上下文、prefill 重的负载，这是他们给自己标的长处。
- **Prefix-aware routing**：把请求送到**已经握着这段 KV** 的那台实例。无状态 round-robin 会把前缀命中率踩碎；后来 [Router](router.md) 把这件事做成 Rust 网关，[Mooncake](mooncake.md) 再把「没握着也能从池子里捞」补上。
- **可观测性**：引擎状态 + 查询级 TTFT / TBT / 吞吐。没有这三项，autoscaling 只是在猜。
- **Autoscaling** 与故障：负载涨了起新节点，人走了把请求从死人身上揭下来。

流量的故事很短：应用把请求送来 → 路由看池子里谁已经算过这段 context → 转发；集群管理看整体负载起新的 vLLM 节点；看板盯 TTFT / TBT / 吞吐和 KV hit rate。原文有一张对照表，把 production-stack 和当时能找到的邻近方案并排放——表在网页里，本地不搬。

## 部署

Helm 一条（原文带 `sudo`，按你集群的习惯取舍）：

```bash
helm repo add llmstack-repo https://lmcache.github.io/helm/
helm install llmstack llmstack-repo/vllm-stack
```

仓库 README 里有搭 K8s 和改 Helm values 的教程。这是参考实现，不是「装完就等于生产」。

## 成绩与看板（演示）

基准是多轮问答——正好是 prefix cache 该发光的形状。对手包括 vLLM+KServe 和某个商业 endpoint。TTFT / ITL 图在原文。监控面板：延迟分布、请求随时间、KV hit rate。多轮问答上好看，不代表短问短答、或完全没有共享前缀的负载也会好看。旋钮的意义随负载变，NVIDIA 那组压测文说的是同一句。

## 和下一篇的关系

一个月后 ByteDance 交出 [AIBrix](aibrix.md)。FAQ 写得很干脆：一个从生产里长出来，一个从社区积木长出来；production-stack 的长处是 KV 向的传输、blending、路由；短期内还打算借用 AIBrix 的零件。读这两篇不是为了选边，是为了看清 2025 年初社区在补的那一层——引擎已经很快，缺的是**认得出记忆的控制面**。
