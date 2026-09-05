---
source: https://vllm.ai/blog/2026-04-14-vllm-korea-meetup-2026
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 韩国 Meetup 2026：V1、playground、NPU 插件、Omni 拆管线

英文对照：[en/vllm/blog/serving/korea-meetup-2026.md](../../../../en/vllm/blog/serving/korea-meetup-2026.md)  
原文：https://vllm.ai/blog/2026-04-14-vllm-korea-meetup-2026  
2026-04-14 写的是 **2026-04-02** 首尔那场。署名 **vLLM Team**。主办 vLLM KR Community；支持：Rebellions、SqueezeBits、Red Hat APAC、PyTorch Korea。社区纪要，不是 kernel 论文。2025 首场：[korea-meetup-2025.md](korea-meetup-2025.md)。点到的亲戚：[v1-alpha.md](../architecture/v1-alpha.md)、[playground.md](playground.md)、[vllm-omni.md](vllm-omni.md)、[semantic-router.md](semantic-router.md)、[production-stack.md](production-stack.md)、[hardware-plugin.md](../architecture/hardware-plugin.md)。会后问卷约 **75%** 回收，满意度写得很高。除了 NAVER Omni 解码路径约 **3×**，没有 SLA 数字。

![banner](../../../../assets/vllm/blog/serving/korea-meetup-2026/01-banner.jpg)

现场工程师讲生产 LLM serving。页上的框：推理已经是基础设施，从云到企业；vLLM 当公共 serving 层。

## Intro：生态往外扩

![networking](../../../../assets/vllm/blog/serving/korea-meetup-2026/02-networking.jpg)

Rebellions 的 Dr. Hongseok Kim，Red Hat APAC 的 Li Ming。

Kim：距首场六个月——Steering Group 治理，定期 meetup 和 workshop。技术：**v0 → v1** 整迁（更简单、更模块）。内部：async scheduling、Model Runner。点名的能力：streaming API、semantic router、vLLM-Omni。

![Li Ming](../../../../assets/vllm/blog/serving/korea-meetup-2026/03-intro_liming.jpg)

Li Ming：[vllm-playground](playground.md) 把 **140+** 旋钮收成 GUI——缩短 time-to-first-run，CPU 和 macOS，带性能可视化。

他们记下的一句话：serving 不再是「选哪个框架」，是在不像的环境里都跑得省。

## 加速器接 vLLM

Rebellions 的 `vllm-rbln` 插件接自家 NPU。已经有：paged attention、continuous batching。还在做：投机解码、分布式 KV、Prefill/Decode 拆分。下一代 NPU **Rebel100™** 点名给大规模推理集群。更大的转向：别再按芯片各做一套——vLLM 当公共层。插件门：[hardware-plugin.md](../architecture/hardware-plugin.md)。

## Production stack

![Hongseok](../../../../assets/vllm/blog/serving/korea-meetup-2026/04-intro_hongseok.jpg)

SqueezeBits CTO Taesoo Kim：讲 [production stack](production-stack.md) 在真实运维里有什么、怎么长起来的、往哪走。主题：过了「只 serve 一个模型」；朝生产真要的运维能力走。

## 两条 track

中段拆开：Track 1 Open Source，Track 2 Business。各两场。

![production stack](../../../../assets/vllm/blog/serving/korea-meetup-2026/05-production_stack.jpg)

### Track 1-1 — XCENA：内存和 KV 才是 serving 问题

Juho Lee（XCENA，CXL 3.0 智能内存）。LLM serving 首先是 **集群效率**：KV 怎么存、怎么复用，同时决定性能和成本。LMCache 分层 + 路由，少吃加速器 HBM；CXL 当大容量 cache 扩展层。含义：过了算力，走进数据搬运和内存层次。

### Track 1-2 — Upstage：开源权重 → 生产服务

Inseo Song（Upstage / Solar）。训完才是难的。Chat template 要同时伺候 OpenAI 兼容 API、多轮、reasoning、function calling、结构化输出；token 级状态解析。vLLM 里用 parser 和 logits processor 细抠生成。带走的一句：「稳地 serve」比「模型好」难。

### Track 2-1 — 三星：air-gap 企业

Sungsu Kim（Samsung Electronics），题目 Protecting Sensitive Data with vLLM。安全第一：外部 SaaS 免谈。内网 GPU 上的私有 LLM API，air-gap。**4000+** 员工走 OpenWebUI、OpenAI 兼容 API、Dify、Claude Code。按任务拆开的 RAG agent 加访问控制；少写自研，多靠开源。性能只是设计的一截。

### Track 2-2 — NAVER：HyperCLOVA Omni 管线

Jaeeun Gil（NAVER Cloud）。Omni 模态（文本 / 图 / 音频）：自回归栈 + diffusion decoder——异构，常规一体 serve 不合身。拆开：encoder / LLM / decoder 各成一段。**Vision decoder** 占满 e2e 延迟。Sequence parallelism + kernel：**>3×**。Serving 变成多组件管线优化，不是跑一只模型。（这是 NAVER 的 Omni 模型，不是 vLLM-Omni 那个项目。）

## 收束

![closing](../../../../assets/vllm/blog/serving/korea-meetup-2026/06-closing.jpg)

贯穿各场：多样模型、异构硬件、复杂管线，还要规模。硬件厂、云、AI 服务公司、最终用户都围着 vLLM 建策略。
