---
source: https://docs.nvidia.com/nim/large-language-models/latest/reference/benchmarking.html
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Benchmarking — NVIDIA NIM 产品页

NIM for LLMs 产品文档里的短入口（抓取时产品版本标 2.0.11）。它不教你怎么打命令。完整仪式在同目录 `nim-01`～`nim-05`；尺子本身在 `../tools/aiperf.md`。

生成式应用铺开以后，钱花在「用户还愿意等、还愿意看」的前提下，每秒能打发走多少请求。**精度达标之前不要谈成本。** 本页不覆盖 accuracy。

## 两把尺子，不要混用

市面上能打 LLM 的客户端很多：老牌 Locust / K6，以及专门认 token 的 NVIDIA **AIPerf**（旧名 GenAI-Perf）。它们都会吐「延迟」「吞吐」，但定义、测量点、除法经常对不齐。同一张表上的两个数字，可能在说两件不同的等待。

| | 负载测试 | 性能基准 |
|---|---|---|
| 典型工具 | K6、Locust | **AIPerf** |
| 问的是 | 系统：容量、弹性伸缩、网线、资源 | 模型在给定负载下：吞吐、延迟、token 级指标 |
| 崩的时候 | 门厅、自动扩缩、连接池 | 配置、量化、batch、KV |

只做负载测试，你会不知道刀钝不钝。只做性能基准，第一个真实高峰到来时，你会发现门厅根本不够站。官方要你**两端都做**。这一页只把你推向 Benchmarking Guide。

Important 框原话方向：要学怎么打 LLM，去 **NIM for LLMs Benchmarking Guide**。本地就是 `nim-01-overview.md` 起那一组。
