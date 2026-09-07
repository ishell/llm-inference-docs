---
source: https://docs.nvidia.com/nim/large-language-models/latest/reference/benchmarking.html
lang: zh
voice: book-zh
fetched: 2026-09-06
---

# Benchmarking — NVIDIA NIM 产品页

NIM for LLMs 产品文档里的短入口（抓取时产品版本标 2.0.11）。它不教怎么打命令。完整流程在同目录 `nim-01`～`nim-05`；尺子本身在 `../tools/aiperf.md`。

生成式应用铺开以后，钱花在「用户还愿意等、还愿意看」的前提下，每秒能完成多少请求。**精度达标之前不要谈成本。** 本页不覆盖 accuracy。

## 两把尺子，不要混用

市面上能打 LLM 的客户端很多：老牌 Locust / K6，以及专门认 token 的 NVIDIA **AIPerf**（旧名 GenAI-Perf）。它们都会吐「延迟」「吞吐」，但定义、测量点、除法经常对不齐。同一张表上的两个数字，可能在说两种不同的等待。

| | 负载测试 | 性能基准 |
|---|---|---|
| 典型工具 | K6、Locust | **AIPerf** |
| 问的是 | 系统：容量、弹性伸缩、网络、资源 | 模型在给定负载下：吞吐、延迟、token 级指标 |
| 出问题时 | 排队、自动扩缩、连接池 | 配置、量化、batch、KV |

只做负载测试，我们看不出模型本身够不够快。只做性能基准，真实高峰到来时系统可能先撑不住。官方要求**两端都做**。这一页只把我们推向 Benchmarking Guide。

Important 框原话方向：要学怎么打 LLM，去 **NIM for LLMs Benchmarking Guide**。本地就是 `nim-01-overview.md` 起那一组。
