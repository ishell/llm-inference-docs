---
source: https://docs.nvidia.com/nim/large-language-models/latest/reference/benchmarking.html
lang: zh
fetched: 2026-08-31
---

# Benchmarking — NVIDIA NIM 产品页（短入口）

完整流程看本目录 `nim-01`～`nim-04`（NIM LLMs Benchmarking Guide）。

- 成本取决于「还能响应的前提下每秒多少请求」。精度达标后再谈成本。
- 工具：Locust / K6（通用压测）vs NVIDIA AIPerf（LLM token 指标）。定义经常对不齐。
- **负载测试（K6）**：容量、弹性伸缩、网络、资源。
- **性能基准（AIPerf）**：吞吐、延迟、token 级指标。

两端都要做。本页只是指向完整指南。
