---
source: https://docs.nvidia.com/nim/benchmarking/llm/latest/parameters.html
lang: zh
fetched: 2026-08-30
---

# 参数与实践

看完指标定义后，按真实部署配置测试参数和扫描范围，结果才有可比性。

## 业务场景（决定 ISL / OSL）

业务决定输入长度 ISL 和输出长度 OSL。它们影响：输入处理有多快、KV cache 怎么建、生成有多快。

- ISL 更长 → prefill 更吃显存，TTFT 更大
- OSL 更长 → 生成更吃显存/带宽，ITL 更大

要优化硬件利用率，先摸清线上输入/输出分布。

常见场景的大致 ISL/OSL：

- **翻译**（语言或代码）：ISL 和 OSL 接近，大约各 500–2000 token
- **生成**（代码、故事、邮件、搜索生成）：OSL 约 1000，ISL 约 100
- **摘要**（检索、思维链、多轮对话）：ISL 约 1000，OSL 约 100

有线上流量的话，直接用真实 prompt 最好。

## 负载怎么控

**Concurrency N（并发）**：同时在途的客户端数，每个客户端一个活跃请求。某个请求完成后立刻再发一个，系统里始终保持 N 个活跃请求。这是描述和控制系统负载最常用的方式。

**Max batch size（最大 batch）**：引擎同一时刻真正在算的那一组请求，可以是并发请求的子集。若 `concurrency > max_batch_size × 副本数`，多出来的请求会排队，TTFT 会因排队变大。

**Request rate（请求速率）**：按到达速率发请求。恒定速率 `r` 表示每 `1/r` 秒发 1 个；泊松到达则设定平均间隔。

AIPerf 两种都支持。**大多数基准测试请用 concurrency**：用 request rate 时，到达超过吞吐，在途请求会无限堆积。

扫描并发：从 1 扫到略大于 max batch size。超过 max batch 后请求排队，吞吐往往在 max batch 附近饱和，延迟仍会继续涨。

## 其他参数

**`ignore_eos`：** 真实使用应尊重 EOS、生成完就停。**做基准测试时请设为 `true`**，让模型一直生成到 `max_tokens`，OSL 才能对齐，测量才稳定。

**采样 vs greedy：** 采样策略会影响生成速度。Greedy 直接取最大 logit，不用归一化和排序。同一套基准里采样方法必须固定。细节见 Hugging Face generation strategies。
