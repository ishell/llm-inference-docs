---
source: https://docs.vllm.ai/en/stable/benchmarking/cli/
lang: zh
fetched: 2026-08-30
---

# Benchmark CLI — vLLM（中文摘译）

英文全文（很长，含全部数据集）：`en/vllm/docs/benchmark-cli.md`  
原文：https://docs.vllm.ai/en/stable/benchmarking/cli/

本页文档偏 **功能回归 / 特性评测**。生产 serving 官方更推荐 **GuideLLM**（进度条、自动报告、数据集和流量形态更灵活）。

## 在线压测（先起服务）

```bash
vllm serve NousResearch/Hermes-3-Llama-3.1-8B

vllm bench serve \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path <path>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10
```

成功时会打印：Successful requests、duration、input/output tokens、request throughput、output token throughput、Mean/Median/P99 **TTFT**、**TPOT**（不含首 token）、**ITL**。

这些延迟是在 **benchmark 客户端**测的。

## 负载怎么打

几个参数经常一起用：

- `--request-rate`：默认 `inf`，请求立刻发完，打最大吞吐。设成有限值则按到达过程发（默认 Poisson，`--burstiness=1.0`）。burstiness &lt; 1 更突发，&gt; 1 更均匀。只在 rate 不是 inf 时生效。
- `--max-concurrency`：默认不限制。用来模拟网关/负载均衡限制的并发连接。

**生产最常用：最大吞吐模式**

```
--request-rate inf --max-concurrency <N>
```

模拟前面有限流器、后面引擎能吃多少就吃多少。

也有 **offline throughput** 模式（`vllm bench throughput`），测离线批处理，不是 HTTP serving。

## 数据集

ShareGPT、Random（合成）、Prefix Repetition、HuggingFace 上的 GSM8K/HumanEval/VisionArena 等，以及自定义 jsonl。完整表见英文稿。HuggingFace 数据集要把 `--dataset-name` 设为 `hf`，本地路径再用 `--hf-name` 标 Hugging Face ID。

## 和 NVIDIA AIPerf 的关系

`vllm bench serve` 是 vLLM 自带客户端。NVIDIA 系列文档用的是 GenAI-Perf / AIPerf，打的是同一类 OpenAI 兼容接口。指标名字接近，**公式仍可能不同，不要直接横比数字**。
