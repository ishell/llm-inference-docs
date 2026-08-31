---
source: https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/
lang: zh
fetched: 2026-08-30
---

# 系列第 3 篇：用 TensorRT-LLM 调性能（中文摘译）

原文：https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/

先把 GPU 拉回默认功耗策略，再查/设功耗上限（`nvidia-smi`）。

## trtllm-bench

不经过完整 HTTP serving，直接打引擎。PyTorch flow 示例：

```bash
trtllm-bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset dataset.jsonl \
  --tp 1 \
  --backend pytorch \
  --report_json results.json \
  --streaming \
  --concurrency $CONCURRENCY
```

自定义数据集每行一个 json，例如：

```json
{"task_id": 1, "prompt": "...", "output_tokens": 128}
```

输出里重点看 **PERFORMANCE OVERVIEW**：Request Throughput、Total Output Throughput、TTFT、TPOT、Per User Output Speed。以及 **Max Runtime Batch Size** / **Max Runtime Tokens**（即 `max_batch_size` / `max_num_tokens`）。

- max tokens：一轮 batch 里能处理的 token 上限（所有 context 请求的输入 token 之和 + 每个 generation 请求各 1 token）
- max batch size：一轮最多多少个请求。先撞到 batch size 就会卡住，哪怕 token 预算没用完。

用 `--concurrency` 扫出「每 GPU 吞吐 vs 每用户速度」曲线。文中例子：用户体验目标约 50 tok/s/user（约 20ms/token）。Llama-3.1 8B FP16 大约 256 并发还能维持 ~72 tok/s/user；FP8 量化能在同样延迟预算里撑到 512 并发 ~66 tok/s/user。

多卡用 `--tp` / `--pp` / `--ep`。高级项走 `--extra_llm_api_options`。

## 调完用 trtllm-serve 上线

`trtllm-serve` 默认不做你刚扫出来的那套配置，要手动带上：

```bash
trtllm-serve serve nvidia/Llama-3.1-8B-Instruct-FP8 \
  --backend pytorch \
  --max_num_tokens 7680 \
  --max_batch_size 3840 \
  --tp_size 1 \
  --extra_llm_api_options llm_api_options.yml
```

yml 里可配 CUDA graph 的 `max_batch_size` 和 padding。起来之后再用 GenAI-Perf/AIPerf 或 `benchmark_serving.py` 验证。

更细的旋钮见 TensorRT-LLM Performance Tuning Guide：https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/
