---
source: https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# trtllm-bench：手册里的秒表

英文全文抓取：`en/nvidia/performance-tuning/trtllm-bench.md`（官方页很长，含 LoRA、多模态、Medusa 低延迟引擎）。这一页是中文导读：怎么用这把尺子去配合调优手册。手册第 1 章已经用它打过 70B 基线。博客实操见 `blog-03-tensorrt-llm.md`。

官方声明：这套 CLI **仍在演进，API 可能破**。它想让你更容易复现他们 Performance Overview 上的数字。全部基准都走 **in-flight batching**。

可从 Hugging Face 自动拉模型：把 token 放到 `HF_TOKEN`。量化方面，bench 目前只帮你配一个子集：**None / FP8 / NVFP4**。引擎本身支持的量化更多，只是这把 CLI 还没全包。

## 严谨测试前的 GPU

不是日常推理的硬性要求；要发论文或横比数字时再上：

- `sudo nvidia-smi -pm 1` — persistence mode
- `sudo nvidia-smi -rgc` — 让时钟自己调；锁满频有时会热节流，反而更慢
- 查最大功耗再 `sudo nvidia-smi -pl <max>`
- 若支持 boost slider：`sudo nvidia-smi boost-slider --vboost <max>`

## 三条路

1. 准备数据集（jsonl，一行一条完整 JSON）。
2. `trtllm-bench build` 造引擎——**PyTorch flow 不必这一步**。
3. `throughput` 打满，或 `latency` 测低延迟。

Quickstart（Llama-3.1-8B，ISL/OSL 128/128，3000 条，FP8）：

```bash
python benchmarks/cpp/prepare_dataset.py --stdout \
  --tokenizer meta-llama/Llama-3.1-8B \
  token-norm-dist \
  --input-mean 128 --output-mean 128 \
  --input-stdev 0 --output-stdev 0 \
  --num-requests 3000 \
  > /tmp/synthetic_128_128.txt

trtllm-bench --model meta-llama/Llama-3.1-8B build \
  --dataset /tmp/synthetic_128_128.txt --quantization FP8

trtllm-bench --model meta-llama/Llama-3.1-8B throughput \
  --dataset /tmp/synthetic_128_128.txt \
  --engine_dir /tmp/meta-llama/Llama-3.1-8B/tp_1_pp_1
```

报表分三块：ENGINE DETAILS、WORLD + RUNTIME（TP/PP、max batch/tokens、Guaranteed No Evict、KV 90%）、PERFORMANCE OVERVIEW（token/s、req/s、total latency）。

## 数据集

| Key | 必填 | 类型 | 含义 |
|---|---|---|---|
| `task_id` | 是 | String | 请求 ID |
| `prompt` | 二选一 | String | 文本 prompt |
| `input_ids` | 二选一 | List[int] | 已经 tokenize 的 id |
| `output_tokens` | 是 | Integer | 要生成多少 token |

`prompt` 和 `input_ids` 不能同时当真：有 `input_ids` 就忽略 `prompt`。必须一行一条完整 JSON。

## `build`

不手动给 `max_batch_size` / `max_num_tokens` 时，会按数据集的平均 ISL/OSL、最大序列长做启发式。日志里能看见它打开 `multiple_profiles`、`use_paged_context_fmha`——和手册第 2 章的建议对齐。

也可以自己钉：

```bash
trtllm-bench --model meta-llama/Llama-3.1-8B build \
  --quantization FP8 --max_seq_len 4096 \
  --max_batch_size 1024 --max_num_tokens 2048
```

不指定则默认 2048 / 8192。并行：`--tp_size` × `--pp_size`，world size **≤ 8**。

引擎目录出现在日志末尾，例如 `/tmp/meta-llama/Llama-3.1-8B/tp_1_pp_1`。

## `throughput`

把整个数据集尽快丢进 Executor（offline，打满）。等全部返回，再算统计。这是**上限吞吐**，不是线上到达率。手册第 1 章那条 1000×2048/2048 也是这种倒法；可用 `throughput -h` 改到达率和请求上限。

## PyTorch flow

不必 `build`。`throughput` 会按 `--dataset`（或你手写的 max batch/tokens）初始化 `tensorrt_llm._torch`。CUDA graph 默认开。额外配置：`--extra_llm_api_options path/to.yaml`。本地权重用 `--model_path`；`--model` 仍要填，给报表和启发式查表。

## `latency`

工作流类似，但引擎要**另外建**，通常 `max_batch_size 1`。官方还示范了非 Medusa 的 FP8 70B 低延迟引擎（关掉 paged context FMHA、打开 multiple profiles / reduce fusion），以及带 Medusa 头的投机解码引擎（NVIDIA Hugging Face 上有预量化 checkpoint，choices 用一份 YAML 树）。环境变量那一串（MMHA multi-block、PDL）是他们低延迟配方的一部分，细节以英文全文为准。

手册第 1 章把 latency 钉在 batch 1，100 条大约一个半小时；迭代时 10 条往往够看方向。

## 命令速查

| 场景 | 阶段 | 命令 |
|---|---|---|
| 数据集 | 准备 | `prepare_dataset.py ... token-norm-dist ... > $DATASET_PATH` |
| 吞吐 | 建引擎 | `trtllm-bench --model $HF_MODEL build --dataset $DATASET_PATH` |
| 吞吐 | 打 | `trtllm-bench --model $HF_MODEL throughput --dataset $DATASET_PATH --engine_dir $ENGINE_DIR` |
| 延迟 | 建引擎 | 见英文页 low latency 一节 |
| 非 Medusa 延迟 | 打 | `trtllm-bench --model $HF_MODEL latency --dataset $DATASET_PATH --engine_dir $ENGINE_DIR` |
| Medusa 延迟 | 打 | 同上，加 `--medusa_choices $MEDUSA_CHOICES` |

LoRA、多模态、PyTorch 量化细项在英文抓取里。调优手册要的是：同一把尺子，先打基线，再每拧一章旋钮打一次。
