---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/benchmarking-default-performance.html
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# 第 1 章：先打一条默认基线

在拧任何旋钮之前，先让引擎按默认活一次，留下四个数字：token 吞吐、request 吞吐、TTFT、ITL。后面每一页都在跟这张成绩单吵架。

数字是演示用的。你的卡、你的网、你的 2048/2048 是否真的 2048，都会改写结局。

## LLM-API：一行里完成转换和建引擎

```python
# quickstart.py
from tensorrt_llm import LLM, SamplingParams

def main():
    prompts = [
        "Hello, I am",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(
        model="meta-llama/Llama-3.3-70B-Instruct",  # HF 名即可，不必事先下载
        tensor_parallel_size=4,
    )
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(output.prompt, output.outputs[0].text)

if __name__ == "__main__":
    main()
```

多卡走 MPI。因此：

- 入口必须用 `if __name__ == "__main__"`（mpi4py 的规矩）。
- 有的环境要 `mpirun -n 1 --oversubscribe --allow-run-as-root python quickstart.py`。**`-n 1` 是故意的**：TensorRT-LLM 自己去孵其余 GPU 上的进程。单机多卡通常不必加 `mpirun`；出现 MPI 报错再加。
- Llama 若 gated：去 Hugging Face 申请，再按他们的 quickstart 在环境里登录。门没开，权重不会自己走进来。

保存引擎：

```python
from tensorrt_llm import LLM

def main():
    llm = LLM(model="/scratch/Llama-3.3-70B-Instruct", tensor_parallel_size=4)
    llm.save("baseline")

if __name__ == "__main__":
    main()
```

### CLI 两条路

1. `convert_checkpoint.py` 把 HF / NeMo 变成 TensorRT-LLM checkpoint（各模型在 `examples/` 下有自己的脚本，Llama 亦然）。
2. `trtllm-build` 吃 checkpoint，写出引擎。装 `tensorrt_llm` 时这个命令会进来。

具体花样见 NVIDIA/TensorRT-LLM 仓库里该模型的 README。手册后文用 LLM-API 说话；CLI 旗标是同一套旋钮的另一扇门。

## 用 trtllm-bench 量吞吐和延迟

### 数据集

手册造了 1000 条、每条 ISL/OSL 都是 2048 的假数据。克隆 TensorRT-LLM 后：

```bash
python benchmarks/cpp/prepare_dataset.py --stdout \
  --tokenizer /path/to/hf/Llama-3.3-70B-Instruct/ \
  token-norm-dist \
  --input-mean 2048 --output-mean 2048 \
  --input-stdev 0 --output-stdev 0 \
  --num-requests 1000 \
  > synthetic_2048_2048.txt
```

也可以喂真实请求，格式见 `trtllm-bench` 文档。

### 吞吐

可能仍需要 `mpirun -n 1 ...` 前缀。

```bash
trtllm-bench \
  --model /path/to/hf/Llama-3.3-70B-Instruct/ \
  throughput \
  --dataset /path/to/dataset/synthetic_2048_2048_1000.txt \
  --engine_dir /path/to/engines/baseline
```

这会把 1000 条请求立刻倒进引擎。`trtllm-bench throughput -h` 可调到达率和请求上限。官方内部在 4 卡 NVLink H100 上，上面这条大约 **20 分钟**。

报表里有 ENGINE DETAILS、WORLD（TP=4、默认 max batch 2048、max tokens 8192、Guaranteed No Evict、KV 90%）、PERFORMANCE OVERVIEW。基线案例（与后文表格对齐的那组）大约：

| 指标 | 基线 |
|---|---|
| Token Throughput (tokens/sec) | 1564.3 |
| Request Throughput (req/sec) | 0.7638 |
| Average TTFT (ms) | 147.7 |
| Average ITL (ms) | 31.3 |

吞吐命令打印的 1585 tok/s 与表格 1564 是同量级的两次跑；后文比较一律用表里这组。

### 延迟

延迟基准把 batch 钉在 **1**，为了把「一个人说话」测干净。时间会变得很长。例子里 `--num-requests 100 --warmup 10`，案例跑了大约 **一个半小时**。真实迭代用 **10 条** 往往已经够看方向。按你的耐心改 `--num-requests`。

```bash
trtllm-bench \
  --model /path/to/hf/Llama-3.3-70B-Instruct/ \
  latency \
  --dataset /path/to/dataset/synthetic_2048_2048_1000.txt \
  --num-requests 100 \
  --warmup 10 \
  --engine_dir /path/to/engines/baseline
```

延迟报表会拆 TTFT、ITL、generation latency 的 MIN/MAX/AVG/P90/P95/P99。Acceptance Rate 在没开 speculative 时是 1.00——没有草稿可拒。

后面每一章，都是在问：能不能让 1564 变大、让 31 ms 变小，而不把第一个字等成一场雨。
