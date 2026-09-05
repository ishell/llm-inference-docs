---
source: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/performance_tuning.html
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Triton：把训好的模型调到能上线

给定一个已经训好的模型，怎样用 Triton 以接近最优的 `config.pbtxt` 送到生产。LLM 生成式那条线请走 AIPerf + TensorRT-LLM / vLLM 手册。这一页是 **Triton 传统 serving**：ONNX、TensorRT、PyTorch、Python backend。秒表是 Perf Analyzer，搜配置的是 Model Analyzer。

概念课见 Triton Conceptual Guide。从没碰过 Triton 可以先 Quickstart，再回来做下面这条。

## 总流程

1. **后端认不认？** 落在官方支持的 backend 里，按 Quickstart 部署。ONNX Runtime 和 TensorRT 可以 AutoComplete：`config.pbtxt` 不是必须，除非你要显式钉参数。`--log-verbose=1` 会在日志里打印 Triton 内部看到的完整 config。其它 backend 先写 Minimal Model Configuration。
2. **不在支持列表？** Python Backend 用普通脚本接请求，快，但不一定快。C++ 自定义 backend 更重、通常更快。用 Python 换来的是人时，用 C++ 换来的是延迟。先问这笔买卖值不值。
3. **能不能推理？** `perf_analyzer -m my_model`。简化输出像：

   ```
   Concurrency: 1, throughput: 482.8 infer/sec, latency 12613 usec
   ```

   这是 sanity：输入对得上、输出回得来。失败且日志说不清时，先对 `config.pbtxt` 的输入输出名字和 dtype；再在原框架里跑一遍。没有自己的脚本就用 Polygraphy（ONNX Runtime / TensorRT / TensorFlow 1.x）。
4. **「好」是什么？** 吞吐、延迟、GPU 利用率，每个业务自己定。`config.pbtxt` 里能拧的变量很多。模型、配置、用例一变，就再打一遍 Perf Analyzer。
5. **怎么变好？** Model Analyzer 自动或手动搜 instance 数、dynamic batching、`max_batch_size`。把打赢的 config 拷回模型仓库，再测。backend 私有旋钮（例如 ONNX Runtime 的并行度）不在自动搜索里，用 Manual Configuration Search。更细的优化见 Triton Optimization 文档。

## 另外两件常被问的

**冷启动很慢。** 加载时跑 ModelWarmup，暖好了再标 READY。第一位客人不应承担 JIT 和缓存的学费。

**GPU 并没有明显更快。** 官方 backend 多数默认就走 GPU。再往上：Framework Specific Optimizations；或整模转到 TensorRT。若这些都救不了，模型也许更属于 CPU，OpenVINO backend 是那条路。不是所有网络都该被逼着住在 GPU 上。

## 端到端：`densenet_onnx`

ONNX 当作「从大多数框架都能出口」的例子。数字**绑在当时那台机器上**。官方自己写了 Warning。

建仓库并下载：

```bash
mkdir -p ./models/densenet_onnx/1
wget -O models/densenet_onnx/1/model.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/densenet-121/model/densenet-7.onnx
```

最小 `config.pbtxt`（22.07 起 ONNX 可跳过这一步，靠 AutoComplete；要钉参数再写）：

```
name: "densenet_onnx"
backend: "onnxruntime"
max_batch_size: 0
input: [
  { name: "data_0", data_type: TYPE_FP32, dims: [ 1, 3, 224, 224] }
]
output: [
  { name: "prob_1", data_type: TYPE_FP32, dims: [ 1, 1000, 1, 1 ] }
]
```

`max_batch_size: 0` 是因为这个模型把 batch 写死在 dims 第一维。支持动态 batch 的模型，Model Analyzer 才会去拧 `max_batch_size`。

起服务（原页容器标签 `26.07`；`-v $PWD:/mnt` 把宿主机当前目录挂进容器）：

```bash
docker run -ti --rm --gpus=all --network=host -v $PWD:/mnt \
  --name triton-server nvcr.io/nvidia/tritonserver:26.07-py3
tritonserver --model-repository=/mnt/models
```

日志里应看到 `densenet_onnx` version 1 **READY**。

另一只壳开 SDK 容器打基线：

```bash
docker run -ti --rm --gpus=all --network=host -v $PWD:/mnt \
  --name triton-client nvcr.io/nvidia/tritonserver:26.07-py3-sdk
perf_analyzer -m densenet_onnx --concurrency-range 1:4
```

远程改 `-u 127.0.0.1:8000`。原页示例数量级：concurrency 1→4，吞吐从约 265 升到约 965 infer/s，延迟从约 3.8 ms 变到约 4.1 ms（中间 concurrency=2 时延迟反而更低）。具体数字随机器变。

## Model Analyzer

SDK 里预装了它，也能连远程 Triton。原页为了省事，装进 **server 容器** 走默认 `local` 模式（它自己拉起 Triton）。其它连接方式见 `--triton-launch-mode`。

```bash
docker exec -ti triton-server bash
# 停掉已有 tritonserver，Analyzer 要自己起一份
kill $(ps | grep tritonserver | awk '{ print $1 }')

pip install --upgrade pip
pip install triton-model-analyzer wkhtmltopdf

model-analyzer profile \
  --model-repository=/mnt/models \
  --profile-models=densenet_onnx \
  --output-model-repository-path=results

model-analyzer analyze --analysis-models=densenet_onnx
```

原页说这一轮大约 10 分钟。示例结论：51 次测量、6 套配置，`densenet_onnx_config_3` 最好，**323 infer/s**，相对默认 **168 infer/s 约 +92%**。

| Config | Max batch | Dynamic batching | Instance | p99 (ms) | infer/s | Max GPU mem (MB) | GPU util % |
|---|---|---|---|---|---|---|---|
| config_3 | 0 | On | 4/GPU | 35.8 | 323 | 3695 | 58.6 |
| config_2 | 0 | On | 3/GPU | 59.6 | 296 | 3615 | 58.9 |
| config_4 | 0 | On | 5/GPU | 69.9 | 291 | 3966 | 58.2 |
| default | 0 | Off | 1/GPU | 12.7 | 168 | 3116 | 51.3 |

这台机器上 4 个 GPU instance 拿到最高吞吐，p99 也几乎最低。5 个反而退步。更小的 GPU 上加 instance 可能完全没有这笔赚头。CPU / GPU / 内存换一套，表会重写。**在接近生产的硬件上 profile。**

把赢家拷回去：

```bash
cp /mnt/models/densenet_onnx/config.pbtxt /tmp/original_config.pbtxt   # 可选备份
cp ./results/densenet_onnx_config_3/config.pbtxt /mnt/models/densenet_onnx/
```

有时最高吞吐和最低延迟不是同一套配置。读完 Analyzer 的报告再选座位，不要只看表头那一行。

再往后的手工拧法：Model Configuration 与 Optimization 文档。Perf Analyzer 用来确认「拷回去之后，数字确实动了」。
