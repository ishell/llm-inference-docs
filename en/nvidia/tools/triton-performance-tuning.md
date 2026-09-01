---
source: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/performance_tuning.html
lang: en
fetched: 2026-09-01
---

# Triton: deploy and tune a trained model

Given a trained model, how to serve it on Triton with a near-optimal `config.pbtxt`. Generative LLMs belong on AIPerf plus the TensorRT-LLM / vLLM guides. This page is **classic Triton serving**: ONNX, TensorRT, PyTorch, Python backend. The stopwatch is Perf Analyzer; the config search is Model Analyzer.

Conceptual extra: Triton Conceptual Guide. If Triton is new, do the Quickstart first, then this path.

## Flow

1. **Does a backend accept it?** If it is a supported backend, deploy as in Quickstart. ONNX Runtime and TensorRT can AutoComplete: `config.pbtxt` is optional unless you want to pin values. `--log-verbose=1` prints the full config Triton actually sees. Other backends start from the Minimal Model Configuration.
2. **Not on the list?** Python Backend is a generic script — simple, not always fast. A custom C++ backend is heavier and usually quicker. Python buys engineer time; C++ buys latency. Ask whether the trade is worth it.
3. **Can it infer?** `perf_analyzer -m my_model`. Simplified:

   ```
   Concurrency: 1, throughput: 482.8 infer/sec, latency 12613 usec
   ```

   Sanity: inputs match, outputs return. If it fails and the log is unclear, check `config.pbtxt` names/dtypes, then run in the original framework. No script of your own: Polygraphy (ONNX Runtime / TensorRT / TensorFlow 1.x).
4. **What is “good”?** Throughput, latency, GPU util — each product decides. Many knobs live in `config.pbtxt`. When the model, config, or use case moves, measure again.
5. **How to improve?** Model Analyzer searches instance count, dynamic batching, `max_batch_size` (auto or manual). Copy the winner back to the model repo and measure. Backend-private knobs (ONNX Runtime parallelism, etc.) are not in automatic search — use Manual Configuration Search. Deeper reading: Triton Optimization docs.

## Two frequent questions

**Slow on first load.** Run ModelWarmup while loading so the model is READY only after it is warm. The first guest should not pay for JIT and caches.

**GPU is not much faster.** Most official backends already prefer GPU. Next: Framework Specific Optimizations, or convert the whole model to TensorRT. If none of that helps, the network may belong on CPU; the OpenVINO backend is that path. Not every graph should be forced to live on a GPU.

## End-to-end: `densenet_onnx`

ONNX as “most frameworks can export this.” Numbers are **tied to the machine they used**. The page itself warns about that.

Create the repo and download:

```bash
mkdir -p ./models/densenet_onnx/1
wget -O models/densenet_onnx/1/model.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/densenet-121/model/densenet-7.onnx
```

Minimal `config.pbtxt` (since 22.07 ONNX can skip this and AutoComplete; write it only to pin values):

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

`max_batch_size: 0` because this model hard-codes batch in the first dim. Dynamic-batch models are where Model Analyzer also tunes `max_batch_size`.

Start the server (page tag `26.07`; `-v $PWD:/mnt` mounts the host cwd):

```bash
docker run -ti --rm --gpus=all --network=host -v $PWD:/mnt \
  --name triton-server nvcr.io/nvidia/tritonserver:26.07-py3
tritonserver --model-repository=/mnt/models
```

Logs should show `densenet_onnx` version 1 **READY**.

In another shell, SDK container for a baseline:

```bash
docker run -ti --rm --gpus=all --network=host -v $PWD:/mnt \
  --name triton-client nvcr.io/nvidia/tritonserver:26.07-py3-sdk
perf_analyzer -m densenet_onnx --concurrency-range 1:4
```

Remote: `-u 127.0.0.1:8000`. Their ballpark: concurrency 1→4, throughput ~265 to ~965 infer/s, latency ~3.8 ms to ~4.1 ms (concurrency=2 was actually lower latency). Hardware changes the table.

## Model Analyzer

It ships in the SDK and can attach to a remote Triton. The page installs it **inside the server container** and uses default `local` mode (Analyzer starts its own Triton). Other attach modes: `--triton-launch-mode`.

```bash
docker exec -ti triton-server bash
# stop the existing tritonserver; Analyzer will start its own
kill $(ps | grep tritonserver | awk '{ print $1 }')

pip install --upgrade pip
pip install triton-model-analyzer wkhtmltopdf

model-analyzer profile \
  --model-repository=/mnt/models \
  --profile-models=densenet_onnx \
  --output-model-repository-path=results

model-analyzer analyze --analysis-models=densenet_onnx
```

They clocked this example at ~10 minutes. Result: 51 measurements, 6 configs, `densenet_onnx_config_3` wins at **323 infer/s**, about **+92%** vs default **168 infer/s**.

| Config | Max batch | Dynamic batching | Instance | p99 (ms) | infer/s | Max GPU mem (MB) | GPU util % |
|---|---|---|---|---|---|---|---|
| config_3 | 0 | On | 4/GPU | 35.8 | 323 | 3695 | 58.6 |
| config_2 | 0 | On | 3/GPU | 59.6 | 296 | 3615 | 58.9 |
| config_4 | 0 | On | 5/GPU | 69.9 | 291 | 3966 | 58.2 |
| default | 0 | Off | 1/GPU | 12.7 | 168 | 3116 | 51.3 |

On that box, 4 GPU instances took the highest throughput and almost the lowest p99. 5 got worse. A smaller GPU may not pay for extra instances at all. Change CPU / GPU / RAM and the table rewrites. **Profile on hardware that looks like production.**

Copy the winner back:

```bash
cp /mnt/models/densenet_onnx/config.pbtxt /tmp/original_config.pbtxt   # optional backup
cp ./results/densenet_onnx_config_3/config.pbtxt /mnt/models/densenet_onnx/
```

Highest throughput and lowest latency are not always the same config. Read the Analyzer report before picking a seat; do not trust only the first row.

Further manual knobs: Model Configuration and Optimization docs. Perf Analyzer is how you confirm the numbers actually moved after the copy-back.
