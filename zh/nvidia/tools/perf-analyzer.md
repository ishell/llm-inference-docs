---
source: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/perf-analyzer-README.html
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Triton Performance Analyzer

测 **Triton 上传统模型**（分类、检测、embedding 这类一次进一次出）性能的 CLI。你改优化策略，它告诉你吞吐和延迟动了没有。

**LLM / 生成式请用 AIPerf**（旧名 GenAI-Perf）。那把尺子认识 token 和流式。Perf Analyzer 认识的是 infer/sec 和客户端平均 batch 延迟。GenAI-Perf 底下仍会叫它；你不必直接对 GPT 挥这把刀。

官方页对应 Triton 用户指南里一个较旧的点（抓取时目录写 2.65.0；Triton 当前发布号会走得更前）。流程没变。

## 负载怎么发

- **Concurrency**：维持 N 条在途请求。门口永远站着这么多人。
- **Request rate**：按指定速率连发。
- **Custom interval**：按你给的间隔序列发。

## 成绩怎么收

- **Time windows**：按时间窗反复测，直到判定稳态。
- **Count windows**：按「这么多条请求」为一窗，同样等到稳态。

还支持 sequence、ensemble、decoupled（输出次数和输入不必 1:1）。输入可自动生成，也可自备，并可校验输出。

## 五步 quickstart

镜像标签用 `yy.mm`，例如 `23.02`。LLM 工具链常用更新的 SDK 标签；这里跟原页。

**1. Triton 容器**

```bash
export RELEASE=<yy.mm>
docker pull nvcr.io/nvidia/tritonserver:${RELEASE}-py3
docker run --gpus all --rm -it --net host \
  nvcr.io/nvidia/tritonserver:${RELEASE}-py3
```

**2. 示例模型 `simple`**

```bash
git clone --depth 1 https://github.com/triton-inference-server/server
mkdir model_repository
cp -r server/docs/examples/model_repository/simple model_repository
```

**3. 起服务**

```bash
tritonserver --model-repository $(pwd)/model_repository &> server.log &
curl -v localhost:8000/v2/health/ready   # 期望 HTTP/1.1 200 OK
# 从容器 detach：CTRL-p CTRL-q
```

**4. SDK 容器（里面有 `perf_analyzer`）**

```bash
docker pull nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
docker run --gpus all --rm -it --net host \
  nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
```

**5. 打**

```bash
perf_analyzer -m simple
```

远程服务加 `-u host:8000`。扫并发见 Triton 调优页里的 `--concurrency-range`。

输出怎么读、稳态判定的细节，官方让你去「full quick start」。贡献与提问流程在原页底部；需要最小可复现时走 Stack Overflow 那套 MCVE。

把训好的模型从「能 load」调到「能上线」，整条仪式在 `triton-performance-tuning.md`：Perf Analyzer 打基线，Model Analyzer 搜 `config.pbtxt`，再打一遍。
