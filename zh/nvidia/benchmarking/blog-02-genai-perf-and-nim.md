---
source: https://developer.nvidia.com/blog/llm-performance-benchmarking-measuring-nvidia-nim-performance-with-genai-perf/
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# 系列第 2 篇：用 GenAI-Perf 测 NIM

英文对照：`en/nvidia/benchmarking/blog-02-genai-perf-and-nim.md`

**今天请用 AIPerf。** GenAI-Perf 已停更，命令几乎同构；NIM 手册第 4 章是 AIPerf 版：`nim-04-aiperf.md`。本篇保留官方当时的 GenAI-Perf 流程，因为 NIM Performance 页面上那些数字，就是用这套仪式测出来的。换工具，不要换尺子的定义。

第 1 篇讲等待第一个字意味着什么。本篇把 Llama 3.1 8B Instruct 用 NIM 拉起来，让秒表真的跑起来。

你需要这些数字，通常出于三种并不浪漫的理由：找出瓶颈、在服务质量与吞吐之间做交易、决定买多少机器。GenAI-Perf 是客户端工具，报 TTFT、ITL、TPS、RPS。它打任何符合 OpenAI API 的服务。本篇的服务端是 **NVIDIA NIM**：预打包微服务，后端可以是 TensorRT-LLM 或 vLLM，带企业级的门锁。


本地图（原文版权仍归原站；学习对照用）：

![Figure 1. Sample output by genAI perf](../../../assets/nvidia/benchmarking/blog-02-genai-perf-and-nim/01-Figure-1.-Sample-output-by-genAI-perf.png)

![latency throughput curve plot](../../../assets/nvidia/benchmarking/blog-02-genai-perf-and-nim/02-latency-throughput-curve-plot.png)

## 为什么用它测

NIM 是装好的容器，云上、机房、RTX 工作站都能跑。同一代硬件上，NIM 还会继续改内核。官网上的性能表不是神话，是 GenAI-Perf 打出来的成绩单。你的卡、你的机房、你的网线，只有自己测过才算数。

## 起一个 OpenAI 兼容的 Llama-3 服务

NIM 是把模型送进生产较快的一条路。硬件和 NGC key 见 NIM LLM 文档。

```bash
export NGC_API_KEY=<YOUR_NGC_API_KEY>
export CONTAINER_NAME=llama-3.1-8b-instruct
export IMG_NAME="nvcr.io/nim/meta/${CONTAINER_NAME}:latest"
export LOCAL_NIM_CACHE=./cache/nim
mkdir -p "$LOCAL_NIM_CACHE"

docker run -it --rm --name=$CONTAINER_NAME \
  --gpus all \
  --shm-size=16GB \
  -e NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u $(id -u) \
  -p 8000:8000 \
  $IMG_NAME
```

本地目录当模型缓存。启动时容器会把需要的东西下载下来，然后在 8000 端口开门。成功时像这样：

```
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

试探一下这扇门还在不在：

```python
from openai import OpenAI
client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="not-used")
prompt = "Once upon a time"
response = client.completions.create(
    model="meta/llama-3.1-8b-instruct",
    prompt=prompt,
    max_tokens=16,
    stream=False
)
print(response.choices[0].text)
```

「从前」是实验室喜欢的开头：短、干净、人人认识。真实用户很少这么客气。

## 安装 GenAI-Perf，先打一轮热身

客户端尽量和 NIM **同机**，除非你故意要把网线请进入场。远程测到的常常是网络的性格，不是模型的性格。

```bash
export RELEASE="24.12"  # 建议最新 yy.mm
docker run -it --net=host --gpus=all -v $PWD:/workdir \
  nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
```

把当前目录挂进去，成绩才带得走。容器里：

```bash
export INPUT_SEQUENCE_LENGTH=200
export INPUT_SEQUENCE_STD=10
export OUTPUT_SEQUENCE_LENGTH=200
export CONCURRENCY=10
export MODEL=meta/llama-3.1-8b-instruct

genai-perf profile \
    -m $MODEL \
    --endpoint-type chat \
    --service-kind openai \
    --streaming \
    -u localhost:8000 \
    --synthetic-input-tokens-mean $INPUT_SEQUENCE_LENGTH \
    --synthetic-input-tokens-stddev $INPUT_SEQUENCE_STD \
    --concurrency $CONCURRENCY \
    --output-tokens-mean $OUTPUT_SEQUENCE_LENGTH \
    --extra-inputs max_tokens:$OUTPUT_SEQUENCE_LENGTH \
    --extra-inputs min_tokens:$OUTPUT_SEQUENCE_LENGTH \
    --extra-inputs ignore_eos:true \
    --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct \
    -- \
    -v \
    --max-threads=256
```

`ignore_eos` 让模型说到钟响，OSL 才可控。Llama-3 tokenizer 在 Hugging Face 上是 gated 仓库，要申请权限再登录：

```bash
pip install huggingface_hub
huggingface-cli login
```

终端里会打出一张表（原文 Figure 1）。那是热身，不是决赛。

## 扫描多种场景

基准测试通常要扫几种 ISL/OSL，和几种并发。先热身，再 `bash benchmark.sh`。

```bash
declare -A useCases
useCases["Translation"]="200/200"
useCases["Text classification"]="200/5"
useCases["Text summary"]="1000/200"
useCases["Code generation"]="200/1000"

runBenchmark() {
    local description="$1"
    local lengths="${useCases[$description]}"
    IFS='/' read -r inputLength outputLength <<< "$lengths"
    echo "Running genAI-perf for $description with ISL=$inputLength OSL=$outputLength"
    for concurrency in 1 2 5 10 50 100 250; do
        genai-perf profile \
            -m meta/llama-3.1-8b-instruct \
            --endpoint-type chat \
            --service-kind openai \
            --streaming \
            -u localhost:8000 \
            --synthetic-input-tokens-mean $inputLength \
            --synthetic-input-tokens-stddev 0 \
            --concurrency $concurrency \
            --output-tokens-mean $outputLength \
            --extra-inputs max_tokens:$outputLength \
            --extra-inputs min_tokens:$outputLength \
            --extra-inputs ignore_eos:true \
            --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
            --measurement-interval 30000 \
            --profile-export-file ${inputLength}_${outputLength}.json \
            -- \
            -v \
            --max-threads=256
    done
}

for description in "${!useCases[@]}"; do
    runBenchmark "$description"
done
```

`--measurement-interval 30000` 是每个测量窗口（毫秒）。窗口里要能结束足够多的请求。70B、并发 250 这种晚上，把窗口拉到 100000 ms（100 秒）。秒表也需要耐心。

## 读结果

默认写在 `artifacts/`，按模型、并发、ISL/OSL 分目录。主文件是 `*_genai_perf.csv`。用 pandas 抽出某场景（例如 200/5）在各并发下的 RPS 和 TTFT：

```python
import os
import pandas as pd

root_dir = "./artifacts"
directory_prefix = "meta_llama-3.1-8b-instruct-openai-chat-concurrency"
concurrencies = [1, 2, 5, 10, 50, 100, 250]
RPS, TTFT = [], []

for con in concurrencies:
    df = pd.read_csv(os.path.join(root_dir, directory_prefix+str(con), "200_5_genai_perf.csv"))
    RPS.append(float(df.iloc[8]["avg"].replace(",", "")))
    TTFT.append(float(df.iloc[0]["avg"].replace(",", "")))
```

画延迟–吞吐曲线，每个点标并发：

```python
import matplotlib.pyplot as plt
plt.plot(TTFT, RPS, "x-")
plt.xlabel("TTFT(ms)")
plt.ylabel("RPS")
for i, label in enumerate([1, 2, 5, 10, 50, 100, 250]):
    plt.annotate(label, (TTFT[i], RPS[i]), textcoords="offset points", xytext=(0,10), ha="center")
```

原文 Figure 2：横轴 TTFT，纵轴系统吞吐，点上写并发。

- **有延迟预算**：取可接受的最大 TTFT，对应的纵轴和并发就是该预算下最高吞吐。用户愿意等这么久，你最多能卖这么快。
- **有目标并发**：找到那个点，读出该负载下的延迟和吞吐。今晚有这么多人进店，菜会慢成什么样。

图上也能看见延迟陡增、吞吐几乎不再涨的并发。NVIDIA 示例里是 **`concurrency=50`**。再往上加人，只是让队列变长。横轴也可以换成 ITL、e2e_latency 或 TPS_per_user——同一座山，不同的登山口。

## 定制模型 / LoRA

通用问答、会议摘要，基座模型往往够用。公司内部的黑话、产品目录、流程，常常需要 LoRA 这种低成本的裁缝。NIM 能加载多个 adapter（NeMo 训练的，或 Hugging Face PEFT）。目录结构见 Parameter-Efficient Fine-Tuning。加载后，把 `model` 换成 LoRA 名字即可：

```bash
curl -X POST http://0.0.0.0:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama3-8b-instruct-lora_vhf-math-v1",
    "prompt": "John buys 10 packs of magic cards. Each pack has 20 cards and 1/4 of those cards are uncommon. How many uncommon cards did he get?",
    "max_tokens": 128
  }'
```

GenAI-Perf 用 `-m` 一次塞多个 ID：

```bash
genai-perf profile \
    -m llama-3-8b-lora_1 llama-3-8b-lora_2 llama-3-8b-lora_3 \
    --model-selection-strategy random \
    --endpoint-type completions \
    --service-kind openai \
    --streaming
```

`--model-selection-strategy {round_robin,random}`：轮流叫，还是随机叫。多 adapter 时，流量会像一副洗过的牌。

## 小结

第 1 篇对齐尺子，本篇让 NIM 在你的硬件上留下自己的曲线。第 3 篇离开 HTTP，直接用 `trtllm-bench` 调引擎；第 4 篇把曲线变成钱。
