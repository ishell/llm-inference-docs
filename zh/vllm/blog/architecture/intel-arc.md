---
source: https://vllm.ai/blog/2025-11-11-intel-arc-pro-b
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Intel Arc Pro B：把 MoE 专家塞进一张消费级卡

英文对照：[en/vllm/blog/architecture/intel-arc.md](../../../../en/vllm/blog/architecture/intel-arc.md)  
原文：https://vllm.ai/blog/2025-11-11-intel-arc-pro-b  
2025-11-11。署名 **Intel vLLM Team**。XPU / SYCL。数字是当时 **4–8 张 Intel Arc Pro B60** 上的演示，不是 SLA。Sleep 见 [sleep-mode](sleep-mode.md)；投机见 [spec-decode](../performance/spec-decode.md)；卡从主干请出去见 [hardware-plugin](hardware-plugin.md)。这张卡上的 W4A16 走 AutoRound：[autoround-llmc](autoround-llmc.md)；CPU 亲戚：[arm-cpus](arm-cpus.md)；`torch.compile` 的 FP16/BF16：[torch-compile](torch-compile.md)。镜像 `intel/vllm:0.10.2-xpu`。当时宿主 Ubuntu 25.04、KMD 6.14.0。MoE / gpt-oss 从 **0.10.2** XPU 镜像起。

适用：DeepSeek 蒸馏 / Qwen / Llama / GPT-OSS 在 Arc Pro B60 上 serve，persistent MoE kernel、TP、`--enforce-eager`。不适合：把表里的 **1210.74 / 1495.12 tok/s** 当承诺——Intel 自己的免责声明就印在页上。

[Intel Arc Pro B-Series](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/workstations/b-series/overview.html)：专业卡、大显存、能叠多卡，性价比叙事是本地跑大模型。vLLM 是他们点名的 serving 核。几个月 Intel × vLLM 把功能、多卡、PCIe P2P 往上送。

页上的卡规格：**24 GB** 显存、**456 GB/s** 带宽、**160** 个 Intel XMX。当时支持的模型表：[intel/ai-containers vllm 0.10.2-xpu](https://github.com/intel/ai-containers/blob/main/vllm/0.10.2-xpu.md#supported-models)。

原文列的功能：

- DeepSeek 蒸馏的 Llama / Qwen，输出 token 吞吐像样
- 长上下文 **>50K**，batch size 能涨
- Embedding、reranker、pooling
- 多模态
- MoE（GPT-OSS、DeepSeek-v2-lite、Qwen3-30B-A3B 等）
- 逐层在线量化（省显存）
- DP / TP / PP
- `torch.compile` 的 FP16 / BF16 路径
- 投机解码：n-gram、EAGLE、EAGLE3
- Async scheduling
- Prefill/Decode 分离
- LoRA
- Reasoning output
- Sleep mode
- Structured outputs
- Tool calling
- 混精度：BF16、FP16、INT4、FP8 菜谱

## MoE：不要一只一只 GEMM

MoE：gate 按 token 挑一部分专家 FFN。等价计算是许多并行 GEMM，结构化稀疏。GEMM 和 Flash Attention 之外，专家 + gate 才是大头。

![moe](../../../../assets/vllm/blog/architecture/intel-arc/01-moe.png)

**Figure。** MoE 专家和 gating（学习对照；版权仍归原站）。

朴素路径：每轮 `for` 循环按专家 launch 一只 GEMM——launch 税、调度延迟。Gate 算完 GEMM 才能开 → 流水线卡住，设备空转。

他们做 **persistent zero-gap kernel**，号称 Arc Pro B60 上超过硬件容量 **80%**。

### 1. 单 kernel、persistent loop

一次 launch；persistent loop 让 launch 参数不必等路由网络。设备并行度留着。

之前：host 在等，设备空转。

![persistent kernel1](../../../../assets/vllm/blog/architecture/intel-arc/02-persistent-kernel1.png)

**Figure。** 持久化之前的 kernel trace——host 等、设备闲（学习对照）。

之后：设备一直忙。

![persistent kernel2](../../../../assets/vllm/blog/architecture/intel-arc/03-persistent-kernel2.png)

**Figure。** Persistent loop 把设备占住（学习对照）。

B60 有 **20 个 XeCore**，资源一样，每核能塞多组 SYCL group。设计：**每 XeCore 两组**，在算力和带宽之间找平衡。

### 2. 计算组动态抢活

专家路由不均，各组活不一样。固定 stride：最慢的那组定节奏；间隙能攒到 MoE GEMM 总时间的约 **15%**。更好的办法：谁先做完这一圈，谁接下一块。

具体：40 组、200 个 GEMM block。静态 stride → group 0 走 0、40、80…；group 1 走 1、41、81…。MoE 的 block 计算强度本来就不齐；随机访问让有的组早收工、干坐。

| 之前 | 之后 |
| --- | --- |
| ![thread load1](../../../../assets/vllm/blog/architecture/intel-arc/04-thread-load1.png) | ![thread load2](../../../../assets/vllm/blog/architecture/intel-arc/05-thread-load2.png) |

**Figure。** 原子抢活前后的 thread load（学习对照）。

改法：各组拿 **原子计数器** 抢下一份活。做完一个 GEMM block → 从原子里拿一个 rank → 那个 rank 就是下一块。小间隙没了；他们说各种专家路由都能排匀。

### 3. 快的 MXFP4 → BF16，再预打包

预打包提高加载效率。4-bit 加载，硬件友好 layout 在他们这边最多约 **+30%**。朴素 FP4→BF16 指令太多。替代（从 oneDNN 借：把 E2M1 编码跨到单精度 E/M 位置，再乘两型之间的 scale 差）：

`Bitcast-bf16 ((x << 12) >> 6 & 0x81c0) * 2^126`

转换指令压到最少。

## 数字（页上的演示）

DeepSeek 蒸馏 **8B–70B**，FP8，八张 Arc Pro——输出 token 吞吐在 Figure 1。

![perf figure1](../../../../assets/vllm/blog/architecture/intel-arc/06-perf-figure1.png)

**Figure 1。** SLA 下最大并发的 FP8 输出 token 吞吐，**8× Arc Pro B60**（学习对照）。

下一 token 延迟在负载下压在 **100 ms** 内（Figure 2，Qwen-32B，**4× B60**，prompt 数往上加）。

![perf figure2](../../../../assets/vllm/blog/architecture/intel-arc/07-perf-figure2.png)

**Figure 2。** Qwen-32B 下一 token 延迟 vs prompt 数，**4× Arc Pro B60**（学习对照）。

Llama-70B，单 batch，输入 **1K–40K**：TTFT / TPOT 还能齐。他们归功于沿序列维并行的 Flash Attention kernel。

![perf figure3](../../../../assets/vllm/blog/architecture/intel-arc/08-perf-figure3.png)

**Figure 3。** Llama-70B 单 batch 的 TTFT / TPOT，1K–40K 输入，**8× Arc Pro B60**（学习对照）。

GPT-OSS MXFP4，x8 Arc Pro B-series 机器（1–4 GPU）：

| Model | Data type | TP | Input/output seq | Concurrency | TTFT (s) | TPOT (ms) | Output tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-OSS-20b | MXFP4 | 1 | 1024/1024 | 75 | **7.614** | **53.96** | **1210.74** |
| GPT-OSS-20b | MXFP4 | 1 | 2048/2048 | 38 | 7.823 | 42.35 | 818.92 |
| GPT-OSS-20b | MXFP4 | 1 | 5120/5120 | 15 | 8.36 | 34.27 | 416.94 |
| GPT-OSS-120b | MXFP4 | 4 | 1024/1024 | 100 | 8.04 | 58.78 | **1495.12** |
| GPT-OSS-120b | MXFP4 | 4 | 2048/2048 | 50 | 8.11 | 41.98 | 1085.58 |
| GPT-OSS-120b | MXFP4 | 4 | 5120/5120 | 20 | 8.60 | 30.60 | 619.10 |

**Table 1。** GPT-OSS 在 1–4 GPU、x8 Arc Pro B-series 上的 vLLM 推理（页上的数）。

MLPerf Inference **v5.1**：Llama 8B 上 B60 有性价比条目，serving 框架是 vLLM。链接：[MLCommons inference datacenter](https://mlcommons.org/benchmarks/inference-datacenter/)。

## 怎么起

镜像：[Docker Hub 上的 intel/vllm](https://hub.docker.com/r/intel/vllm)。MoE / gpt-oss 从 **vllm 0.10.2** docker 起。例子假定宿主 **Ubuntu 25.04**、KMD **6.14.0**、Xeon 上 PCIe 插 **4× Arc Pro B60**。

```bash
docker pull intel/vllm:0.10.2-xpu
```

```bash
docker run -t -d --shm-size 10g --net=host --ipc=host --privileged \
  -v /dev/dri/by-path:/dev/dri/by-path --name=vllm-test \
  --device /dev/dri:/dev/dri --entrypoint= intel/vllm:0.10.2-xpu /bin/bash
```

gpt-oss-120b 四张 B60：

```bash
vllm serve openai/gpt-oss-120b --dtype=bfloat16 --enforce-eager \
  --port 8000 --host 0.0.0.0 --trust-remote-code \
  --gpu-memory-util=0.9 --no-enable-prefix-caching \
  --max-num-batched-tokens=8192 --disable-log-requests \
  --max-model-len=16384 --block-size 64 -tp 4
```

另开一只 shell 做 bench：

```bash
vllm bench serve --model openai/gpt-oss-120b \
  --dataset-name sonnet --dataset-path="./benchmarks/sonnet.txt" \
  --sonnet-input-len=1024 --sonnet-output-len=1024 --ignore-eos \
  --num-prompt 1 --trust_remote_code --request-rate inf \
  --backend vllm --port=8000 --host 0.0.0.0
```

对过的模型表：[Supported Models](https://github.com/intel/ai-containers/blob/main/vllm/0.10.2-xpu.md#supported-models)。

## 往后

跟核心 vLLM 接得更深。页上的路线：上游功能全覆盖；Intel 硬件上热门 LLM 的 SOTA 优化；改动送回上游。

## 致谢

vLLM 团队——点名的合作。

## 他们印的免责

成绩随用法、配置和其他因素变：[Intel Performance Index](http://www.intel.com/PerformanceIndex)。数字是所示日期的测试，未必含后来的更新。详见 [MLCommons](https://mlcommons.org/)。没有产品绝对安全。Intel 技术可能要开硬件、软件或服务。
