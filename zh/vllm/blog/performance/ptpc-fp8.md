---
source: https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# PTPC-FP8：ROCm 上更贴 BF16 的 FP8

英文对照：[en/vllm/blog/performance/ptpc-fp8.md](../../../../en/vllm/blog/performance/ptpc-fp8.md)  
原文：https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm  
2025-02-24。署名 **AMD and Embedded LLM**。PR：[vllm#12501](https://github.com/vllm-project/vllm/pull/12501)。需要 **vLLM ≥ 0.7.3**。数字是 **MI300X** 上、commit `4ea48fb35cf67d61a1c3f18e3981c362e1d8e26f` 的演示，不是你卡上的 SLA。

这是 **权重 + 激活** 的 FP8，现场从 Hugging Face 量化，不必预量化。和 [FP8 KV](fp8-kvcache.md) 分清：那篇是 KV dtype 和 attention 计算；这篇是 GEMM 上的权重量化。ROCm 上后来的 attention 编排：[rocm-attention](../architecture/rocm-attention.md)。KV 再往 3–4 bit 压：[turboquant](turboquant.md)。

**原文 TL;DR**

- **新东西：** `--quantization ptpc_fp8`（v0.7.3+，AMD ROCm）。
- **好在哪：** 速度接近其他 FP8，精度更贴 BF16。原文把它写成当时 ROCm 上最好的 FP8 选项。
- **怎么用：** 装 ROCm；vLLM ≥ 0.7.3；跑 Hugging Face 模型时加 `--quantization ptpc_fp8`。不必预量化。

本地图（原文版权仍归原站；学习对照用）：

![What is PTPC-FP8](../../../../assets/vllm/blog/performance/ptpc-fp8/01-PTPC121.png)

**PTPC-FP8** = Per-Token-Activation, Per-Channel-Weight FP8。激活按 token 缩放，权重按 channel 缩放，比传统 per-tensor FP8 更准。

## 引言：量化的麻烦，和这把尺子

LLM 算得贵。FP8 减显存、加速矩阵乘，但传统量化撞上 outlier。PTPC 的说法：近 BF16 的精度，FP8 的速度，直接吃 Hugging Face 权重。

### Outlier

模型大到某个尺度，激活会出现异常大的值：

- per-tensor 量化时，多数值只分到很少有效 bit
- outlier **钉在同一批通道**上，跨 token 也还在
- 权重相对均匀、好量化；激活不是

### PTPC：按观察对粒度

三句话撑起双粒度：

1. outlier 总出现在同一批通道
2. 同一个 token 里，通道幅度差得很开
3. **同一通道**跨 token 的幅度相对稳

于是：

- **Per-token 激活：** 每个输入 token 自己的 scale
- **Per-channel 权重：** 每一列权重自己的 scale

![PTPC Diagram](../../../../assets/vllm/blog/performance/ptpc-fp8/02-PTPC-Diagram.png)

图里两套量化。张量：

- $X$：输入激活（$T \times C_i$）
- $W$：权重（$C_i \times C_o$）
- $T$：token 序列长；$C_i / C_o$：入 / 出通道；$*$：矩阵乘

缩放：

- **上（per-tensor）：** 整张量各一个标量 $\Delta_X[1]$、$\Delta_W[1]$
- **下（PTPC）：** 向量 $\Delta_X[T \times 1]$（每 token 一个）、$\Delta_W[1 \times C_o]$（每输入通道一个）

粒度细了，才有「贴着 BF16、还是 8 bit」的位置。

## 融合 kernel：别让细粒度变成两步 HBM

细粒度 scale 若不融合，会先慢下来。ROCm 的答案：**fused FP8 rowwise scaled GEMM**。

### 两步 vs 融合

朴素路径要两步：

```python
# Naive 2-step approach:
output = torch._scaled_mm(input, weight)       # Step 1: FP8 GEMM
output = output * token_scales * channel_scales  # Step 2: Apply scaling factors
```

中间结果写出再读回，带宽和周期都浪费。

融合把乘和缩放收成一次硬件操作：

```python
# Optimized fused operation:
output = torch._scaled_mm(input, weight,
                         scale_a=token_scales,
                         scale_b=channel_scales)
```

![Fused GEMM](../../../../assets/vllm/blog/performance/ptpc-fp8/03-FusedGEMM.svg)

MI300X 上有原生 FP8。原文写的好处：缩放在片上做完再写回；少一次多余算；相对两步实现最多约 **2.5×**。没有这只融合核，PTPC 的精度优势会先被内存税吃掉。

## MI300X 上的速度和精度

vLLM on AMD MI300X，commit `4ea48fb35cf67d61a1c3f18e3981c362e1d8e26f`。

### 吞吐：PTPC vs per-tensor FP8

- 模型：Llama-3.1-70B-Instruct
- 数据：SharedGPT
- GPU：**1× MI300X**
- 结果：吞吐和 per-tensor FP8 **几乎一样**（还略好，约 **1.01×**）。融合核把更复杂的 scaling 开销抹平了。

![Throughput reqs/s](../../../../assets/vllm/blog/performance/ptpc-fp8/04-PTPCReqs.svg)

![Speedup vs per-tensor FP8](../../../../assets/vllm/blog/performance/ptpc-fp8/05-PTPCSpeedup.svg)

### 精度：Wikitext perplexity（越低越好）

- 模型：Llama-3.1-8B-Instruct
- 数据：Wikitext
- 设置：**2× MI300X**，tensor parallelism

Perplexity 是「模型对下一词有多困惑」。低 = 更有把握；高 = 更常被吓到。原文提醒：即使只高 **0.1**，对已经压得很干净的大模型也可能是实质退化。

![bits and byte perplexity](../../../../assets/vllm/blog/performance/ptpc-fp8/06-PerplexityBits.png)

![Word perplexity](../../../../assets/vllm/blog/performance/ptpc-fp8/07-Perplexitywords.png)

| Precision | Word Perplexity | % Degradation |
| --- | ---: | ---: |
| BF16（基线） | 9.4281 | — |
| PTPC-FP8 | 9.5093 | 0.86% |
| Standard FP8 | 9.5124 | 0.89% |

PTPC 略优于 standard FP8（9.5093 vs 9.5124）；相对 BF16 只高 **0.86%**。bits_per_byte / byte_perplexity 同一方向。小的质量缺口会在推理和生成里叠起来——这是他们把 perplexity 写这么细的理由。

### 精度：GSM8K（数学推理）

GSM8K：小学应用题。多步推理、数字要准、逻辑要自洽。量化一伤能力，往往先伤这里。

两种算法：

- **Flexible-extract：** 正确答案的数字在回复里出现就算
- **Strict-match：** 必须是期望格式里的精确答案

![GSM8K 8B](../../../../assets/vllm/blog/performance/ptpc-fp8/08-GSM8K8B.png)

**8B，strict-match：**

| Method | Strict-match | 相对 BF16 |
| --- | ---: | ---: |
| BF16 | 73.2% | 100% |
| PTPC-FP8 | 70.8% | 96.7% |
| Standard FP8 | 69.2% | 94.5% |

![GSM8K 70B](../../../../assets/vllm/blog/performance/ptpc-fp8/09-GSM8K70B.png)

**70B：** PTPC strict-match **87.3%**，略高于 BF16 的 **86.3%**。两者的 strict-match 都高于 standard FP8。70B 上 PTPC「赢」BF16——**当噪声，别当免费精度**。

原文自己的读法：推理能力还在；PTPC 在两个尺寸上都压过 standard FP8；近 BF16 的质量，8 bit 的显存和速度；模型越大，量化方法之间的缝越窄，大模型上 PTPC 更值得。

## 上手 CLI

1. 装一版还新的 ROCm。
2. 当时的路径：clone vLLM，用 `Dockerfile.rocm` 构建。

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
DOCKER_BUILDKIT=1 docker build -f Dockerfile.rocm -t vllm-rocm .
docker run -it \
   --network=host \
   --group-add=video \
   --ipc=host \
   --cap-add=SYS_PTRACE \
   --security-opt seccomp=unconfined \
   --device /dev/kfd \
   --device /dev/dri \
   -v <path/to/model>:/app/model \
   vllm-rocm \
   bash
```

3. 打开 `--quantization ptpc_fp8`。权重会**现场**量化。把 `<your-model>` 换成任意 Hugging Face 模型：

```bash
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve <your-model> --max-seq-len-to-capture 16384 --enable-chunked-prefill=False --num-scheduler-steps 15 --max-num-seqs 1024 --quantization ptpc_fp8
```

**原文示例里的坑：** 关着 chunked prefill（`--enable-chunked-prefill=False`），开着多 step scheduler（`--num-scheduler-steps 15`），还把 `VLLM_USE_TRITON_FLASH_ATTN=0`。旗标以你那版文档为准，不要把这一行冻成永远正确的默认。

## 收束

原文把 PTPC 写成准确率和速度之间的那块甜区：近 BF16 的精度，FP8 的速度，让更多人在 AMD 硬件上用得起大模型。邀请跑、反馈、给 vLLM 提 PR。页上的数字仍是那一次 MI300X 演示。

## 附录：lm-evaluation-harness

Wikitext（8B，`HIP_VISIBLE_DEVICES=0,1`，TP2，`max_model_len=2048`，`gpu_memory_utilization=0.6`，`--batch_size 16`）：

```bash
# Unquantized (Bfloat16)
MODEL=meta-llama/Llama-3.1-8B-Instruct
HIP_VISIBLE_DEVICES=0,1 lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True,tensor_parallel_size=2,kv_cache_dtype=auto,max_model_len=2048,gpu_memory_utilization=0.6 \
  --tasks wikitext --batch_size 16

# Per-Tensor FP8 Quantization
MODEL=meta-llama/Llama-3.1-8B-Instruct
HIP_VISIBLE_DEVICES=0,1 lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True,tensor_parallel_size=2,quantization=fp8,kv_cache_dtype=fp8_e4m3,max_model_len=2048,gpu_memory_utilization=0.6 \
  --tasks wikitext --batch_size 16

# Per-Token-Activation Per-Channel-Weight FP8 Quantization
MODEL=meta-llama/Llama-3.1-8B-Instruct
HIP_VISIBLE_DEVICES=0,1 lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True,tensor_parallel_size=2,quantization=ptpc_fp8,kv_cache_dtype=fp8_e4m3,max_model_len=2048,gpu_memory_utilization=0.6 \
  --tasks wikitext --batch_size 16
```

GSM8K（原文写 8B，注释说 70B 同样改路径；`--num_fewshot 5 --batch_size auto --limit 250`）。附录里的模型路径是容器内的 `/app/model/...`：

```bash
# FP8 (Per-Tensor)
MODEL=/app/model/Llama-3.1-8B-Instruct/  # Or Llama-3.1-70B-Instruct
lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True,quantization=fp8,kv_cache_dtype=fp8_e4m3 \
  --tasks gsm8k  --num_fewshot 5 --batch_size auto --limit 250

# PTPC FP8
MODEL=/app/model/Llama-3.1-8B-Instruct/  # Or Llama-3.1-70B-Instruct
lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True,quantization=ptpc_fp8,kv_cache_dtype=fp8_e4m3 \
  --tasks gsm8k  --num_fewshot 5 --batch_size auto --limit 250

# BF16
MODEL=/app/model/Llama-3.1-8B-Instruct/  # Or Llama-3.1-70B-Instruct
lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True,kv_cache_dtype=auto \
  --tasks gsm8k  --num_fewshot 5 --batch_size auto --limit 250
```

评测命令里 PTPC / 标准 FP8 还带了 `kv_cache_dtype=fp8_e4m3`。那是 **KV** 的 dtype，不是 PTPC 本身。对照 [fp8-kvcache](fp8-kvcache.md)。
