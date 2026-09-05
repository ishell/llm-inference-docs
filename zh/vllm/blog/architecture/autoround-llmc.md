---
source: https://vllm.ai/blog/2025-12-09-intel-autoround-llmc
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# AutoRound × LLM Compressor：W4A16 进 compressed-tensors，vLLM 直接 serve

英文对照：[en/vllm/blog/architecture/autoround-llmc.md](../../../../en/vllm/blog/architecture/autoround-llmc.md)  
原文：https://vllm.ai/blog/2025-12-09-intel-autoround-llmc  
2025-12-09。署名 **Intel Neural Compressor Team, Red Hat AI Model Optimization Team**。第一次把 `AutoRoundModifier` 送进 [LLM Compressor](https://github.com/vllm-project/llm-compressor) 的笔记。Omni 侧（一次量化、vLLM-Omni 里 serve W4A16）见 [omni-autoround](../serving/omni-autoround.md)；Arm 上 INT8 / W4A8 亲戚：[arm-cpus](arm-cpus.md)；Arc XPU serving：[intel-arc](intel-arc.md)。GSM8K **0.911** 是他们 `lm_eval` 演示（5-shot、`--limit 1000`）——页上写了会晃，不是 SLA。

适用：Llama / Qwen dense 做 W4A16 PTQ，存成 `compressed-tensors`，再 `vllm serve`。量化在工作站 GPU，serve 可以换机器（他们点名单张 **Intel Arc Pro B60**）。不适合：XPU 上不带 `--enforce-eager`；也不要指望这第一阶段就覆盖 FP8 / MXFP4 / NVFP4 / MoE——那些是路线图。

论文：[AutoRound (EMNLP 2024)](https://aclanthology.org/2024.findings-emnlp.662.pdf)。代码：[intel/auto-round](https://github.com/intel/auto-round)。

## 他们印的 TL;DR

AutoRound——Intel 那套 tuning-based PTQ——进了 LLM Compressor：

- 低 bit 精度更高
- 调参轻（几百步，不是上千）
- 推理路径零额外开销
- `compressed-tensors` checkpoint，[vLLM](https://github.com/vllm-project/vllm) 直接 serve
- 几行代码量化再 serve

更广的 scheme 和模型覆盖，页上写「接下来」。

## AutoRound 是什么

给 LLM 和 VLM 的 PTQ。**每个量化张量** 三个可训量：

- `V`——rounding 偏移
- `α` 和 `β`——学出来的 clip 范围

Decoder 层 **按顺序** 走。Signed gradient descent 一起调 rounding 和 clipping，最小化 **块级输出重建误差**。

他们点名的长处：

- 极低 bit 时精度仍像样
- 多种 dtype：W4A16、MXFP8、MXFP4、FP8、NVFP4，还会再加
- Mixed-bit、按层搜精度
- LLM **和** VLM

页上点名的硬件：Intel Xeon、Intel Gaudi、Intel Data Center GPU、Intel Arc B-Series，以及其他 GPU（点了 CUDA）。往后：下一代 Data Center GPU（代号 **Crescent Island**）原生 FP8 / MXFP8 / MXFP4——AutoRound 的 checkpoint 就沿这条路长。

## 为什么塞进 LLM Compressor

LLM Compressor 已经把量化、剪枝这类压缩原语收成一套。AutoRound 进去：

- 对齐现有 modifier 架构（比如 `GPTQModifier`）
- 复用顺序校准和 layer-onloading
- 以后能跟别的 modifier 拼菜谱
- Checkpoint 直接给 vLLM——压缩到部署一条流水线

## 集成（第一阶段）

LLM Compressor 里新的 `AutoRoundModifier`，产出 `W{n}A16`（演示是 **W4A16**），vLLM 能加载。第一阶段 PR：[llm-compressor#1994](https://github.com/vllm-project/llm-compressor/pull/1994)。配置就是模型和校准数据。Dense LLM：**Llama** 和 **Qwen** 族。

## 上手（原文逐步）

### 1. 安装

```bash
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
```

### 2. 载入模型和 tokenizer

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_ID = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 3. 校准数据

```python
from auto_round.calib_dataset import get_dataset
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048
ds = get_dataset(tokenizer=tokenizer,
                 seqlen=MAX_SEQUENCE_LENGTH,
                 nsamples=NUM_CALIBRATION_SAMPLES)
```

### 4. 量化

CPU 或 GPU 都能跑。量化和 serving 不必同一台设备——现在工作站 GPU，以后 AIPC，他们的例子。

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier

recipe = AutoRoundModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
    iters=200,
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    shuffle_calibration_samples=False,
)

SAVE_DIR = MODEL_ID.split("/")[-1] + "-W4A16-G128-AutoRound"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

页上的经验：**128 条校准 + 约 200 iters** 常常就收敛。极低 bit 或更紧的精度目标，再加样本或步数。

### 5. 在 vLLM 里 serve

压缩后的 checkpoint 可以换硬件 serve，不必跟 tuning 同一台。例子：量化过的 `Qwen3-8B-W4A16-G128-AutoRound` 放一张 **Intel Arc Pro B60**：

```bash
vllm serve Qwen3-8B-W4A16-G128-AutoRound \
    --dtype=bfloat16 \
    --gpu-memory-utilization 0.8 \
    --max-num-batched-tokens 8192
```

**页上的坑：** vLLM 要从 [PR #29484](https://github.com/vllm-project/vllm/pull/29484/) 装。**XPU** 上必须 `--enforce-eager`。

### 6. 评测（`lm_eval` 跑 GSM8K）

```bash
lm_eval --model vllm \
  --model_args pretrained="./Qwen3-8B-W4A16-G128-AutoRound,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,enforce_eager=True" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 1000 \
  --batch_size 128
```

他们印的表：

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.911|±  | 0.009|
|     |       |strict-match    |     5|exact_match|↑  |0.911|±  | 0.009|
```

页上注明：非确定性，数字会晃。

## 收束和计划

第一阶段：W4A16 端到端，配置简单，Llama / Qwen dense。路线图：FP8、MXFP4、MXFP8、NVFP4；自动 mixed-bit 搜索；MoE 族；跟 LLM Compressor 里别的算法拼更丰富的 multi-modifier 菜谱。

想插队：[RFC #1968](https://github.com/vllm-project/llm-compressor/issues/1968)，或 Intel Community。

## 致谢

LLM Compressor 与 vLLM 社区。点名：Kyle Sayers、Dipika Sikka、Brian Dellabetta、Charles Hernandez、Robert Shaw、Kunshang Ji——早期提案和 PR 审阅。

### 相关 RFC / PR

[llm-compressor#1968](https://github.com/vllm-project/llm-compressor/issues/1968)、[llm-compressor#1994](https://github.com/vllm-project/llm-compressor/pull/1994)、[llm-compressor#2055](https://github.com/vllm-project/llm-compressor/pull/2055)、[llm-compressor#2062](https://github.com/vllm-project/llm-compressor/pull/2062)、[auto-round#993](https://github.com/intel/auto-round/pull/993)、[auto-round#1053](https://github.com/intel/auto-round/pull/1053)、[auto-round#1055](https://github.com/intel/auto-round/pull/1055)、[auto-round#1072](https://github.com/intel/auto-round/pull/1072)、[vllm#29484](https://github.com/vllm-project/vllm/pull/29484)。
