---
source: https://vllm.ai/blog/2026-06-02-vllm-omni-autoround
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Omni × AutoRound：W4A16 一次量化、直接 serve

英文对照：[en/vllm/blog/serving/omni-autoround.md](../../../../en/vllm/blog/serving/omni-autoround.md)  
原文：https://vllm.ai/blog/2026-06-02-vllm-omni-autoround  
2026-06-02。署名 **vLLM-Omni Community, Intel AutoRound Team**。[AutoRound](https://github.com/intel/auto-round) 的 PTQ 接到 [vLLM-Omni](https://github.com/vllm-project/vllm-omni)：量化离线，热路径只推理。W4A16 是 4-bit 权重、16-bit 激活。LLM Compressor 那条（compressed-tensors 进 vLLM）见 [autoround-llmc.md](../architecture/autoround-llmc.md)。同一条 Omni 线：[vllm-omni.md](vllm-omni.md)、[qwen3-omni.md](qwen3-omni.md)。页上的 OmniBench / TIIF / B60 是他们的实验合同，不是你的 SLA。

读 `quantization_config.quant_method = "auto-round"`，serve 时**不必**再加 `--quantization`。Qwen3-Omni-30B-A3B：**66 GB → 25 GB**（约 **62%**）。evalscope 上 100 道图+音频题，W4A16 的 OmniBench 总分略**高于** BF16；TIIF 九个结构轴平均漂约 **1.3%**。Intel B60：FLUX.1-dev BF16 transformer **23 GB**，单卡 **24.4 GB** 装不下，要 TP4；W4A16 **7 GB** 单卡装得下（约 **19%** 余量）。腾出的卡做 CFG Parallel，引导生成约 **1.55–1.67×**。Wan2.2 / GLM-Image / FLUX 已通；BAGEL / Ovis 当时 checkpoint 有、runtime 还在接。Intel XPU 与 NVIDIA GPU 都验过。

图仍在原页（本地没有拷贝）。图注：

**Figure 1.** OmniBench：Qwen3-Omni-30B-A3B-Instruct 的 BF16 vs W4A16 AutoRound。

**Figure 2.** TIIF-Bench：多阶段 T2I 的九个结构子属性。

**Figure 3.** Wan2.2 T2V-A14B，W4A16 AutoRound 的 text-to-video。

**Figure 4.** Wan2.2 I2V-A14B，W4A16 AutoRound 的 image-to-video。

**Figure 5.** 各 Omni 家族 VRAM：BF16 vs W4A16 AutoRound。

**Figure 6.** 延迟/显存交易：W4A16 把 FLUX 最低硬件从 4 卡降到 1 卡，才能开 CFG Parallel。

**Figure 7.** Intel XPU B60 上 CFG Parallel：相对顺序 BF16 **1.55–1.67×**。

正文表格**没有**把 OmniBench / TIIF / VRAM 的格子数字抄下来；那些在图里。

## 1. 引言：vLLM-Omni 遇见 AutoRound

Omni 要伺候扩散、多模态 Omni、多阶段生成。这里量化不是「把一只 transformer 捏瘦」——是让一整柜**异构 runtime** 能进真实的卡。

AutoRound（Intel；EMNLP 2024，signed GD 做 weight rounding）是 tuning 式 PTQ。每个被量化张量三个可训量：rounding 偏移 `V`，clip 的 `alpha` / `beta`。低 bit 比 naive round-to-nearest 稳；checkpoint 是静的——推理路径**零**额外量化开销。页上三层：算法（AutoRound）+ runtime（Omni）+ Hugging Face 上的 INT4 目录。

运行时只读 checkpoint。看见 `quantization_config.quant_method = "auto-round"`，把块 remap 到 runtime 模块，再选计算后端。serve API 和普通 load 一样。

## 2. 模型覆盖

原文三条范式。

### 2.1 Omni 多模态

文本 / 视觉 / 音频合一；跨模态 embedding 对齐是量化的暗礁。

| Model | Checkpoint | Status |
|---|---|---|
| Qwen3-Omni-30B-A3B-Instruct | [Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound](https://huggingface.co/Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound) | 已接入并验证 |
| Qwen2.5-Omni-7B | [Intel/Qwen2.5-Omni-7B-int4-AutoRound](https://huggingface.co/Intel/Qwen2.5-Omni-7B-int4-AutoRound) | 已接入并验证 |

### 2.2 扩散与多阶段图像

| Model | Checkpoint | Status |
|---|---|---|
| GLM-Image | [Intel/GLM-Image-int4-AutoRound](https://huggingface.co/Intel/GLM-Image-int4-AutoRound) | 已接入并验证 |
| FLUX.1-dev | [vllm-project-org/FLUX.1-dev-AutoRound-w4a16](https://huggingface.co/vllm-project-org/FLUX.1-dev-AutoRound-w4a16) | 已接入并验证 |
| BAGEL-7B-MoT | [Intel/BAGEL-7B-MoT-int4-AutoRound](https://huggingface.co/Intel/BAGEL-7B-MoT-int4-AutoRound) | checkpoint 有；runtime 当时还在接 |
| Ovis-Image-7B | [Intel/Ovis-Image-7B-int4-AutoRound](https://huggingface.co/Intel/Ovis-Image-7B-int4-AutoRound) | checkpoint 有；runtime 当时还在接 |

### 2.3 视频扩散（Wan2.2）

时空视频；INT4 checkpoint 在 Omni 里验过：

- [Intel/Wan2.2-I2V-A14B-Diffusers-int4-AutoRound](https://huggingface.co/Intel/Wan2.2-I2V-A14B-Diffusers-int4-AutoRound)
- [Intel/Wan2.2-T2V-A14B-Diffusers-int4-AutoRound](https://huggingface.co/Intel/Wan2.2-T2V-A14B-Diffusers-int4-AutoRound)
- [Intel/Wan2.2-TI2V-5B-Diffusers-int4-AutoRound](https://huggingface.co/Intel/Wan2.2-TI2V-5B-Diffusers-int4-AutoRound)

## 3. 用法

量化和 tuning **离线**。生产代码只推理。serving 路径上没有校准。

### 3.1 用量化模型推理

FLUX.1-dev 的 Python API 就是普通 Omni load——变的只是 checkpoint 路径：

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

if __name__ == '__main__':
    omni = Omni(model="vllm-project-org/FLUX.1-dev-AutoRound-w4a16")
    outputs = omni.generate(
        "A cat sitting on a windowsill",
        OmniDiffusionSamplingParams(num_inference_steps=28, guidance_scale=3.5),
    )
    outputs[0].images[0].save("output.png")
```

Wan2.2：标准 `vllm serve`，视频 endpoint 和 BF16 同一条。

```bash
vllm serve Intel/Wan2.2-T2V-A14B-Diffusers-int4-AutoRound --omni --port 8091
```

```bash
curl -X POST "http://127.0.0.1:8091/v1/videos/sync" \
  -F 'prompt=Cherry blossoms swaying gently in the breeze, cinematic motion' \
  -F 'width=832' -F 'height=480' -F 'num_frames=48' \
  -F 'num_inference_steps=40' -F 'guidance_scale=5.0' \
  --output t2v_output.mp4
```

Qwen2.5-Omni：OpenAI 兼容 chat，也不改。

```bash
vllm serve Intel/Qwen2.5-Omni-7B-int4-AutoRound --omni --port 8091
```

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Intel/Qwen2.5-Omni-7B-int4-AutoRound",
    "messages": [{"role": "user", "content": "What is 2 + 3?"}],
    "max_tokens": 128
  }'
```

Omni 自己读量化 metadata。已经量化好的 AutoRound：**不要**再加 `--quantization`。

### 3.2 量化一只新模型

离线 AutoRound，再 serve。页上三道菜**不是**同一道：FLUX 用 `--iters 0` 加 `--disable_opt_rtn`（没有 signed-GD tuning 圈）；Wan 是 `--iters 100` / `--nsamples 32`；Qwen3-Omni 是 `--bits 4 --group_size 128 --iters 200 --lr 5e-3`。

```bash
# FLUX.1-dev
auto-round \
  --model black-forest-labs/FLUX.1-dev \
  --scheme W4A16 \
  --batch_size 1 \
  --disable_opt_rtn \
  --dataset coco2014 \
  --iters 0

# Wan2.2-T2V-A14B
auto-round \
  --model_name Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --format auto_round \
  --scheme W4A16 \
  --iters 100 \
  --nsamples 32 \
  --batch_size 1 \
  --num-inference-steps 3 \
  --guidance-scale 5.0 \
  --dataset coco2014 \
  --output_dir Wan2.2-T2V-A14B-Diffusers-int4-AutoRound

# Qwen3-Omni-30B-A3B-Instruct
auto-round \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --bits 4 \
  --group_size 128 \
  --format auto_round \
  --iters 200 \
  --lr 5e-3 \
  --output_dir tmp_qwen3_omni_w4a16 \
  --trust_remote_code
```

`config.json` 里的 metadata：

```json
{
  "quantization_config": {
    "quant_method": "auto-round",
    "bits": 4,
    "group_size": 128,
    "sym": true,
    "packing_format": "auto_round:auto_gptq"
  }
}
```

页上的经验：**128** 条校准、约 **200** 轮优化常常够；更大、更敏感的模型可能要加。具体看家族、任务、部署约束。那句「常常够」**不是** FLUX 那段用的配方（`iters 0`）。

### 3.3 质量校验

扩散：同一 seed，对照 BF16。

```bash
python -m vllm_omni.quantization.tools.compare_diffusion_trajectory_similarity \
  --task t2i \
  --reference-model black-forest-labs/FLUX.1-dev \
  --candidate-model vllm-project-org/FLUX.1-dev-AutoRound-w4a16 \
  --prompt "a cup of coffee on the table" \
  --height 512 --width 512 \
  --num-inference-steps 20 \
  --seed 142 \
  --output-json /tmp/flux_similarity/result.json
```

原文没写这个工具的 JSON 字段，也没给 pass/fail 阈值。

## 4. 定量：精度与质量

### 4.1 Omni 多模态（OmniBench）

evalscope，**100** 道多模态题，图**和**音频同时进。W4A16 的 OmniBench 总分略**高于** BF16。正文没有 OmniBench 总分数字。**Figure 1。**

### 4.2 多阶段扩散（TIIF-Bench）

九个结构子属性：对齐、构图、保真。平均精度掉约 **1.3%**。九个轴的名字正文没列全。**Figure 2。**

### 4.3 视频（Wan2.2）

naive 标量量化容易把时间一致性撕开。T2V-A14B（**Figure 3**）和 I2V-A14B（**Figure 4**）用客观指标。W4A16 AutoRound 下，T2V-A14B 的结构一致性还略**升**——页上的假说：clip 优化有时像正则。指标名和 delta 在图里，不在表里。

## 5. 性能、脚印、serving

### 5.1 VRAM 脚印

一阶收益：checkpoint 体积和运行时内存。W4A16 把量化权重从 BF16 压到大约原来的 **¼**。端到端加速还看工作负载先前卡在容量还是带宽。**Figure 5。**

不是每一段都被量化。VAE decode、附属阶段、多阶段系统的一部分可以留在更高精度——权重压缩比通常**大于**端到端延迟加速。不要把 Omni 的 66→25 GB 读成「每一段的每一个字节都是 INT4」。

### 5.2 用内存余量换延迟

这节 case study 全在 **Intel XPU B60**。不是一张 NVIDIA GPU 的 CFG Parallel 表。

**最低硬件：4 卡 → 1 卡。** BF16 FLUX.1-dev transformer **23 GB**，加上激活就塞不进单张 B60（**24.4 GB**）——要 TP=4。W4A16 transformer **7 GB**，单卡装得下，约 **19%** 余量。

**W4A16 + CFG Parallel = 引导生成 1.55×–1.67×。** Classifier-Free Guidance 每步两趟去噪（prompt + negative）。BF16 四卡都给 tensor parallelism 占满，两趟只能**串行**（延迟 2×）。W4A16 用 TP=2 就够，腾出两卡，两条 guidance 分支在两组 GPU 上同时跑。**Figure 6**（硬件下降 + CFG Parallel）。**Figure 7**（B60 上 CFG Parallel 延迟）。**1.55–1.67×** 是对照那套 B60 布局上的顺序 BF16，不是对照一张塞不进去的单卡 BF16。

主张不只是「装得下」：内存余量让扩散**换一种跑法**，并行策略带来的加速可以大于单纯 dequant 省下来的那一点。

## 6. 结语

对着运维要的东西：离线 checkpoint、自动识别、可预期的内存、上线前能验质量。点名覆盖：FLUX、Wan、GLM、BAGEL、Ovis、Qwen Omni。BAGEL 和 Ovis 当时只有 checkpoint，runtime 还没接完。

还在做：Linear 和 MoE 的 **MXFP4** / **MXFP8**；attention 低 bit（比如 SageAttention）。

## 7. 致谢

vLLM-Omni 的 Hongsheng Liu、Shunyang Li、WeiQing Chen；Intel 的 Chendi Xue。以及很快把 AutoRound 用起来的 Omni 社区。
