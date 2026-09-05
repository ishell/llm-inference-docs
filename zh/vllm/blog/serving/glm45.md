---
source: https://vllm.ai/blog/2025-08-19-glm45-vllm
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# GLM-4.5 / 4.5V：hybrid thinking，parser 叫 glm45，当时不要 V0

英文对照：[en/vllm/blog/serving/glm45.md](../../../../en/vllm/blog/serving/glm45.md)  
原文：https://vllm.ai/blog/2025-08-19-glm45-vllm  
2025-08-19。**Yuxuan Zhang**。生产后续：[glm52-b300.md](glm52-b300.md)。nightly 安装；**vLLM V0 不支持**。图在 GitHub（`bench.png`、`bench_45v.jpeg`）——这里不收。

[GLM](https://aclanthology.org/2022.acl-long.26/) 来自 Zhipu.ai（现 [Z.ai](https://z.ai/)）。合作很早，能追到 ChatGLM。这篇：[GLM-4.5](https://arxiv.org/abs/2508.06471) 与 [GLM-4.5V](https://arxiv.org/abs/2507.01006)，跑 NVIDIA Blackwell 与 Hopper。

| 模型 | 总参 / 激活 |
|---|---|
| GLM-4.5 | 355B / 32B |
| GLM-4.5-Air | 106B / 12B |

Hybrid reasoning：**thinking**（复杂推理 + 工具）对 **non-thinking**（立刻答）。他们 12 项基准：GLM-4.5 **63.2**（专有 + 开源里第 3）；Air **59.8**。4.5V 基于 Air；同规模 42 项公开 VL 基准他们自称 SOTA。仓库：[zai-org/GLM-4.5](https://github.com/zai-org/GLM-4.5)、[zai-org/GLM-V](https://github.com/zai-org/GLM-V)。

## 当时怎么装

最新 `main`。nightly vLLM + 一份 preview transformers：

```shell
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
pip install transformers-v4.55.0-GLM-4.5V-preview
```

## 怎么起

FP8 和 BF16 用 **同一条** `vllm serve`。

GLM-4.5 / Air：

```shell
vllm serve zai-org/GLM-4.5-Air \
    --tensor-parallel-size 4 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice
```

GLM-4.5V：

```shell
vllm serve zai-org/GLM-4.5V \
     --tensor-parallel-size 4   \
     --tool-call-parser glm45   \
     --reasoning-parser glm45   \
     --enable-auto-tool-choice  \
     --allowed-local-media-path / \
     --media-io-kwargs '{"video": {"num_frames": -1}}'
```

### 他们印死的注意事项

- 思考写在 `reasoning_content`；`content` 只有终答。关掉思考：`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
- 8× H100 跑满血 GLM-4.5 显存不够：`--cpu-offload-gb 16`
- `flash_infer` 不顺：临时 `VLLM_ATTENTION_BACKEND=XFORMERS`；或设 `TORCH_CUDA_ARCH_LIST`（例如 `'9.0+PTX'`）让 FlashInfer 跑——**arch 字符串按卡换**
- **vLLM V0 不支持这些模型**

### GLM-4.5V 的 grounding

提示里要框；模型先想再给框。例：

- Help me to locate `<expr>` in the image and give me its bounding boxes.
- Please pinpoint the bounding box `[[x1,y1,x2,y2], …]` in the image as per the given description. `<expr>`

框是 \([x_1,y_1,x_2,y_2]\) 左上 / 右下；x 按宽、y 按高归一，再 **×1000**。特殊 token `<|begin_of_box|>` / `<|end_of_box|>`。括号写法可以变（`[]`、`[[]]`、`()`、`<>`）；意思一样。

## 合作 / 致谢

发布前 vLLM 就和 GLM 组对着干，好让 `main` 当天能起。名字：Kaichao You、Simon Mo、Zifeng Mo、Lucia Fang、Rui Qiao、Jie Li、Ce Gao、Roger Wang、Lu Fang、Wentao Ye、Zixi Qi。
