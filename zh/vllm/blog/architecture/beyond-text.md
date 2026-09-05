---
source: https://vllm.ai/blog/2025-09-05-beyond-text-generation
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 文本之外：pooling 模型把图吐回来

英文对照：[en/vllm/blog/architecture/beyond-text.md](../../../../en/vllm/blog/architecture/beyond-text.md)  
原文：https://vllm.ai/blog/2025-09-05-beyond-text-generation  
2025-09-05。署名 **Christian Pinto、Michele Gazzetti、Michael Johnston**（IBM Research Europe — Dublin），**Maximilien Philippe Marie de Bayser**（IBM Research — Brazil）。后来的多模态流水线见 [vllm-omni](../serving/vllm-omni.md)；插件面见 [plugin-system](plugin-system.md)；整族 backend 的亲戚是 [transformers-backend](transformers-backend.md)；pooling 里把 hidden 掏出来的另一条线：[extract-hidden-states](extract-hidden-states.md)。

适用：非自回归、一次前向吐出多模态输出——地理空间 / 视觉 / 结构化张量——当 pooling 模型 serve，输入输出要自己处理。不适合：把 `/pooling` 当成 chat completions；也不要指望 Transformers processor 会吃 GeoTIFF。

## 概览

生成式 serving 很久都绑在 **自回归文本** 上：一个 token 一个 token，自然语言。vLLM 先会文本进文本出。再来 MLLM（图、视频、音频当输入）和 LLaVA 式支持：多模态进、**文本出**。

下一页翻过来：**非自回归、一次推理就吐多模态输出**。引擎眼里它们像 [pooling 模型](https://docs.vllm.ai/en/latest/models/pooling_models.html)，但输入输出还要另管。原文点名的应用：图像分类与分割、音频合成、结构化数据。

落地是 **地理空间基础模型**——卷积或 ViT，要的不只是 RGB（多光谱 / 雷达），还要元数据（地理位置、拍摄日期）。灾害响应、卫星影像上的土地利用。改动是通用的；地理空间只是第一家。

具体：[TerraTorch](https://github.com/IBM/terratorch) 里所有地理空间模型（有的跟 NASA / ESA 合作）经 **generic backend** 进 vLLM，当一等公民。

## 地理空间模型当 pooling 接进去

不像文本模型，它们常常不需要把输出 token 解成字。一张输入图 → 一次前向 → 生输出 → 后处理成输出图。大图会切成补丁、打成 batch，再按元数据缝回去。

![models diff](../../../../assets/vllm/blog/architecture/beyond-text/01-models-diff.png)

**Figure。** 自回归文本 vs LLaVA 式 MLLM vs pooling 形态的地理空间 / 视觉模型（学习对照；版权仍归原站）。

vLLM 的 pooling 已经覆盖 embedding 和分类。**Identity pooler** 把 hidden 原样交出来——正是这条路。输入：把图预处理成张量，复用已有的多模态入口。TerraTorch 的 model-implementation backend，跟 HuggingFace Transformers backend 同一套路。

原文点名的引擎改动：

- 无 attention 的模型
- 不需要 tokenizer 的模型
- 原始输入张量，而不是默认 multimodal embedding
- serving API 加长

## IO Processor：tensor↔tensor 只走了一半

上面这些，只能 **tensor 进 tensor 出**。用户还得在 vLLM 外把图预处理成张量，再在外面把生张量后处理。没有一个端点是「图进去、图出来」。

此前，多模态输入的预处理只走 Transformers processor——认标准类型，不认 GeoTIFF（像素加地理元数据）。输出处理是解成文本，或对 hidden 做 pooler。没有第三条。

**IO Processor** 插件：在 **同一份 serving 实例** 里自定义前处理、后处理。字符串、JSON、图像张量、自定义结构——插件翻成客户端要的样子再交出去。

![io plugins flow](../../../../assets/vllm/blog/architecture/beyond-text/02-io-plugins-flow.png)

**Figure。** 请求 → IO Processor 前处理 → 引擎 `/pooling` → 后处理 → 客户端（学习对照）。

宣称：非文本模型（出图、图→分割掩膜、表→分类）可以共用 vLLM serving 栈。当时每个实例一只插件；只挂在 `/pooling`。别的端点是后话。

### 怎么用

插件实现预定义的 [IO Processor 接口](https://github.com/vllm-project/vllm/blob/main/vllm/plugins/io_processors/interface.py)，活在 vLLM 源码树 **外面**。安装时往 `vllm.io_processor_plugins` 注册 entry point。引擎初始化时发现、加载。

跟 vLLM 装在同一个 Python 环境，再加：

```bash
--io-processor-plugin <plugin_name>
```

当时的规矩：每个 vLLM 实例 **一只** IO Processor 插件。`/pooling` 上自动做前 / 后处理。

## 逐步：Prithvi 洪水检测

模型类例子：[Prithvi for flood detection](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11)。完整插件：[christian-pinto/prithvi_io_processor_plugin](https://github.com/christian-pinto/prithvi_io_processor_plugin)。

### 插件伪代码

把数据侧变换跟模型张量拆开——理想上任何模型、任何输入输出；以后还可以同一份输出挂多只插件。

```python
def pre_process(request_data: dict):
    # Downloads geotiff
    # In this example the input image has 7 bands
    image_url = request_data["url"]
    image_obj = download_image(image_url)

    # Extract image data:
    # - pixel_values([n, 6, 512, 512])
    #   - 6 input bands R, G, B, +3 multispectral wavelengths
    #   - n > 1 if the size of the input image is > [512, 512]
    # - metadata
    #   - GPS coordinates
    #   - date
    pixel_values, metadata = process_image(image_obj)

    # Process the image data into n vLLM prompts
    model_prompts = pixels_to_prompts(pixel_values)

    return model_prompts


def post_process(model_outputs: list[PoolingRequestOutput]):
    # Uses the previously extracted metadata to guarantee the output
    # contains the same georeferences and date.
    return image_object(model_outputs, metadata)
```

### 安装

`terratorch` **>=1.1rc3** 和 `vllm`。发文时所需改动 **还没进发版**（当时最新 **v0.10.1.1**）——装 [最新主干](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#install-the-latest-code_1)。

```bash
git clone git@github.com:christian-pinto/prithvi_io_processor_plugin.git
cd prithvi_io_processor_plugin
pip install .
```

装上的是 `prithvi_to_tiff` 插件。

### Serve

```bash
vllm serve \
    --model=ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
    --model-impl terratorch \
    --task embed --trust-remote-code \
    --skip-tokenizer-init --enforce-eager \
    --io-processor-plugin prithvi_to_tiff
```

原文日志：API server 听 `http://0.0.0.0:8000`。

### 请求 `/pooling`

JSON：`model` 和 `softmax` 是定好的；`data` 由插件定义。**`softmax: false`** 才能让插件拿到 **生** 输出。这个例子用 URL 送图，要 GeoTIFF 的 `b64_json`；脚本写成 `online_prediction.tiff`。

```python
import base64
import os
import requests

def main():
  image_url = "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff"
  server_endpoint = "http://localhost:8000/pooling"

  request_payload = {
      "data": {
          "data": image_url,
          "data_format": "url",
          "image_format": "tiff",
          "out_data_format": "b64_json",
      },
      "model": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
      "softmax": False,
  }

  ret = requests.post(server_endpoint, json=request_payload)

  if ret.status_code == 200:
    response = ret.json()
    decoded_image = base64.b64decode(response["data"]["data"])
    out_path = os.path.join(os.getcwd(), "online_prediction.tiff")
    with open(out_path, "wb") as f:
        f.write(decoded_image)
  else:
    print(f"Response status_code: {ret.status_code}")
    print(f"Response reason:{ret.reason}")


if __name__ == "__main__":
    main()
```

输入（左）：西班牙 Valencia **2024** 洪水时的卫星图。输出（右）：模型判为淹没的区域（白）。

![prithvi prediction](../../../../assets/vllm/blog/architecture/beyond-text/03-prithvi-prediction.png)

**Figure。** Prithvi 在 Valencia 2024 上的洪水掩膜（学习对照）。

## 下一步 / 文档

把 IO Processor 扩到更多 TerraTorch 模型和模态，安装更省事。更远：视觉语言模型、结构化推理 agent、多模态流水线，同一套栈。欢迎社区用 IO Processor 把边界再往外推。

上手：[IO Processor 插件文档](https://docs.vllm.ai/en/latest/design/io_processor_plugins.html)、[examples](https://github.com/vllm-project/vllm/tree/main/examples)、[TerraTorch](https://github.com/IBM/terratorch)。

## 致谢

vLLM 社区；尤其 **[Cyrus Leung](https://github.com/DarkLight1337)** 帮着把「文本之外」这个概念成形。IBM TerraTorch：**[Paolo Fraccaro](https://github.com/paolo-fraccaro)**、**[Joao Lucas de Sousa Almeida](https://github.com/Joao-L-S-Almeida)**，generic TerraTorch backend。
