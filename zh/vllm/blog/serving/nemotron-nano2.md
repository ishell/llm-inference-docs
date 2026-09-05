---
source: https://vllm.ai/blog/2025-10-23-now_serving_nvidia_nemotron_with_vllm
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Nemotron Nano 2：9B hybrid，Thinking Budget 是客户端两刀，不是引擎旋钮

英文对照：[en/vllm/blog/serving/nemotron-nano2.md](../../../../en/vllm/blog/serving/nemotron-nano2.md)  
原文：https://vllm.ai/blog/2025-10-23-now_serving_nvidia_nemotron_with_vllm  
2025-10-23。署名 **NVIDIA Nemotron Team**。这系列里第一篇 Nemotron 上 vLLM 的 day-0。后继 30B：[Nano](nemotron-3-nano.md)。同一家：[Super](nemotron-3-super.md)、[Ultra](nemotron-3-ultra.md)、[Lightning](nemotron-35-lightning.md)。多模态：[Nano 2 VL](nemotron-nano-vl.md)、[Nano Omni](nemotron-omni.md)。Mamba 拆分：[hybrid-ssm.md](hybrid-ssm.md)。他们报 thinking token 相对同尺寸 dense 最高约 **6×**——页上的演示，不是你的 SLA。

**原文 TL;DR：**

- [NVIDIA Nemotron Nano 2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)：9B hybrid Transformer–Mamba；thinking budget 可配。
- 开源权重，Hugging Face 上还有超过 **9T** token 的 pre- / post-training 数据。
- 起服：`--trust-remote-code --mamba_ssm_cache_dtype float32`。
- Budget 是**应用层协议**：先 `max_tokens=budget` 拿 reasoning；若无 `</think>` 就人工补上；再 `/completions` 加 `continue_final_message`。**不是** `--thinking-budget`。

## 为什么要这只

Agentic 系统要的工具：开、快、能放大。[NVIDIA Nemotron](https://developer.nvidia.com/nemotron) 被写成开源模、数据、技术的一家，给专门的 agentic AI。

vLLM 是把这家人送到机房和边缘的路。开箱：开源权重、开源数据。

## NVIDIA Nemotron Nano 2

当时最新：小号推理模，[hybrid Transformer–Mamba](https://arxiv.org/pdf/2504.03624)，thinking budget 可拧——精度、吞吐、成本自己调。

- **Open。** [Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)。声称 reasoning、coding、agentic（instruction following、tool calling、长上下文聊天）领先。超过 **9T** token 的 [pre- / post-training 数据](https://huggingface.co/nvidia/datasets?search=nemotron)，NVIDIA 产，许可宽松。

- **Efficient。** Hybrid：关键 thinking token 相对同尺寸次好开源 dense 最高 **6×** 快（用 vLLM）。吞吐上去 → 想得快、搜索空间大、自我反省更好、精度更高。

本地图（原文版权仍归原站；学习对照用）：

![figure1](../../../../assets/vllm/blog/serving/nemotron-nano2/01-figure1.png)

**Figure 1。** Nano 2 9B 在常见榜上的精度。分数在图里，不在表里。

- **Optimized Thinking。** Thinking budget：别让代理 overthink；推理成本更好预期。放着不管，模型会想太多——钱上去，精度有时还下来。Budget 让你选精度–token 的甜区。

![figure2](../../../../assets/vllm/blog/serving/nemotron-nano2/02-figure2.png)

**Figure 2。** Nano 2 9B 在不同 “Token Budget” 阈值下的精度。轴上的数字在图里。

## 用 vLLM 上手 Nemotron

```bash
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --trust-remote-code \
    --mamba_ssm_cache_dtype float32
```

然后用 `ThinkingBudgetClient` 包住 OpenAI-compatible 端点。两刀，tokenizer 在客户端——**不是**引擎旗标。

```python
from typing import Any, Dict, List
import openai
from transformers import AutoTokenizer

class ThinkingBudgetClient:

   def __init__(self, base_url: str, api_key: str, tokenizer_name_or_path: str):
       self.base_url = base_url
       self.api_key = api_key
       self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
       self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)

   def chat_completion(
       self,
       model: str,
       messages: List[Dict[str, Any]],
       max_thinking_budget: int = 512,
       max_tokens: int = 1024,
       **kwargs,
   ) -> Dict[str, Any]:

       assert (
           max_tokens > max_thinking_budget
       ), f"thinking budget must be smaller than maximum new tokens. Given {max_tokens=} and {max_thinking_budget=}"

       # 1. first call chat completion to get reasoning content
       response = self.client.chat.completions.create(
           model=model, messages=messages, max_tokens=max_thinking_budget, **kwargs
       )
       content = response.choices[0].message.content
       reasoning_content = content

       if not "</think>" in reasoning_content:
           # reasoning content is too long, closed with a period (.)
           reasoning_content = f"{reasoning_content}.n</think>nn"
       reasoning_tokens_len = len(
           self.tokenizer.encode(reasoning_content, add_special_tokens=False)
       )
       remaining_tokens = max_tokens - reasoning_tokens_len

       assert (
           remaining_tokens > 0
       ), f"remaining tokens must be positive. Given {remaining_tokens=}. Increase the max_tokens or lower the max_thinking_budget."

       # 2. append reasoning content to messages and call completion
       messages.append({"role": "assistant", "content": reasoning_content})
       prompt = self.tokenizer.apply_chat_template(
           messages,
           tokenize=False,
           continue_final_message=True,
       )

       response = self.client.completions.create(
           model=model, prompt=prompt, max_tokens=remaining_tokens, **kwargs
       )

       response_data = {
           "reasoning_content": reasoning_content.strip().strip("</think>").strip(),
           "content": response.choices[0].text,
           "finish_reason": response.choices[0].finish_reason,
       }

       return response_data
```

`.n</think>nn` 按原文抄（多半是想写 `\n</think>\n\n`）。流程：chat.completions 用 `max_tokens=budget` → 也许缝上 `</think>` → 追加 assistant 轮 → `apply_chat_template(..., continue_final_message=True)` → `/completions` 写答案。

示例请求（`max_thinking_budget=32`，`max_tokens=512`，`temperature=0.6`，`top_p=0.95`；系统提示带 `/think`）：

```python
tokenizer_name_or_path = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"

client = ThinkingBudgetClient(
   base_url="http://localhost:8000/v1",  # Nano 9B v2 deployed in thinking mode
   api_key="EMPTY",
   tokenizer_name_or_path=tokenizer_name_or_path,
)

result = client.chat_completion(
   model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
   messages=[
       {"role": "system", "content": "You are a helpful assistant. /think"},
       {"role": "user", "content": "What is 2+2?"},
   ],
   max_thinking_budget=32,
   top_p=0.95,
   temperature=0.6,
   max_tokens=512,
)

print(result)
```

页上的返回：

```
{'reasoning_content': 'Okay, the user asked "What is 2+2?" Let me think. This is a basic arithmetic question. The answer should be straightforward. I need.', 'content': '2 + 2 equals **4**. nnLet me know if you need help with anything else! 😊n', 'finish_reason': 'stop'}
```

vLLM 这边：KV-cache 管理和长上下文，对得上 hybrid Transformer-Mamba。模型卡：[NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)。vLLM：[Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart.html)。

## Run anywhere

配置成能在 GPU 加速系统上跑，从开发到生产。

托管端点：[build.nvidia.com](https://build.nvidia.com/nvidia/nvidia-nemotron-nano-9b-v2)，或从 Hugging Face 下。

想法墙：[nemotron.ideas.nvidia.com](http://nemotron.ideas.nvidia.com/?ncid=so-othe-692335)。订阅：[NVIDIA Nemotron](https://developer.nvidia.com/nemotron)，NVIDIA AI 的 [LinkedIn](https://www.linkedin.com/showcase/nvidia-ai/posts/?feedView=all)、[X](https://x.com/NVIDIAAIDev)、[YouTube](https://www.youtube.com/@NVIDIADeveloper)，Discord 上的 [Nemotron channel](https://discord.com/channels/1019361803752456192/1407781691698708682) / [invite](https://discord.com/invite/nvidiadeveloper)。
