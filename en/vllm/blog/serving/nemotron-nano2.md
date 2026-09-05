---
source: https://vllm.ai/blog/2025-10-23-now_serving_nvidia_nemotron_with_vllm
lang: en
fetched: 2026-09-04
---

# Nemotron Nano 2: 9B hybrid; Thinking Budget is a two-call client, not an engine flag

Chinese: [zh/vllm/blog/serving/nemotron-nano2.md](../../../../zh/vllm/blog/serving/nemotron-nano2.md)

2025-10-23. **NVIDIA Nemotron Team**. First Nemotron-on-vLLM day-0 in this series. Successor 30B: [Nano](nemotron-3-nano.md). Same family: [Super](nemotron-3-super.md), [Ultra](nemotron-3-ultra.md), [Lightning](nemotron-35-lightning.md). Multimodal: [Nano 2 VL](nemotron-nano-vl.md), [Nano Omni](nemotron-omni.md). Hybrid Mamba: [hybrid-ssm.md](hybrid-ssm.md). Claimed thinking-token speedup up to ~**6×** vs similar-size dense — page demo, not your SLA.

**TL;DR from the page:**

- [NVIDIA Nemotron Nano 2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2): 9B hybrid Transformer–Mamba; configurable thinking budget.
- Open weights and **>9T** tokens of pre- and post-training data on Hugging Face.
- Serve: `--trust-remote-code --mamba_ssm_cache_dtype float32`.
- Budget is an **application protocol**: first `max_tokens=budget` for reasoning; if no `</think>`, append it; then `/completions` with `continue_final_message`. **Not** `--thinking-budget`.

## Why this model

Agentic systems need tools that are open, efficient, and ready to scale. [NVIDIA Nemotron](https://developer.nvidia.com/nemotron) is billed as a family of open models, datasets, and technologies for specialized agentic AI.

vLLM is the path to deploy that family on datacenter and edge hardware. Out of the box, open weights and open data.

## NVIDIA Nemotron Nano 2

Latest addition then: a small language reasoning model with a [hybrid Transformer–Mamba architecture](https://arxiv.org/pdf/2504.03624) and a configurable thinking budget — dial accuracy, throughput, and cost.

- **Open.** [Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2). Claimed leading accuracy for reasoning, coding, agentic tasks (instruction following, tool calling, long-context chat). Over **9T** tokens of [pre- and post-training data](https://huggingface.co/nvidia/datasets?search=nemotron), NVIDIA-generated, permissive license.

- **Efficient.** Hybrid architecture: critical thinking tokens up to **6×** faster vs the next-best open dense model of similar size on vLLM. Higher throughput → think faster, larger search space, better self-reflection, higher accuracy.

Local figures (copyright remains with the original site; study copies):

![figure1](../../../../assets/vllm/blog/serving/nemotron-nano2/01-figure1.png)

**Figure 1.** Accuracy of Nemotron Nano 2 9B on various popular benchmarks. Scores live in the chart, not in a table.

- **Optimized Thinking.** Thinking budget: avoid agent overthinking; more predictable inference cost. Left alone, models can overthink — cost up, accuracy sometimes down. Budget lets you pick an accuracy–token sweet spot.

![figure2](../../../../assets/vllm/blog/serving/nemotron-nano2/02-figure2.png)

**Figure 2.** Accuracy of Nano 2 9B at various “Token Budget” thresholds. Axis numbers are in the chart.

## Get started with Nemotron using vLLM

```bash
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --trust-remote-code \
    --mamba_ssm_cache_dtype float32
```

Then a `ThinkingBudgetClient` around the OpenAI-compatible endpoint. Two calls, tokenizer in the client — **not** an engine flag.

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

The `.n</think>nn` string is as published (likely meant `\n</think>\n\n`). Flow: chat.completions with `max_tokens=budget` → maybe stitch `</think>` → append assistant turn → `apply_chat_template(..., continue_final_message=True)` → `/completions` for the answer.

Example request (`max_thinking_budget=32`, `max_tokens=512`, `temperature=0.6`, `top_p=0.95`; system prompt includes `/think`):

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

Example dump from the page:

```
{'reasoning_content': 'Okay, the user asked "What is 2+2?" Let me think. This is a basic arithmetic question. The answer should be straightforward. I need.', 'content': '2 + 2 equals **4**. nnLet me know if you need help with anything else! 😊n', 'finish_reason': 'stop'}
```

vLLM side: KV-cache management and long context, matching the hybrid Transformer-Mamba. Model card: [NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2). vLLM: [Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart.html).

## Run anywhere

Configured to run across GPU-accelerated systems, development to production.

Hosted endpoint: [build.nvidia.com](https://build.nvidia.com/nvidia/nvidia-nemotron-nano-9b-v2), or download from Hugging Face.

Ideas board: [nemotron.ideas.nvidia.com](http://nemotron.ideas.nvidia.com/?ncid=so-othe-692335). Stay-up-to-date: [NVIDIA Nemotron](https://developer.nvidia.com/nemotron), NVIDIA AI on [LinkedIn](https://www.linkedin.com/showcase/nvidia-ai/posts/?feedView=all), [X](https://x.com/NVIDIAAIDev), [YouTube](https://www.youtube.com/@NVIDIADeveloper), [Nemotron Discord channel](https://discord.com/channels/1019361803752456192/1407781691698708682) / [invite](https://discord.com/invite/nvidiadeveloper).
