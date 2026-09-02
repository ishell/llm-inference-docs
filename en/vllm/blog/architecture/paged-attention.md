---
source: https://vllm.ai/blog/2023-06-20-vllm
lang: en
fetched: 2026-08-31
---

# vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention

Launch post, 2023-06-20. Woosuk Kwon & Zhuohan Li (UC Berkeley). Figures stay on the official page. Numbers are that era’s ShareGPT setup: LLaMA-7B on A10G, LLaMA-13B on A100-40GB.

Claim: up to **24×** throughput vs HuggingFace Transformers, no architecture changes. Up to **3.5×** vs HF TGI. Already serving Chatbot Arena / Vicuna Demo.


Local figures (copyright remains with the original site; study copies):

![perf a100 n1 light](../../../../assets/vllm/blog/architecture/paged-attention/01-perf_a100_n1_light.png)

![perf a10g n1 light](../../../../assets/vllm/blog/architecture/paged-attention/02-perf_a10g_n1_light.png)

![perf a100 n3 light](../../../../assets/vllm/blog/architecture/paged-attention/03-perf_a100_n3_light.png)

![perf a10g n3 light](../../../../assets/vllm/blog/architecture/paged-attention/04-perf_a10g_n3_light.png)

![annimation0](../../../../assets/vllm/blog/architecture/paged-attention/05-annimation0.gif)

![annimation1](../../../../assets/vllm/blog/architecture/paged-attention/06-annimation1.gif)

![annimation2](../../../../assets/vllm/blog/architecture/paged-attention/07-annimation2.gif)

![annimation3](../../../../assets/vllm/blog/architecture/paged-attention/08-annimation3.gif)

![lmsys traffic](../../../../assets/vllm/blog/architecture/paged-attention/09-lmsys_traffic.png)

## Throughput

- One completion per request: **14×–24×** vs HF, **2.2×–2.5×** vs TGI.
- Three parallel completions: **8.5×–15×** vs HF, **3.3×–3.5×** vs TGI.

## The bottleneck is memory

Autoregressive decode keeps K/V in GPU memory (KV cache). LLaMA-13B: up to **1.7GB per sequence**. Size is dynamic. Existing systems wasted **60%–80%** to fragmentation and over-reservation.

## PagedAttention

OS paging metaphor:

| OS | PagedAttention |
|---|---|
| page | KV **block** (fixed token count) |
| byte | token |
| process | sequence |
| page table | **block table** (logical → physical) |

Blocks need not be physically contiguous. Allocate on demand. Waste is mostly the last partially filled block — **<4%** in practice. More sequences in a batch → higher GPU util → the throughput numbers above.

Sharing: parallel sampling / beam search map several sequences onto the same physical prompt blocks, with refcounts and **copy-on-write**. Memory for those algorithms down ~**55%**, up to **2.2×** throughput.

## LMSYS

FastChat moved from HF backend to FastChat-vLLM. Internal microbench up to **30×** vs the original HF path; production absorbed ~**5×** more peak traffic. GPU count for that traffic **−50%**. ~30K req/day average, 60K peak. More than half of Arena requests used vLLM (Apr–May 2023).

## Then-current API

```python
from vllm import LLM
llm = LLM(model="lmsys/vicuna-7b-v1.3")
outputs = llm.generate(["Hello, my name is", "The capital of France is"])
```

```bash
python -m vllm.entrypoints.openai.api_server --model lmsys/vicuna-7b-v1.3
```

Today: `vllm serve`. Next: `anatomy.md`.
