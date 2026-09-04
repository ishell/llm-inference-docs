---
source: https://vllm.ai/blog/2023-06-20-vllm
lang: en
fetched: 2026-09-04
---

# vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention

Chinese: [zh/vllm/blog/architecture/paged-attention.md](../../../../zh/vllm/blog/architecture/paged-attention.md)

Launch post, 2023-06-20. Woosuk Kwon & Zhuohan Li (UC Berkeley). Paper later: [arXiv:2309.06180](https://arxiv.org/pdf/2309.06180.pdf). Links on the original page: [GitHub](https://github.com/vllm-project/vllm), then-current docs on Read the Docs.

Claim: up to **24×** throughput vs HuggingFace Transformers, **no architecture changes**. Up to **3.5×** vs HuggingFace Text Generation Inference (TGI), then the previous SOTA. Already serving [Chatbot Arena / Vicuna Demo](https://chat.lmsys.org/) for two months. Developed at UC Berkeley; LMSYS used it to make LLM serving affordable on a small research team's GPUs.

Local figures (copyright remains with the original site; study copies). Official page also has four GIFs of paging / sharing; the Chinese note uses a study diagram for the OS metaphor.

![perf a100 n1 light](../../../../assets/vllm/blog/architecture/paged-attention/01-perf_a100_n1_light.png)

![perf a10g n1 light](../../../../assets/vllm/blog/architecture/paged-attention/02-perf_a10g_n1_light.png)

![perf a100 n3 light](../../../../assets/vllm/blog/architecture/paged-attention/03-perf_a100_n3_light.png)

![perf a10g n3 light](../../../../assets/vllm/blog/architecture/paged-attention/04-perf_a10g_n3_light.png)

![annimation0](../../../../assets/vllm/blog/architecture/paged-attention/05-annimation0.gif)

![annimation1](../../../../assets/vllm/blog/architecture/paged-attention/06-annimation1.gif)

![annimation2](../../../../assets/vllm/blog/architecture/paged-attention/07-annimation2.gif)

![annimation3](../../../../assets/vllm/blog/architecture/paged-attention/08-annimation3.gif)

![lmsys traffic](../../../../assets/vllm/blog/architecture/paged-attention/09-lmsys_traffic.png)

## Beyond SOTA performance

Benchmarks vs [HF Transformers](https://huggingface.co/docs/transformers/main_classes/text_generation) and [HF TGI](https://github.com/huggingface/text-generation-inference). Workload: ShareGPT-sampled input/output lengths.

- Hardware: **LLaMA-7B on NVIDIA A10G**, **LLaMA-13B on NVIDIA A100 40GB**.
- One completion per request: **14×–24×** vs HF, **2.2×–2.5×** vs TGI.
- Three parallel completions: **8.5×–15×** vs HF, **3.3×–3.5×** vs TGI.

Parallel sampling is more KV-hungry; PagedAttention shares the prompt pages.

## The secret sauce: PagedAttention

Bottleneck is **memory**, not FLOPs. Autoregressive Decode keeps every token's K/V in GPU memory (KV cache):

- **Large:** up to **1.7GB for a single sequence** in LLaMA-13B.
- **Dynamic:** size tracks sequence length, which is variable and unpredictable.

Existing systems wasted **60%–80%** of that memory to fragmentation and over-reservation.

PagedAttention is attention that stores a sequence's KV in **non-contiguous** space, borrowing virtual memory / paging. Partition each sequence's KV into **blocks** of a fixed token count. The kernel looks up blocks through a **block table** (logical blocks → physical blocks). Physical blocks are allocated on demand as new tokens appear.

Waste is almost only the **last, partially filled block** — **under 4%** in practice. More sequences fit in the batch → higher GPU utilization → the throughput numbers above.

Second gift: **sharing**. In parallel sampling, several outputs share one prompt. Sequences map logical blocks onto the same physical block, with **reference counts** and **copy-on-write**. Parallel sampling and beam search cut memory overhead by up to **55%**, up to **2.2×** throughput — those algorithms become serviceable, not lab-only.

Technical follow-up: GitHub then, paper later. The later [Anatomy](anatomy.md), V1, and prefix-cache posts still sit on this page table.

## LMSYS Vicuna and Chatbot Arena

April 2023: LMSYS released Vicuna. FastChat first used an HF Transformers [model worker](https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/model_worker.py). Peak traffic jumped several times; HF became the bottleneck. LMSYS + vLLM shipped FastChat-vLLM (`vllm_worker.py`). Early internal microbench: up to **~30×** vs the original HF backend; production absorbed **~5×** more peak traffic.

From mid-April, Vicuna, Koala, and LLaMA were served with FastChat as the multi-model frontend and vLLM as the backend, on a limited number of university GPUs. LMSYS was expanding to Databricks Dolly, LAION OpenAssistant, and Stability AI StableLM; more model support was forthcoming.

Arena traffic (April–May): **more than half** of requests used vLLM. GPU count for that traffic **cut ~50%**. Average **~30K requests/day**, peak **~60K**.

## Get started (then-current API)

```bash
pip install vllm
```

Offline:

```python
from vllm import LLM

prompts = ["Hello, my name is", "The capital of France is"]
llm = LLM(model="lmsys/vicuna-7b-v1.3")
outputs = llm.generate(prompts)
```

Online OpenAI-compatible server (vintage entrypoint; today: `vllm serve`):

```bash
python -m vllm.entrypoints.openai.api_server --model lmsys/vicuna-7b-v1.3
```

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lmsys/vicuna-7b-v1.3",
    "prompt": "San Francisco is a",
    "max_tokens": 7,
    "temperature": 0
  }'
```

Then-current install / quickstart lived on the docs site.

## Credits

Written by Woosuk Kwon and Zhuohan Li (UC Berkeley). Hao Zhang: FastChat integration and that section of the post. Team thanked: Siyuan Zhuang, Ying Sheng, Lianmin Zheng (UC Berkeley), Cody Yu, Joey Gonzalez (UC Berkeley), Hao Zhang (UC Berkeley & UCSD), Ion Stoica (UC Berkeley).
