---
source: https://vllm.ai/blog/2025-10-23-now_serving_nvidia_nemotron_with_vllm
lang: en
fetched: 2026-09-01
---

# Nemotron Nano 2: 9B hybrid; Thinking Budget is a two-call client, not an engine flag

Chinese: [zh/vllm/blog/serving/nemotron-nano2.md](../../../../zh/vllm/blog/serving/nemotron-nano2.md)  
They quote thinking tokens up to ~**6×** vs similar-size dense.

```
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 --trust-remote-code --mamba_ssm_cache_dtype float32
```

Budget: first `max_tokens=budget` for reasoning; if no `</think>`, append it; then push that assistant turn and `/completions` with `continue_final_message` for the answer. Application protocol, not `--thinking-budget`. Nemotron 3: [nemotron-3-nano](nemotron-3-nano.md).

Local figures (copyright remains with the original site; study copies):

![figure1](../../../../assets/vllm/blog/serving/nemotron-nano2/01-figure1.png)

![figure2](../../../../assets/vllm/blog/serving/nemotron-nano2/02-figure2.png)
