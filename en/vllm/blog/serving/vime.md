---
source: https://vllm.ai/blog/2026-06-09-announcing-vime
lang: en
fetched: 2026-09-01
---

# vime: slime train + vLLM rollout, `--vllm-` CLI prefix

Chinese: `../../zh/vllm/blog/serving/vime.md`  
ROCm: [vime-rocm](vime-rocm.md).

slime trains; vLLM rollouts. Don’t keep two dictionaries — `--vllm-` pins inference knobs on one CLI. GB200 vs H200 step time ~**1.72×** in their sweep. If train GPUs and sample GPUs run different engines, weight sync and logprob alignment crack. vime’s claim is one vLLM path.
