---
source: https://vllm.ai/blog/2026-08-20-verl-omni-v0-2-0
lang: en
fetched: 2026-09-01
---

# verl-Omni v0.2: request-level batching, gen 226s → 108s

Chinese: `../../zh/vllm/blog/serving/verl-omni-v020.md`  
Prior: [verl-omni](verl-omni.md).

v0.1 was “it runs”; v0.2 batches generation at request grain. Wall-clock gen **226s → 108s**. MMK12 val **0.833**. Rollout is usually the RL step’s fat part — wait-for-the-whole-batch → request-in/request-out is why the clock drops. Figures still on the original page.
