---
source: https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting
lang: en
fetched: 2026-09-01
---

# Parallel drafting: P-EAGLE / DFlash / DSpark

Chinese: `../../zh/vllm/blog/performance/parallel-drafting.md`  
**Errata 2026-07-29**: Figure 1 had an environment bug; relative ranking unchanged — trust the corrected numbers, not the first figure.

[P-EAGLE](p-eagle.md) collapses K draft forwards into one. DFlash and DSpark sit on the same road: the draft is not an autoregressive queue; one forward lays out K candidates. Verification is still **rejection sampling** — lossless; the target distribution does not change.


Local figures (copyright remains with the original site; study copies):

![compare interactivity qwen38b math](../../../../assets/vllm/blog/performance/parallel-drafting/01-compare_interactivity_qwen38b_math.png)

![compare interactivity qwen330b humaneval](../../../../assets/vllm/blog/performance/parallel-drafting/02-compare_interactivity_qwen330b_humaneval.png)

![compare interactivity gemma431b humaneval](../../../../assets/vllm/blog/performance/parallel-drafting/03-compare_interactivity_gemma431b_humaneval.png)

![ar vs parallel](../../../../assets/vllm/blog/performance/parallel-drafting/04-ar_vs_parallel.jpg)

![diagram](../../../../assets/vllm/blog/performance/parallel-drafting/05-diagram.jpg)

## How to turn it on

DFlash example (checkpoint from the original post / Speculators repo):

```bash
vllm serve <target> \
  --speculative-config '{"method":"dflash","model":"<dflash-head>","num_speculative_tokens":K}'
```

P-EAGLE uses `method: eagle3` + `"parallel_drafting": true`. Adaptive DSpark verification: [dspark-adaptive](dspark-adaptive.md).

Peak K is often larger than linear EAGLE-3: depth is almost free; linear drafting pays one forward per extra token. Read with [spec-decode](spec-decode.md): this changes **how the draft grows**, not the accept math.
