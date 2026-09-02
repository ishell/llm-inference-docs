---
source: https://vllm.ai/blog/2026-08-07-decode-context-parallelism
lang: en
fetched: 2026-08-31
---

# Decode Context Parallelism

2026-08-07. CLI: `-dcp` / `--decode-context-parallel-size`. Sibling idea in TRT-LLM: Helix Parallelism. 

Long-context agents (64K–1M) make KV huge. TP shards KV **by head**. GQA bottoms out at one KV head per GPU then duplicates; MLA has one latent, fully replicated on every TP rank. DCP shards KV **by sequence**. A 200K request can be 50K per GPU of four. Needs a fast GPU interconnect.

**Demo:** 8×B200, Kimi K2.6 NVFP4, public Mooncake-trace agents (median ~67K in / ~400 out; ~half ≥64K, tail ~1M). Baseline TP: KV 100% at concurrency 64, ~**1,863 tok/s/GPU**. DCP: concurrency 512 at ~**82%** KV, ~**6,091 tok/s/GPU**. 200k+ still sits on the same throughput–interactivity frontier.

Rhythm: AllGather Q → local attention → AllGather+ReduceScatter (`cp_lse_ag_out_rs`). Decode Q is one token. MLA opt-in `VLLM_DCP_Q_REPLICATE=1` skips the Q all-gather. Merge uses online-softmax LSE.

```bash
vllm serve deepseek-ai/DeepSeek-V2-Lite \
    --tensor-parallel-size 2 \
    --decode-context-parallel-size 2
```

MLA: `TP >= DCP` and `TP % DCP == 0` (R1 can be 8/8). GQA: DCP only fills the duplicates after `TP > num_kv_heads`; `(TP // num_kv_heads) >= DCP` and divides (Qwen3-235B, 4 KV heads, TP8 → DCP≤2).

Roadmap: finer TP/DCP, A2A kernels, MTP/spec, P/D, Prefill Context Parallelism (`-pcp`).

Local figures (copyright remains with the original site; study copies):

![kv parallelism overview](../../../../assets/vllm/blog/performance/dcp/01-kv-parallelism-overview.svg)

![figure 1](../../../../assets/vllm/blog/performance/dcp/02-figure-1.png)

![figure 2](../../../../assets/vllm/blog/performance/dcp/03-figure-2.png)

![figure 3](../../../../assets/vllm/blog/performance/dcp/04-figure-3.png)

![figure 4](../../../../assets/vllm/blog/performance/dcp/05-figure-4.png)

![figure 5](../../../../assets/vllm/blog/performance/dcp/06-figure-5.png)
