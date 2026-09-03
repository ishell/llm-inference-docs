---
source: https://vllm.ai/blog/2026-03-13-p-eagle
lang: en
fetched: 2026-09-01
---

# P-EAGLE: K drafts in one forward

Chinese: [zh/vllm/blog/performance/p-eagle.md](../../../../zh/vllm/blog/performance/p-eagle.md)  
vLLM ≥0.16.0, PR#32887. Numbers are **one B200**, GPT-OSS-20B. 

EAGLE drafts autoregressively: K tokens need K draft forwards. Better drafts and larger K make that tax worse. P-EAGLE emits all K in one pass. Versus a public EAGLE-3 checkpoint: about **1.55–1.69×** TPS at low concurrency, still **1.05–1.25×** at c=64. Peak often at K=7; EAGLE-3 often at K=3 — parallel depth is almost free; linear drafting is not.


Local figures (copyright remains with the original site; study copies):

![fig1 speedbench overview](../../../../assets/vllm/blog/performance/p-eagle/01-fig1_speedbench_overview.png)

![fig2 architecture](../../../../assets/vllm/blog/performance/p-eagle/02-fig2_architecture.png)

![fig3 sequence length](../../../../assets/vllm/blog/performance/p-eagle/03-fig3_sequence_length.png)

![fig4 mtbench](../../../../assets/vllm/blog/performance/p-eagle/04-fig4_mtbench.png)

![fig5 humaneval](../../../../assets/vllm/blog/performance/p-eagle/05-fig5_humaneval.png)

![fig6 speedbench](../../../../assets/vllm/blog/performance/p-eagle/06-fig6_speedbench.png)

## Structure

Prefill matches vanilla EAGLE: target finishes the prompt, leaves `h_prompt` / `h_context`. The drafter concatenates embedding and hidden per position and runs N layers once:

- Position 1 (NTP): new token + `h_context`, same as autoregressive EAGLE.
- Positions 2…K (MTP): missing token/hidden filled with a learned **mask embedding** and **shared hidden**.

Acceptance length is higher too: K=7 HumanEval 3.94 vs EAGLE-3 3.03. Deeper speculation in the same forward, more accepts.

## Why the engine squirms

The draft batch no longer matches verification: MASK slots, rebuilt slot maps. A fused Triton kernel expands tokens / positions / masks on GPU; hidden is larger, so a second copy kernel broadcasts the learned placeholder into mask slots. Rejected tokens map to `PADDING_SLOT_ID (-1)` so they do not dirty KV. CUDA-graph capture range grows by `K × max_num_seqs`.

## How to turn it on

Pretrained heads on HF for GPT-OSS 20B/120B and Qwen3-Coder 30B. `"parallel_drafting": true`. GPT-OSS-20B + EAGLE still needed patch PR#36684 at the time.

```bash
vllm serve openai/gpt-oss-20b \
  --speculative-config '{"method":"eagle3","model":"amazon/gpt-oss-20b-p-eagle","num_speculative_tokens":5,"parallel_drafting":true}'
```

Training long sequences: N×K positions explode attention; they partition inside a sequence and keep attention deps across chunks (paper). Read with [spec-decode](spec-decode.md) and [parallel drafting](parallel-drafting.md).
