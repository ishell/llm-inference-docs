---
source: https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification
lang: en
fetched: 2026-09-04
---

# Adaptive Verification in vLLM: DSpark confidence-scheduled verification

Chinese: [zh/vllm/blog/performance/dspark-adaptive.md](../../../../zh/vllm/blog/performance/dspark-adaptive.md)

2026-08-14. **vLLM Team**. Study note. Landed in [PR #47808](https://github.com/vllm-project/vllm/pull/47808) as `enable_adaptive_verification`. Demo on the page: DeepSeek-V4-Pro-0813, **TP=8** on **8×B300** (SM100). Parallel-drafting family: [parallel-drafting.md](parallel-drafting.md). Accept math: [spec-decode.md](spec-decode.md). DSpark paper: [arXiv 2607.05147](https://arxiv.org/abs/2607.05147).

Speculative decoding buys fewer Decode steps with more compute. At batch size **1** that is a good trade: the GPU is memory-bound with spare compute, so extra draft tokens are close to free. At batch size **256** the trade is delicate. Draft tokens compete with real tokens for the same compute; every rejected token wastes useful work; enough rejects and throughput drops.

**TL;DR from the page.** [DSpark](https://arxiv.org/abs/2607.05147)'s confidence head scores each drafted token's chance of surviving verification. Instead of picking a speculation length per deployment, vLLM can decide **per step** how much of the draft to verify. With adaptive verification on (`num_speculative_tokens: 7`), speculative decoding still helps out to **concurrency 256**, and still keeps the long-draft win at low concurrency. That cuts the need to tune `num_speculative_tokens` per workload. The post frames DSpark as an easier “on-by-default” win.

This changes **how many tokens you verify per step**, not the draft architecture.

## The problem

Per-position acceptance decays fast. On DeepSeek-V4-Pro-0813, the **last** drafted token of a **7-token** block survives **less than 10%** of the time, against **more than 70%** for the first. That low-probability token still occupies a slot in every verification batch. While the GPU is memory-bound the slot is almost free and worth the gamble; once the GPU saturates, the gamble has a real throughput cost.

The crossover moves with load and with workload-dependent acceptance, so **no static** `num_speculative_tokens` is optimal across concurrencies. DSpark's answer: an adaptive draft **budget** that sees both system load and how confident the DSpark head is that the target will accept each draft token.

## Scheduling the budget

DSpark drafts a block of *k* tokens per pass (`num_speculative_tokens`) and emits a confidence per position from a learned confidence head. The scheduler turns those into survival probabilities: the running product along each request

$$
S(r, i) = \prod_{j \le i} \mathrm{confidence}(r, j)
$$

Survival only decreases with position *i*. Given a draft-token budget *B*, allocating it to the most probable draft sequences is a global top-*B* over survival scores. That admits a **contiguous prefix** of each request's draft with no extra constraint. Slots compete **across** requests: position 5 of a confident request can outrank position 1 of a low-confidence one.

![fig1 policy](../../../../assets/vllm/blog/performance/dspark-adaptive/01-fig1-policy.svg)

**Figure 1.** Same batch, two policies. Fixed verification pays for all **21** slots, including near-zero survival. Adaptive verification verifies only the best **B=11**.

*B* maximizes expected tokens per unit of step time:

$$
B^* = \arg\max_B \frac{N_\mathrm{sampling} + \sum_{j < B} S_\mathrm{sorted}[j]}{\mathrm{draft\_cost}[\mathrm{num\_reqs}] + \mathrm{verify\_cost}[T + B]}
$$

Numerator: one bonus token per sampling request, plus survival of the *B* best draft slots. *N*<sub>sampling</sub> counts requests that will **actually sample this step** — a request still in chunked Prefill contributes nothing. Denominator: a profiled cost table, indexed by the step's token count. *T* is already-scheduled tokens that are not drafts, so *T* + *B* is the whole step. Both sides are arrays; the choice is `np.argmax` over a cumulative sum. Costs are in **microseconds**.

Sizing runs on the **CPU** while the GPU is still on the previous step, from a **double-buffered** confidence array that is **one step old**. Handing those *B* slots out to individual requests runs on the **GPU** against **current** confidences. Selection is PyTorch, lowered to Triton by `torch.compile`, and **never reads back to the host**.

## Varlen decode CUDA graphs

Variable-sized verifications need **varlen decode CUDA graphs**. That needs attention-kernel support: sparse MLA kernels are naturally varlen (each query token has an independent top-k). DeepSeek open-sourced a varlen indexer kernel in [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), integrated as part of [PR #47808](https://github.com/vllm-project/vllm/pull/47808).

Decode graphs are captured with `num_reqs = min(num_tokens, max_num_seqs)` and a promised `max_query_len = num_speculative_tokens + 1`. One graph then serves any mix of **1** to `num_speculative_tokens + 1` tokens per request.

## The cost model

The budget rule divides by a step cost, so that cost has to be cheap to look up and close to reality. At startup the engine times **dummy steps** across a fixed set of shapes (CUDA graph shapes plus a couple above the max cudagraph size), taking the **median of five runs** per shape. That becomes two flat lookup tables:

- verification table, indexed by **token count**
- drafter table, indexed by **request count** (drafting costs the same no matter how many tokens are verified)

The two are summed.

![fig2 costcurve](../../../../assets/vllm/blog/performance/dspark-adaptive/02-fig2-costcurve.svg)

**Figure 2.** Both cost tables from a real startup profile; cost is the median of 5 samples. The post does not print the table entries as numbers.

Inside captured CUDA graphs, cost is a **staircase**, not a line: cudagraph padding means a batch of **121** tokens runs the **128**-token graph and (mostly) pays for all 128. Past the capture limit the staircase ends and cost is continuous. There is a notable jump leaving the cudagraph region; that jump is sharp enough that the budget algorithm is **strongly encouraged to stay inside** the cudagraph region.

Profiling noise: the curve is forced **monotonic**. Real step cost can fall as the batch grows (kernel tile sizes), so monotonicity is a smoother, not a claim that hardware is monotonic. Dummy steps use a synthetic KV context, **8192** tokens by default, tunable with `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN`.

## Results

Setup on the page: **DeepSeek-V4-Pro-0813**, **TP=8** on **8×B300** (SM100), expert parallel, FP8 KV cache, `max_model_len` **16384**, `max_cudagraph_capture_size` **4096**, vLLM `main` at `73b8394`. Benchmark: **880** prompts, temperature **1.0**, up to **2048** output tokens, concurrency swept **1 to 256**.

![fig3 pareto](../../../../assets/vllm/blog/performance/dspark-adaptive/03-fig3-pareto.svg)

**Figure 3.** Throughput versus interactivity for different speculation schemes. Adaptive verification stays on the Pareto frontier throughout.

The page's claim: adaptive verification stays on the **edge of the Pareto curve for the whole sweep**, and well outside no-speculation at **both** ends. Read the graph as: long fixed block at low concurrency, short fixed block at high concurrency — both, without knowing the workload shape in advance. The TL;DR repeats the same claim out to **c=256**. The post does **not** print tok/s or ITL numbers for the points; `output_throughput` from the result JSON is what they plotted.

## Limitations

- **FULL** varlen decode graphs require `AttentionCGSupport.ALWAYS`. The DSV4 sparse-MLA, sparse-SWA, and indexer backends report that on **SM100**. Elsewhere adaptive verification is **rejected at startup**, not fallen back to PIECEWISE.
- `--enforce-eager` (step costs are profiled from captured graphs), **LoRA**, and **pipeline parallelism** were all unsupported at the time.
- Output **logprobs** are rejected when adaptive verification is on: verification **compacts logits** after the forward pass.

## Appendix: reproducing

Commands below use [PR #47808](https://github.com/vllm-project/vllm/pull/47808), then merged into vLLM `main`. The numbers above were measured at `73b8394`.

**Server** (all measurements; ablations are `--speculative-config` deltas):

```bash
vllm serve deepseek-ai/DeepSeek-V4-Pro-0813 \
  --tokenizer-mode deepseek_v4 --trust-remote-code \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --kv-cache-dtype fp8 --max-model-len 16384 --max-num-seqs 256 \
  --max-num-batched-tokens 16384 --gpu-memory-utilization 0.8 \
  --compilation-config '{"max_cudagraph_capture_size":4096}' \
  --speculative-config '{"method":"dspark","attention_backend":"FLASH_ATTN","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}'
```

Draft defaults to the **target checkpoint**, so `"model"` can be omitted. `--kv-cache-dtype fp8` is **required**: the `fp8_ds_mla` layout rejects other KV dtypes. `--max-num-seqs` matters: default is **128**, which would cap the batch below the top of the concurrency sweep. They raise `max_cudagraph_capture_size` to `(num_speculative_tokens + 1) * max_num_seq` so every verification batch sits inside a cudagraph. Larger capture needs more memory for cudagraphs, hence `--gpu-memory-utilization 0.8`; at the default it **OOMs while capturing**.

Ablations:

- fixed k: `"enable_adaptive_verification": false`, `"num_speculative_tokens": k`, for k ≥ `dspark_block_size` (**5** on this checkpoint)
- no speculation: omit `--speculative-config`

**Throughput sweep**, per concurrency `c ∈ {1, 16, 32, 64, 128, 256}`, after one warmup (`--speed-bench-output-len 256 --num-prompts 64 --max-concurrency 32`):

```bash
MODEL=deepseek-ai/DeepSeek-V4-Pro-0813
for c in 256 128 64 32 16 1; do
  n=880; [ "$c" = 1 ] && n=240
  vllm bench serve \
    --backend openai-chat --base-url http://127.0.0.1:8000 \
    --endpoint /v1/chat/completions --model "$MODEL" \
    --tokenizer "$MODEL" --tokenizer-mode deepseek_v4 \
    --dataset-name speed_bench --dataset-path <speed-bench-dir> \
    --speed-bench-dataset-subset qualitative --speed-bench-output-len 2048 \
    --num-prompts $n --max-concurrency $c --request-rate inf \
    --skip-chat-template --disable-shuffle --temperature 1.0 --seed 0 \
    --save-result --result-filename adaptive_on_c${c}.json
done
```

`--disable-shuffle` plus a fixed prompt set: every arm gets identical prompts in identical order. `output_throughput` from the result JSON is the tok/s on Figure 3. `--speed-bench-output-len` is a **cap**, not a target — requests stop at EOS, so realized average is well under 2048. At **c=1**, `n=240` prompts; otherwise **880**.

## Acknowledgments

Lucas Wilkinson (Red Hat) and Benjamin Chislett (NVIDIA). Thanks to the [DSpark](https://arxiv.org/abs/2607.05147) authors for the drafting algorithm and the confidence head, and to DeepSeek for the DeepSeek-V4 checkpoints.
