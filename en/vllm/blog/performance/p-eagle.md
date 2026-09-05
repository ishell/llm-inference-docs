---
source: https://vllm.ai/blog/2026-03-13-p-eagle
lang: en
fetched: 2026-09-05
---

# P-EAGLE: Faster LLM inference with Parallel Speculative Decoding in vLLM

Chinese: [zh/vllm/blog/performance/p-eagle.md](../../../../zh/vllm/blog/performance/p-eagle.md)

2026-03-13. **Amazon and NVIDIA Team**. Also on [AWS Blogs](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/). Study note. In vLLM from [v0.16.0](https://github.com/vllm-project/vllm/releases/tag/v0.16.0) (PR [#32887](https://github.com/vllm-project/vllm/pull/32887)). Flag: `"parallel_drafting": true`. Numbers below are **one NVIDIA B200**, GPT-OSS-20B, unless the sentence names another model.

[EAGLE](https://arxiv.org/pdf/2503.01840) is SOTA speculative decoding, but autoregressive drafting is a hidden ceiling: more speculated tokens means more sequential draft forwards. **P-EAGLE** emits all K draft tokens in **one** forward. The page's headline: up to **1.69×** over vanilla EAGLE-3 on real workloads on B200.

Read with [spec-decode.md](spec-decode.md) (accept math) and [parallel drafting](parallel-drafting.md) (P-EAGLE / DFlash / DSpark together). Training still needs verifier hidden; that export path is [extract-hidden-states.md](../architecture/extract-hidden-states.md).

Artifacts named on the page:

- Paper: [arXiv 2602.01469](https://www.arxiv.org/pdf/2602.01469)
- HuggingFace: [GPT-OSS 120B](https://huggingface.co/amazon/gpt-oss-120b-p-eagle), [GPT-OSS 20B](https://huggingface.co/amazon/GPT-OSS-20B-P-EAGLE), [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/amazon/Qwen3-Coder-30B-A3B-Instruct-P-EAGLE)
- vLLM: [Unified Parallel Drafting PR #32887](https://github.com/vllm-project/vllm/pull/32887)
- Speculators: [RFC](https://github.com/vllm-project/speculators/issues/292), [PR](https://github.com/vllm-project/speculators/pull/343)

![fig1 speedbench overview](../../../../assets/vllm/blog/performance/p-eagle/01-fig1_speedbench_overview.png)

**Figure 1.** P-EAGLE vs other methods on SPEED-BENCH, concurrency **1**, one B200.

## Quick start P-EAGLE

One field on `SpeculativeConfig` (this is a code comment in the original, not a heading):

```python
# vllm/config/speculative.py
   parallel_drafting: bool = True
```

Serve with a parallel-capable drafter head:

```bash
vllm serve openai/gpt-oss-20b \
   --speculative-config '{"method": "eagle3", "model": "amazon/gpt-oss-20b-p-eagle", "num_speculative_tokens": 5, "parallel_drafting": true}'
```

Pre-trained heads on HuggingFace for GPT-OSS 120B, GPT-OSS 20B, and Qwen3-Coder 30B. Download (or train) a parallel-capable head, then set `"parallel_drafting": true`.

## EAGLE's drafting bottleneck

EAGLE is **2–3×** over standard autoregressive decoding and shows up in vLLM, SGLang, and TensorRT-LLM. It still drafts **autoregressively**: K draft tokens need **K** draft forwards. As drafters get better at long outputs, that cost scales linearly with speculation depth and caps how aggressively you can speculate.

## Approach: Parallel-EAGLE (P-EAGLE)

P-EAGLE turns EAGLE from autoregressive to parallel draft generation. On B200, vs vanilla EAGLE-3 on GPT-OSS 20B: **1.05×–1.69×** across MT-Bench, HumanEval, and SpeedBench. Integrated in vLLM.

All K draft tokens come from **one** forward. Two steps (Figure 2).

**Step 1: Prefilling.** The target processes the prompt and emits a new token, as in normal inference. P-EAGLE captures hidden: `h_prompt` at each prompt position, `h_context` for the newly generated token. Same as autoregressive EAGLE.

**Step 2: P-EAGLE drafter.** Each position's input is a token embedding concatenated with a hidden state.

- **Prompt positions:** pair `emb(p)` with the matching `h_prompt`. Same shift as autoregressive EAGLE: position *i* gets token and hidden from *i−1*, so it predicts token *i*.
- **Position 1, Next-Token-Prediction (NTP):** pair the newly generated token `emb(new)` with `h_context`. Same as standard autoregressive EAGLE.
- **Positions 2 through K, Multi-Token-Prediction (MTP):** the needed token embedding and hidden do not exist yet. Fill with two **learned** parameters: a shared mask embedding `emb(mask)` and a shared hidden `h_shared`. Neutral placeholders, trained.

All positions go through **N** transformer layers, then the LM head, and predict `t1, t2, t3, t4` in one pass.

![fig2 architecture](../../../../assets/vllm/blog/performance/p-eagle/02-fig2_architecture.png)

**Figure 2.** P-EAGLE architecture: Prefill as EAGLE, then one parallel drafter pass with mask / shared-hidden placeholders on MTP slots.

## Training P-EAGLE on long sequences

Reasoning models emit long outputs. Figure 3: GPT-OSS 120B on UltraChat (prompt + generation), reasoning level **Medium** — median **3,891** tokens, P90 **10,800**. Draft models need matching context length at train time.

![fig3 sequence length](../../../../assets/vllm/blog/performance/p-eagle/03-fig3_sequence_length.png)

**Figure 3.** Sequence-length distribution (prompt + generation) on UltraChat with GPT-OSS 120B.

Parallel drafting **amplifies** train memory. K parallel groups on a sequence of length N → **N × K** positions. With **N = 8,192** and **K = 8**, one example has **65,536** positions. Attention is each position to every valid position: **65K × 65K** is over **4 billion** elements, **8 GB** in bf16.

Position sampling ([An et al., 2025](https://arxiv.org/pdf/2504.18583)) skips positions at random and saves memory, but skipping too hard hurts draft quality. Gradient accumulation splits **across examples**; when **one** sequence does not fit, there is nothing to split.

P-EAGLE's answer in this post: a **sequence partition** algorithm for intra-sequence splitting. Contiguous chunks of the N × K sequence, attention dependencies kept across chunk boundaries, gradients accumulated across chunks of the **same** sequence. Details in the [paper](https://arxiv.org/pdf/2602.01469).

## Implementation in vLLM

### Parallel drafting challenges

In many speculative setups, draft and verify share a per-request token layout. EAGLE is close: the drafter window already matches what the verifier checks — K drafted tokens plus one extra sampled token.

Parallel drafting breaks that. Predicting K tokens in one drafter forward means appending MASK placeholders (e.g. `[token, MASK, MASK, …]`). Those extra slots exist **only** for drafting, so draft batch shape ≠ verification batch shape. Verification metadata cannot be reused. Rebuild: expand input token IDs, hidden states, and positions for mask slots; increment positions per request; recompute slot mapping and per-request start indices from the new positions.

### The Triton kernel

To keep that rebuild cheap, a **fused Triton kernel** populates the drafter input batch on-GPU from the target-model batch. In one pass it:

- copies previous token IDs and positions from the target batch into new destination slots
- inserts the per-request **bonus token** sampled by the target
- fills extra parallel-drafting slots with a special MASK token ID
- emits lightweight metadata: rejected-token mask, masked-token mask for parallel slots, new-token indices for sampling draft tokens, hidden-state mapping

Otherwise this is many GPU ops (copy/scatter + insert + fill + mask + remap). One kernel cuts launch overhead and extra memory traffic.

### Hidden-state management

EAGLE-style methods that pass hidden to the draft populate those fields separately. Hidden is much larger than the rest of the batch, so the work splits: the Triton kernel outputs a **mapping**; a dedicated copy kernel broadcasts the learned hidden placeholder into mask slots.

```python
# Copy target hidden states to their new positions
self.hidden_states[out_hidden_state_mapping] = target_hidden_states

# Fill masked positions with the learned Parallel Drafting hidden state
mask = self.is_masked_token_mask[:total_num_output_tokens]
torch.where(
    mask.unsqueeze(1),
    self.parallel_drafting_hidden_state_tensor,
    self.hidden_states[:total_num_output_tokens],
    out=self.hidden_states[:total_num_output_tokens],
)
```

`parallel_drafting_hidden_state_tensor` comes from the model's `mask_hidden` buffer: a learned representation that those positions should predict future tokens.

KV cache slot mapping: valid tokens get normal slots; rejected tokens map to `PADDING_SLOT_ID` (**-1**) so they do not write spurious cache. CUDA graphs: capture range grows by **K × max_num_seqs** for the larger draft batch.

## vLLM benchmarking on P-EAGLE

Train P-EAGLE on GPT-OSS-20B. Three benches: [MT-Bench](https://arxiv.org/abs/2402.14762) (multi-turn instruction), [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) Code (long-term code generation), [HumanEval](https://github.com/openai/human-eval) (function-level synthesis). Versus the public [vanilla EAGLE-3 checkpoint](https://huggingface.co/RedHatAI/gpt-oss-20b-speculator.eagle3): **55–69%** higher throughput at low concurrency (**c=1**), **5–25%** still at high concurrency (**c=64**). Figures 4–6.

Drafter: lightweight **4-layer** model, trained to predict up to **10** tokens in parallel. Sweep speculation depths **K ∈ {3, 5, 7}** and concurrency **C ∈ {1, 2, 4, 8, 16, 32, 64}**. Goal: right deployment config for both P-EAGLE and vanilla EAGLE-3. **Linear drafting** for both. “Best P-EAGLE” / “best EAGLE-3” = the K that peaks **TPS** under those serving conditions.

Pattern on the page: P-EAGLE peaks at **K=7** at **all** concurrency levels. Vanilla EAGLE-3 peaks at **K=3**, sometimes shifting deeper with concurrency. Parallel drafting pays for depth in one forward; autoregressive drafters pay per extra token.

Hardware and serve flags: **one NVIDIA B200 (Blackwell)**.

```bash
VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 \
vllm serve openai/gpt-oss-20b \
    --speculative-config '{
      "method": "eagle3",
      "model": "amazon/GPT-OSS-20B-P-EAGLE",
      "num_speculative_tokens": 7,
      "parallel_drafting": true}' \
    --port 8000 \
    --max-num-seqs 1024 \
    --max-model-len 100000 \
    --max-num-batched-tokens 100000 \
    --max-cudagraph-capture-size 4096 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --kv-cache-dtype fp8 \
    --async-scheduling \
    --stream-interval 20
```

**Note from the page.** Serving GPT-OSS-20B with EAGLE drafters then needed a one-line vLLM patch ([PR #36684](https://github.com/vllm-project/vllm/pull/36684)). Apply before launch. Expected in an upcoming release at the time of the post.

![fig4 mtbench](../../../../assets/vllm/blog/performance/p-eagle/04-fig4_mtbench.png)

**Figure 4.** MT-Bench TPS, P-EAGLE vs EAGLE-3, GPT-OSS-20B. P/E speedup: **1.55×** (c=1), **1.29×** (c=2), **1.35×** (c=4), **1.28×** (c=8), **1.27×** (c=16), **1.09×** (c=32), **1.05×** (c=64).

![fig5 humaneval](../../../../assets/vllm/blog/performance/p-eagle/05-fig5_humaneval.png)

**Figure 5.** HumanEval TPS. P/E: **1.55×** (c=1), **1.53×** (c=2), **1.45×** (c=4), **1.35×** (c=8), **1.31×** (c=16), **1.37×** (c=32), **1.23×** (c=64).

![fig6 speedbench](../../../../assets/vllm/blog/performance/p-eagle/06-fig6_speedbench.png)

**Figure 6.** SPEED-Bench TPS. P/E: **1.69×** (c=1), **1.61×** (c=2), **1.54×** (c=4), **1.45×** (c=8), **1.40×** (c=16), **1.22×** (c=32), **1.25×** (c=64).

Throughput also tracks **acceptance length (AL)**: average draft tokens accepted per speculation round. Higher AL → more draft work becomes real output → higher effective OTPS/TPS.

**P-EAGLE (AL):**

| Config | HumanEval | SPEED-Bench | MT-Bench |
| --- | ---: | ---: | ---: |
| K=3 | 3.02 | 2.87 | 2.87 |
| K=7 | 3.94 | 3.38 | 3.70 |

**EAGLE-3 (AL):**

| Config | HumanEval | SPEED-Bench | MT-Bench |
| --- | ---: | ---: | ---: |
| K=3 | 2.65 | 2.24 | 2.70 |
| K=7 | 3.03 | 2.59 | 3.27 |

At the same K, P-EAGLE's AL is higher. At **K=7**: **+30%** on HumanEval (3.94 vs 3.03), **+31%** on SPEED-Bench (3.38 vs 2.59), **+13%** on MT-Bench (3.70 vs 3.27). Deeper speculation helps P-EAGLE more: K=3 → K=7, HumanEval AL **+0.92** (3.02 → 3.94) vs EAGLE-3 **+0.38** (2.65 → 3.03). One-pass drafting does not add sequential cost at larger K.

The post does **not** print absolute TPS in the tables — only AL, the P/E ratios in the figure captions, and the 55–69% / 5–25% summary.

## Reproducing the results

After the server is up, `vllm bench serve`:

```bash
# MT-Bench
export MODEL="openai/gpt-oss-20b"
export BASE_URL="http://localhost:8000"
vllm bench serve \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --num-prompts 80 \
    --max-concurrency 1 \
    --model $MODEL \
    --base-url $BASE_URL \
    --temperature 0.0 \
    --hf-output-len 2048

# HumanEval: download openai/openai_humaneval first
vllm bench serve \
    --dataset-name custom \
    --dataset-path <dataset path> \
    --num-prompts 164 \
    --max-concurrency 1 \
    --model $MODEL \
    --base-url $BASE_URL \
    --temperature 0.0 \
    --custom-output-len 2048
```

The original does not paste a SPEED-Bench `vllm bench serve` command in this section.

## Conclusion

P-EAGLE removes the sequential draft bottleneck: up to **1.69×** over vanilla EAGLE-3 on the workloads in the post. Draft count is decoupled from forward-pass count, so larger drafting architectures become interesting — even higher accept rates than single-layer baselines. The vLLM path handles input prep, attention metadata, and KV slot mapping with fused kernels. It needs specially trained models; the page still calls that a worthwhile add to vLLM speculative decoding.

As more parallel-trained heads show up, the authors expect this to be the preferred production choice. Try: download a pre-trained P-EAGLE head, set `"parallel_drafting": true` for a supported model.

## Acknowledgement

**AWS:** Xin Huang, Florian Saupe, Jaime Campos Salas, Ashish Khetan, George Karypis.

**NVIDIA:** Benjamin Chislett, Max Xu, Zeyuan (Faradawn) Yang, Kaihang Jiang, Xin Li, Omri Almog.

vLLM maintainers and community: reviews, guidance, and the infrastructure this landed on.
