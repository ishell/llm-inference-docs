---
source: https://vllm.ai/blog/2026-06-10-diffusion-gemma
lang: en
fetched: 2026-09-05
---

# DiffusionGemma: The First Diffusion LLM (dLLM) Natively Supported in vLLM

Chinese: [zh/vllm/blog/architecture/diffusion-gemma.md](../../../../zh/vllm/blog/architecture/diffusion-gemma.md)  
Source: https://vllm.ai/blog/2026-06-10-diffusion-gemma

2026-06-10. **The vLLM Team and Google DeepMind Team**. Study extract, not an official reprint. Batch=1 demos, not an SLA. Recipe: [vLLM recipe](https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it). Runner hooks: [mrv2.md](mrv2.md). Spec path reused: [spec-decode.md](../performance/spec-decode.md). Gemma4 backbone family: [gemma4.md](../serving/gemma4.md). Omni / diffusion serving: [vllm-omni.md](../serving/vllm-omni.md). Local figures under `assets/vllm/blog/architecture/diffusion-gemma/`.

Fits: serving the first in-tree dLLM — 256-token canvas, bidirectional denoise, ModelState hooks, mixed prefill / denoise / commit in one batch. Does not fit: treating **1288 tok/s** as a multi-batch number; the page is H100/H200 **batch=1**.

> Deploy DiffusionGemma: [vLLM recipe](https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it).

Google’s DiffusionGemma is a **26B** discrete diffusion language model on a Gemma4 backbone — the first dLLM in vLLM. Integration needed a different decoding pattern. dLLMs do not fit the standard autoregressive serving path: bidirectional attention, iterative refinement, block generation, custom sampling at each denoise step.

They used [model runner v2](https://vllm.ai/blog/2026-03-24-mrv2) (this repo: [mrv2.md](mrv2.md)) **ModelState**: custom input prep and hooks for per-request model-specific state. Claim: Hugging Face reference accuracy, plus efficient batched serving.

AR transformers emit one token left to right. Diffusion LMs **denoise a fixed-length canvas**. Many tokens refined per forward pass — compute vs bandwidth, attractive at **low batch** (spare FLOPs, bandwidth-bound). Many tokens per pass can mean very low latency. DiffusionGemma denoises **256 tokens** at a time.

![ar vs diffusion](../../../../assets/vllm/blog/architecture/diffusion-gemma/01-ar-vs-diffusion.svg)

**Figure.** Autoregressive vs block diffusion (study copy; copyright remains with the original site).

## DiffusionGemma Architecture and Sampling Loop

Standard Gemma4 backbone, two modes, **shared weights** — one set of layers, used two ways:

- **Encoder:** *causal* attention, **writes** KV. Twice per block: prefill the prompt; **commit** a finished block.
- **Decoder:** *bidirectional* attention, **reads** KV only. Denoise — every canvas position attends to every other, so the whole block refines at once.

Causal encoder writes KV the AR way, so vLLM’s **automatic prefix cache works out of the box**: shared prompt prefixes reuse across requests with no diffusion-specific cache.

Loop for one 256-token block: prefill (encoder) → canvas = random tokens, state = denoise → each step runs decoder over the full canvas, samples every position, keeps the confident ones → when the block stops changing, encoder commit writes KV and emits 256 tokens → next block from a fresh random canvas.

![sampling loop horizontal](../../../../assets/vllm/blog/architecture/diffusion-gemma/02-sampling-loop-horizontal.svg)

**Figure.** Per-block sampling loop (study copy).

Inside a block: all 256 positions in parallel. Across blocks: still left-to-right — each new block conditions on committed tokens.

### Entropy-bound denoising

Every denoise step re-samples **all** positions; only confident ones are kept, the rest get fresh random tokens. Confidence = entropy of the predicted distribution (low entropy = decided).

**Entropy-bound:** walk most-confident → least, accept until accumulated entropy exceeds a fixed budget. Early: almost everything is unsure, a few anchors lock. Neighbors sharpen; more positions fit the budget; the block snaps into focus over a handful of steps.

![denoising grid](../../../../assets/vllm/blog/architecture/diffusion-gemma/03-denoising-grid.svg)

**Figure.** Entropy-bound denoising over several steps (study copy).

**Converged:** argmax prediction unchanged for a couple of consecutive steps **and** mean per-token entropy below a confidence threshold — or a hard step limit. Commit is that **clean argmax**, not the noisy sampled canvas carried between steps.

### Self-conditioning

To stabilize the loop and converge faster: **self-conditioning** — condition on the model’s **own previous prediction**. Not hard tokens: full softmax → probability-weighted average of token embeddings → add through a small **gated MLP** onto canvas embeddings before the next pass.

![self conditioning](../../../../assets/vllm/blog/architecture/diffusion-gemma/04-self-conditioning.svg)

**Figure.** Self-conditioning feedback (study copy).

Each step remembers last step’s belief. Renoised positions still carry that information instead of starting from scratch. **Decoder/denoise only** — encoder prefill and commit zero the feedback (plain token embeddings).

## Implementation in vLLM

### Reusing the Speculative Decoding Data Path

The engine already has a mature, stable speculative-decoding path. Inspired by [RFC #36155](https://github.com/vllm-project/vllm/issues/36155): reuse it for DiffusionGemma. Natural fit — each step’s canvas is a large set of **draft tokens, fully accepted or fully rejected**. Scheduler and runner stay almost unchanged. Exception: spec decode normally samples one extra (bonus) token; **sampling 0 tokens** was added, controlled by ModelState.

Concretely: scheduler, runner, Gemma4 backbone reused; only ModelState and sampler are diffusion-specific:

![stack](../../../../assets/vllm/blog/architecture/diffusion-gemma/05-stack.svg)

**Figure.** DiffusionGemma in vLLM’s speculative-decoding stack (study copy).

### The ModelState Interface

Without ModelState, a non-AR model on V1 would fork the runner and thread diffusion state through input prep, attention metadata, and sampling. ModelState is hooks the runner calls at each stage of the forward loop:

| Hook | DiffusionGemma uses it to… |
| --- | --- |
| `prepare_inputs()` | Embed canvas tokens and apply self-conditioning |
| `prepare_attn()` | Per-request causal (encoder) vs bidirectional (denoise) |
| `custom_sampler()` | Replace the default sampler with `DiffusionSampler` |
| `add_request()` / `remove_request()` | Init / tear down per-request state (canvas, self-conditioning probs) |

Models register via `get_model_state_cls()` on the model class. Runner stays generic: `prepare_attn(...)`, merge `prepare_inputs(...)` into forward kwargs, sample via `custom_sampler()` → `DiffusionSampler`. New block-diffusion model = one ModelState + one-line registration; no runner / scheduler / shared-infra change. Blueprint for later dLLMs.

### Putting It Together: DiffusionGemmaModelState and DiffusionSampler

`DiffusionGemmaModelState` is the ModelState for `DiffusionGemma`. Per-request diffusion-loop state: phase (commit vs denoise), current `canvas`, history for convergence checks, self-conditioning probabilities, and more. Lives in **pre-allocated GPU tensors**, updated in place. `prepare_inputs()`: embed canvas tokens; self-conditioning from last denoise softmax (internal per-request state) → probability-weighted embeddings through a gated MLP. `prepare_attn()`: causal vs bidirectional from the phase flag. Mixed prefill / denoise / commit in one batch; per-request causal flag set asynchronously on GPU → attention-kernel changes below.

`DiffusionSampler` replaces `(Sampler, RejectionSampler)`. Owns canvas init/reset on phase changes. Per-step work is one `@torch.compile`d `_compiled_sample_step`, vectorized over in-flight decode requests:

- **Prefill:** random canvas, `num_sampled = 0`
- **Denoise:** temperature-scale logits; Gumbel-max (`argmax(logits/T + gumbel_noise)`); accept confident positions up to the entropy bound; renoise the rest. Record argmax canvas; check convergence (stable argmax for N steps and mean entropy below threshold, or step cap)
- **Commit:** emit clean `argmax_canvas` (`num_sampled = 256`), reinit canvas, reset per-request state

During denoise: `num_sampled = 0`, `num_rejected = query_len` — **KV cursor does not move**. Marking every canvas position rejected tells the scheduler to reschedule the same block. Denoise loop stays inside existing spec-decode accounting **with no scheduler change**.

### Dynamic Per-sequence Causal Attention

Encoder causal, decoder bidirectional. Until this work, causality was **batch-wide**. Decoder-only = causal; Whisper encoder = bidirectional. DiffusionGemma alternates per request; vLLM mixes stages in one forward. **Per-sequence causal attention:**

- **Request 0:** prefill length 6, **encoder** causal (above-diagonal masked). Attention in tiles (example 2×2; real tiles hardware-tuned); all-masked tiles skipped (compute + HBM K/V).
- **Request 1:** prefill of length 6 done, **decoder** canvas size 4 — bidirectional inside the canvas, plus all context keys. Nothing masked, no skip.
- **Request 2:** canvas accepted — last **encoder** pass, causal, fill KV with new tokens; queries also attend to cached keys.

![per seq causal attention](../../../../assets/vllm/blog/architecture/diffusion-gemma/06-per_seq_causal_attention.svg)

**Figure.** Dynamic per-sequence causal attention (study copy).

Backends: Triton Attention (`TRITON_ATTN`) and FlashAttention 4 (`FLASH_ATTN`). Boolean `causal` becomes a **per-request tensor**. Mask updates; tiling kept.

### Sliding window attention

Some layers use sliding-window attention. On the canvas it must be **symmetric**: window `W` → attend `W` before **and** `W` after → **`2*W + 1`**.

![per seq sliding window](../../../../assets/vllm/blog/architecture/diffusion-gemma/07-per_seq_sliding_window.svg)

**Figure.** Dynamic causal sliding-window attention (study copy). Same three requests, `W=2`: Requests 0 and 2 keep one-sided causal window; Request 1’s denoise canvas uses the symmetric window (only context tokens that fall inside it).

Both backends: change the window’s **right-hand bound** for bidirectional requests. Causal stays left-only; bidirectional uses `W` on each side.

## Quantized Checkpoint Support

[LLM Compressor](https://github.com/vllm-project/llm-compressor) → [compressed-tensors](https://github.com/vllm-project/compressed-tensors). RedHatAI hub, original order:

1. [diffusiongemma-26B-A4B-it-NVFP4](https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4) — weights and activations NVFP4
2. [diffusiongemma-26B-A4B-it-FP8-dynamic](https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic) — quantized weights, fully dynamic activations

Preliminary evals with/without thinking: AIME 2025, GPQA Diamond, GSM8k on vLLM. Numbers and recovery scores on the model cards.

## Results

Architecture aimed at extremely low latency, interactive use. Batch=1, one H100 / H200, built-in `vllm bench serve`. FP8 diffusion: **1,288 generation tok/s on H200** (~**6×** standard AR, ~**3×** MTP); **1,008 tok/s on H100** (~**5×** and ~**2.6×**).

![perf](../../../../assets/vllm/blog/architecture/diffusion-gemma/08-perf.svg)

**Figure.** Generation throughput on H100 and H200 — FP8 diffusion vs AR baselines (study copy). [Repro commands](https://gist.github.com/LucasWilkinson/89185e4dc05d300df33a4ce030973911).

## Acknowledgements

Google DeepMind × vLLM.

- **Google DeepMind:** Martin Kukla, João Gante, Luciano Martins
- **vLLM:** Lucas Wilkinson, Matthew Bonanni, Nicolò Lucchesi, Dipika Sikka, Doug Smith, Edward Arthur Quarm Jnr, Alon Kellner (Red Hat), Nick Hill (Inferact)
- **NVIDIA:** Dimitrios Bariamis, Alec Kohlhoff, Porras Huang, Eugene Rakhmatulin
