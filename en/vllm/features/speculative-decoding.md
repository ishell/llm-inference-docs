---
source: https://docs.vllm.ai/en/stable/features/speculative_decoding/
lang: en
fetched: 2026-09-04
---

# Speculative decoding (feature page)

Chinese: [zh/vllm/features/speculative-decoding.md](../../../zh/vllm/features/speculative-decoding.md)  
Mechanism + 2024 numbers: [blog spec-decode](../blog/performance/spec-decode.md). Later CATALOG: P-EAGLE, DSpark, EAGLE 3.1, AMD spec-decode. Train drafts: [vllm-project/speculators](https://github.com/vllm-project/speculators). Structured JSON / tool args are another room: [struct-decode](../blog/performance/struct-decode.md).

Use spec decode to cut **ITL** under **medium-to-low QPS**, memory-bound work. Model-based methods (EAGLE, MTP, draft models, PARD, MLP) cut latency hardest; n-gram / suffix give modest speedups without extra peak-traffic load.

## Method selection (qualitative)

Real gains depend on family, traffic, hardware, sampling. Starting point on the page:

| Method | Low QPS (latency) | High QPS (throughput) | Notes |
|---|---|---|---|
| EAGLE | High | Medium–high | Strong general-purpose model-based |
| MTP | High | Medium–high | Best when the target has native MTP |
| Draft model | High | Medium | Needs a separate draft |
| Parallel Draft Model | High | Medium–high | Low draft latency |
| MLP speculator | Medium–high | Medium | When compatible MLP speculators exist |
| N-gram | Low–medium | Medium | Lightweight |
| Suffix decoding | Low–medium | Medium | No extra draft; dynamic depth |
| Custom Proposer | Varies | Varies | Bring your class (experimental) |
| Dynamic Speculative Decoding | High | Higher than base SD | RL / fluctuating QPS |
| Adaptive Verification | High | Higher than base SD | Per-request from drafter confidence; **DSpark only** today |

Measure with `examples/features/speculative_decoding/spec_decode_offline.py` or [benchmark CLI](../benchmarking/cli.md).

## Custom proposer (experimental)

`method = "custom_class"`, `model = "your_module.YourCustomProposerClass"`. Class takes `VllmConfig`, implements `propose`.

## `--speculative-config` schema

CLI JSON; Python `LLM(..., speculative_config={...})`. Not an exhaustive schema — generated engine args + `vllm.config.SpeculativeConfig`. YAML configs use a nested mapping, not an escaped JSON string. **`tensor_parallel_size` is not valid here** — use `draft_tensor_parallel_size`. `temperature` / `top_p` are sampling params, not this object. Internal `target_model_config` / `draft_*_config` are filled by vLLM.

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "draft_model",
    "model": "<draft-model>",
    "num_speculative_tokens": 5
  }'
```

### Common keys

| Key | Type | Default | Meaning |
|---|---|---|---|
| `method` | string | `None` | `draft_model`, `ngram`, `suffix`, `mtp`, `eagle3`, `dflash`, … Often inferred |
| `model` | string | `None` | Draft / EAGLE head / auxiliary. Often omit for `ngram`, `ngram_gpu`, `suffix`, `mtp` |
| `num_speculative_tokens` | int > 0 | `None` | Proposals per step; required if not inferred |
| `draft_tensor_parallel_size` | int ≥ 1 | `None` | TP for the draft |
| `max_model_len` | int ≥ 1 | `None` | Draft context |
| `parallel_drafting` | bool | `false` | EAGLE and draft-model only |
| `rejection_sample_method` | string | `standard` | `standard` / `synthetic` / `block` |
| `synthetic_acceptance_rates` | list[float] | `None` | Per-position rates in `[0,1]`; length = `num_speculative_tokens`; non-increasing |
| `synthetic_acceptance_length` | float | `None` | Target mean accept length in `[1, num_speculative_tokens+1]`; exclusive vs rates |
| `use_heterogeneous_vocab` | bool | `false` | Token-level intersection; **`draft_model` only**. Probabilistic draft sampling not yet supported with this on |

Gemma 4 assistant checkpoints are **MTP speculators**, not generic drafts: `"method": "mtp"` + assistant checkpoint in `model`. If logs show `SpeculativeConfig(method='draft_model', ...)` for that checkpoint, upgrade rather than force draft-model.

### N-gram

`prompt_lookup_max` / `prompt_lookup_min`: default **5** if both omitted; otherwise the omitted one mirrors the other.

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "ngram",
    "num_speculative_tokens": 4,
    "prompt_lookup_min": 2,
    "prompt_lookup_max": 5
  }'
```

### Suffix decoding

| Key | Default | Meaning |
|---|---|---|
| `suffix_decoding_max_tree_depth` | 24 | Combined prefix-match + speculation tree depth |
| `suffix_decoding_max_cached_requests` | 10000 | Global suffix-tree cache; `0` disables |
| `suffix_decoding_max_spec_factor` | 1.0 | Caps speculative length as a multiple of prefix-match length |
| `suffix_decoding_min_token_prob` | 0.1 | Min estimated token prob to speculate |

### Cross-vocabulary drafts (TLI)

Default: same vocab. `use_heterogeneous_vocab: true` builds a token-string intersection at init, constrains draft logits to shared tokens, translates IDs before rejection sampling. Page example: Qwen3-8B target + SmolLM2-135M-Instruct draft, 3 speculative tokens, `gpu_memory_utilization=0.5`. Currently **greedy draft sampling only**.

## Lossless guarantees

- **Theoretical:** sampling is lossless up to hardware numerics ([speculative sampling paper](https://arxiv.org/abs/2302.01318) family). FP error can nudge distributions.
- **Algorithmic:** rejection-sampler convergence tests; greedy-with-SD equals greedy-without (`tests/spec_decode/e2e`).
- **vLLM logprobs are not stable** across runs — see the FAQ “Can the output of a prompt vary across runs in vLLM?”
- Batch size / numerical stability can still move logprobs. Mitigation is that FAQ, not a second flag on this page.

## Known incompatibility

1. Pipeline parallelism **not** composable with speculative decoding as of `vllm<=0.15.0`.
2. Draft-model spec decode **not** supported in `vllm<=0.10.0`.

Contributor pointers on the page: Office Hours #40, Hacker’s Guide, lookahead scheduling, batch expansion, dynamic speculative decoding.
