---
source: https://vllm.ai/blog/2026-03-30-extract-hidden-states
lang: en
fetched: 2026-09-01
---

# Extracting hidden states from vLLM

Chinese: [zh/vllm/blog/architecture/extract-hidden-states.md](../../../../zh/vllm/blog/architecture/extract-hidden-states.md)  
PR#33736, `vllm>=0.18.0`.

EAGLE-3 / P-EAGLE / DFlash drafts eat verifier internals, not surface text. Training them needs lots of hidden states. The old options were ugly: generate with `transformers` (lose vLLM’s distributed path; risk mismatch with production hidden) or patch vLLM (often disable prefix cache / async / auto-batch).


Local figures (copyright remains with the original site; study copies):

![design diagram](../../../../assets/vllm/blog/architecture/extract-hidden-states/01-design_diagram.png)

## Design: fake speculation

vLLM already pipes verifier hidden into Eagle-3 drafts. KV Connector already writes KV to disk / NIXL / shm. Hidden aligns with tokens the same way KV does. So:

1. A **dummy draft** receives hidden on the Eagle-3 path.
2. Its “attention” does not attend — it **inserts hidden into its own KV cache**.
3. A custom KV Connector exports that dummy KV (actually hidden).

VRAM is the paged KV allocator: chunked prefill, preemption, prefix cache still work. No hot-path tax if you leave it off. 8k tokens × 4 layers × 4096 × FP16 ≈ 268 MB — not an HTTP body.

## How to turn it on

Both configs together:

```bash
vllm serve Qwen/Qwen3-8B --speculative_config '{
  "method": "extract_hidden_states",
  "num_speculative_tokens": 1,
  "draft_model_config": {
    "hf_config": { "eagle_aux_hidden_state_layer_ids": [3, 18, 33, 36] }
  }
}' --kv_transfer_config '{
  "kv_connector": "ExampleHiddenStatesConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": { "shared_storage_path": "/tmp/hidden_states" }
}'
```

Responses include `kv_transfer_params.hidden_states_path` → safetensors: `token_ids [prompt_seq_len]`, `hidden_states [prompt_seq_len, num_layers, hidden_size]`. Only the **prompt** is saved; use `v1/completions` with `max_tokens=1`. Single-node TP/DP works. `ExampleHiddenStatesConnector` writes disk and blocks; async and device-to-device still incoming. Speculators ≥0.5.0 uses this for online training.
