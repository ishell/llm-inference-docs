---
source: https://docs.vllm.ai/en/stable/usage/v1_guide/
lang: en
fetched: 2026-08-31
---

# vLLM V1 Guide (notes)

V1 is the current engine. Unified scheduler treats prompt and output tokens with a per-request token budget `{request_id: num_tokens}` — chunked prefill, prefix cache, and spec decode share that budget.

- Chunked prefill **on by default** when possible (V0 was conditional).
- Decode is prioritized; leftover `max_num_batched_tokens` goes to prefill (chunked if needed).
- Prefix cache: near-zero overhead vs V0.
- Policies: FCFS or priority (`--scheduling-policy`).
- Default preemption: `RECOMPUTE` not `SWAP`.

Removed vs old V0: `best_of`, per-request logits processors, GPU↔CPU KV swap, request-level structured-output backend.

See `optimization/optimization.md` for tuning order (CPU cores → `-O*` → `max_num_batched_tokens` → parallelism).
