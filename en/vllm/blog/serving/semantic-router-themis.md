---
source: https://vllm.ai/blog/2026-06-05-v0.3-vllm-sr-themis-release
lang: en
fetched: 2026-09-01
---

# Themis v0.3: ask why this route fired

Chinese: `../../zh/vllm/blog/serving/semantic-router-themis.md`  
~350+ commits since v0.2.

One contract: signals → **projections** → decisions → algorithms → models. CLI, dashboard, DSL, Helm should speak it. Operators can answer: which signals, which decision, which selector, whether safety/replay mutated the path, which config version. Stateful, replayable, protocol-aligned, session-continuous. Athena’s ambition stays; the runtime gets harder edges. Still not the P/D [Router](router.md).
