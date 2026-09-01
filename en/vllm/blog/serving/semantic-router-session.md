---
source: https://vllm.ai/blog/2026-06-02-session-aware-agentic-routing
lang: en
fetched: 2026-09-01
---

# SAAR: agents ask whether a switch is legal

Chinese: `../../zh/vllm/blog/serving/semantic-router-session.md`  
2026-06-02. Ships in [Themis](semantic-router-themis.md). Figures on the original page. Numbers are policy-matrix + ROCm demos.

A prompt router may send a tool result to a cheaper model and break the loop. SAAR keeps signal → decision → selection, then wraps it. Send `x-session-id`.

Router memory (not chat memory). **Hard locks** on tool loops and non-portable provider state. Idle / decision-drift **resets**. Prefix-cache checkout prices warm frontier sessions. Replay explains stay / switch / locked stay. `algorithm.type: session_aware`. Policy, not endpoint LB ([Router](router.md)).

Demo: 21,600 turns, switches **−79.29%**, 3,836 unsafe → 0, estimated cost **−78.71%**. Sticky is cheaper but quality δ **−0.1433** vs SAAR **−0.0453**. Live ROCm: 2,896 req, 0 continuity violations; p95 overhead **6.181 / 26.805 / 283.463 ms** (idle includes wall sleeps). Fault matrices recovered 32/32 and 24/24 sessions. Task traces 18/18; 96/96 replay headers.
