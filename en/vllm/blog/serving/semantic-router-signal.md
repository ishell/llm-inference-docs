---
source: https://vllm.ai/blog/2025-11-19-signal-decision
lang: en
fetched: 2026-09-01
---

# Signal–decision: after 14 MMLU classes

Chinese: `../../zh/vllm/blog/serving/semantic-router-signal.md`  
Ships in [Iris](semantic-router-iris.md).

“Urgent review of an auth vulnerability” becomes “computer science” under 14 MMLU labels and hits a generic coder — urgency, jailbreak, reasoning budget gone. “Urgent patient data breach” may reach a medical model with no PII plugin.

New spine: extract multi-dimensional **signals** (domain / keyword / embedding / factual / feedback / preference), compose **decisions** with AND/OR + **priority**, attach a plugin chain. Highest priority wins; else default. Five built-in plugins then (cache / jailbreak / PII / hallucination / system_prompt-class), per-decision, ordered: mutate, block, or stamp metadata. 50+ enterprise cases do not fit 14 academic tags. Control plane — not the P/D [Router](router.md).
