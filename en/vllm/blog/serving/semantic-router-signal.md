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

Local figures (copyright remains with the original site; study copies):

![signal 0](../../../../assets/vllm/blog/serving/semantic-router-signal/01-signal-0.png)

![signal](../../../../assets/vllm/blog/serving/semantic-router-signal/02-signal.png)

![signal 1](../../../../assets/vllm/blog/serving/semantic-router-signal/03-signal-1.png)

![signal 2](../../../../assets/vllm/blog/serving/semantic-router-signal/04-signal-2.png)

![signal 3](../../../../assets/vllm/blog/serving/semantic-router-signal/05-signal-3.png)

![signal 4](../../../../assets/vllm/blog/serving/semantic-router-signal/06-signal-4.png)

![signal 5](../../../../assets/vllm/blog/serving/semantic-router-signal/07-signal-5.png)

![signal 6](../../../../assets/vllm/blog/serving/semantic-router-signal/08-signal-6.png)

![signal code 0](../../../../assets/vllm/blog/serving/semantic-router-signal/09-signal-code-0.png)

![signal code 1](../../../../assets/vllm/blog/serving/semantic-router-signal/10-signal-code-1.png)

![signal 7](../../../../assets/vllm/blog/serving/semantic-router-signal/11-signal-7.png)

![signal 8](../../../../assets/vllm/blog/serving/semantic-router-signal/12-signal-8.png)
