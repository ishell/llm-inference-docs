---
source: https://vllm.ai/blog/2026-06-05-v0.3-vllm-sr-themis-release
lang: en
fetched: 2026-09-01
---

# Themis v0.3: ask why this route fired

Chinese: [zh/vllm/blog/serving/semantic-router-themis.md](../../../../zh/vllm/blog/serving/semantic-router-themis.md)  
~350+ commits since v0.2.

One contract: signals → **projections** → decisions → algorithms → models. CLI, dashboard, DSL, Helm should speak it. Operators can answer: which signals, which decision, which selector, whether safety/replay mutated the path, which config version. Stateful, replayable, protocol-aligned, session-continuous. Athena’s ambition stays; the runtime gets harder edges. Still not the P/D [Router](router.md).

Local figures (copyright remains with the original site; study copies):

![hero v2](../../../../assets/vllm/blog/serving/semantic-router-themis/01-hero-v2.png)

![release value map](../../../../assets/vllm/blog/serving/semantic-router-themis/02-release-value-map.png)

![config contract](../../../../assets/vllm/blog/serving/semantic-router-themis/03-config-contract.png)

![routing contract](../../../../assets/vllm/blog/serving/semantic-router-themis/04-routing-contract.png)

![session aware routing](../../../../assets/vllm/blog/serving/semantic-router-themis/05-session-aware-routing.png)

![projection layer](../../../../assets/vllm/blog/serving/semantic-router-themis/06-projection-layer.png)

![operator console](../../../../assets/vllm/blog/serving/semantic-router-themis/07-operator-console.png)

![long context binding](../../../../assets/vllm/blog/serving/semantic-router-themis/08-long-context-binding.png)

![hardware backend paths](../../../../assets/vllm/blog/serving/semantic-router-themis/09-hardware-backend-paths.png)

![amd validation path](../../../../assets/vllm/blog/serving/semantic-router-themis/10-amd-validation-path.png)

![routerarena leaderboard vllm sr](../../../../assets/vllm/blog/serving/semantic-router-themis/11-routerarena-leaderboard-vllm-sr.png)

![hermes roadmap](../../../../assets/vllm/blog/serving/semantic-router-themis/12-hermes-roadmap.png)
