---
source: https://vllm.ai/blog/2026-01-23-mom-on-amd-gpu
lang: en
fetched: 2026-09-01
---

# Live MoM on AMD: six models, eleven decisions

Chinese: `../../zh/vllm/blog/serving/semantic-router-mom-amd.md`  
Playground: https://play.vllm-semantic-router.com  
MI300X / MI355X.

Pool: Qwen3-235B, DeepSeek-V3.2, Kimi-K2-Thinking, GLM-4.7, gpt-oss-120b/20b. Priority-200 jailbreak keyword blocks first; Chinese deep-think → Qwen; code+deep-think → DeepSeek; English deep-think → Kimi; fast QA → 20b. Their signal latencies: keyword/language <1ms, embedding/domain 50–100ms. Deploy: `pip install vllm-sr`, `vllm-sr init`, `vllm/vllm-openai-rocm:v0.14.0`, `VLLM_ROCM_USE_AITER=1`, `vllm-sr serve --platform=amd`. **Request-level orchestration**, not MoE expert gating.

Local figures (copyright remains with the original site; study copies):

![mom 1](../../../../assets/vllm/blog/serving/semantic-router-mom-amd/01-mom-1.png)

![mom 0](../../../../assets/vllm/blog/serving/semantic-router-mom-amd/02-mom-0.png)

![mom 2](../../../../assets/vllm/blog/serving/semantic-router-mom-amd/03-mom-2.png)

![mom 4](../../../../assets/vllm/blog/serving/semantic-router-mom-amd/04-mom-4.png)

![mom 3](../../../../assets/vllm/blog/serving/semantic-router-mom-amd/05-mom-3.png)

![mom 7](../../../../assets/vllm/blog/serving/semantic-router-mom-amd/06-mom-7.png)

![mom 5](../../../../assets/vllm/blog/serving/semantic-router-mom-amd/07-mom-5.png)

![mom 6](../../../../assets/vllm/blog/serving/semantic-router-mom-amd/08-mom-6.png)
