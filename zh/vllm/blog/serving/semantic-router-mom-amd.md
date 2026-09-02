---
source: https://vllm.ai/blog/2026-01-23-mom-on-amd-gpu
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# AMD 上的 MoM 现场：六只模型、十一条决策

英文对照：`en/vllm/blog/serving/semantic-router-mom-amd.md`  
原文：https://vllm.ai/blog/2026-01-23-mom-on-amd-gpu  
Playground：https://play.vllm-semantic-router.com  
MI300X / MI355X。

池子：Qwen3-235B、DeepSeek-V3.2、Kimi-K2-Thinking、GLM-4.7、gpt-oss-120b/20b。优先级 200 的 jailbreak 关键词先拦；中文深思走 Qwen；代码+深思走 DeepSeek；英文深思走 Kimi；快问走 20b。信号延迟他们写 keyword/language <1ms，embedding/domain 50–100ms。部署：`pip install vllm-sr`，`vllm-sr init`，ROCm 镜像 `vllm/vllm-openai-rocm:v0.14.0`，`VLLM_ROCM_USE_AITER=1`，`vllm-sr serve --platform=amd`。这是 **请求级编排**，不是 MoE 专家门。

本地图（原文版权仍归原站；学习对照用）：

![mom 1](../../../../assets/vllm/blog/serving/semantic-router-mom-amd/01-mom-1.png)

![mom 0](../../../../assets/vllm/blog/serving/semantic-router-mom-amd/02-mom-0.png)

![mom 2](../../../../assets/vllm/blog/serving/semantic-router-mom-amd/03-mom-2.png)

![mom 4](../../../../assets/vllm/blog/serving/semantic-router-mom-amd/04-mom-4.png)

![mom 3](../../../../assets/vllm/blog/serving/semantic-router-mom-amd/05-mom-3.png)

![mom 7](../../../../assets/vllm/blog/serving/semantic-router-mom-amd/06-mom-7.png)

![mom 5](../../../../assets/vllm/blog/serving/semantic-router-mom-amd/07-mom-5.png)

![mom 6](../../../../assets/vllm/blog/serving/semantic-router-mom-amd/08-mom-6.png)
