---
source: https://vllm.ai/blog/2025-04-23-openrlhf-vllm
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# OpenRLHF × vLLM：生成占 RLHF 的九成

英文对照：[en/vllm/blog/serving/openrlhf.md](../../../../en/vllm/blog/serving/openrlhf.md)  
原文：https://vllm.ai/blog/2025-04-23-openrlhf-vllm  
2025-04。后来的暂停/恢复 API 见 [Native RL](native-rl.md)。

CoT 生成可以吃掉 RLHF 墙钟的 **90%**。OpenRLHF 用 Ray 把 vLLM 生成和 ZeRO-3 训练拼在一起：训练侧更新完权重，经 `ColocateWorkerExtension` **IPC 灌进** 同机 vLLM worker，不必把整份权重经 TCP 再搬一次。

这篇是 **训练框架怎么挂引擎**；Native RL 那篇是 **引擎怎么为 RL 暂停、保 KV、切权重**。两篇不要合成一篇。数字、API 名以 2025-04 原文为准，之后的 `keep` pause / DPEP / `VLLM_SERVER_DEV_MODE` 以 Native RL 为准。

本地图（原文版权仍归原站；学习对照用）：

![ray](../../../../assets/vllm/blog/serving/openrlhf/01-ray.png)
