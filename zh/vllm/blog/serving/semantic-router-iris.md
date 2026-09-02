---
source: https://vllm.ai/blog/2026-01-05-vllm-sr-iris
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Semantic Router v0.1 Iris：从 14 类到信号链

英文对照：`en/vllm/blog/serving/semantic-router-iris.md`  
原文：https://vllm.ai/blog/2026-01-05-vllm-sr-iris  
信号/决策细节见后续 signal-decision 篇。

立项时一只 ModernBERT 切 14 个 MMLU 域。Iris 改成 **信号 → 决策 → 插件**：域 / 关键词 / embedding / 事实 / 反馈 / 偏好，AND/OR 带优先级。Jailbreak、PII、semantic cache、HaluGate 变成按决策可开关的插件。分类核与 Candle 一起改成 **共享基座 + 多 LoRA**：N 次全模型前向变成 1 + N×ε。

HaluGate 三截：Sentinel（这句要不要核事实）→ Detector（哪些 token 没接地）→ Explainer（矛盾还是中性）。工具结果当 ground truth，结果走 HTTP header。

```
pip install vllm-sr
vllm-sr init
```

K8s：`helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router`。MoM 家族是路由专用小模型（域/PII/jailbreak/HaluGate/tool/embedding）。另有 `/v1/responses` 状态会话和语义滤工具。和引擎里的 [Router](router.md) 不要混。

本地图（原文版权仍归原站；学习对照用）：

![iris 0](../../../../assets/vllm/blog/serving/semantic-router-iris/01-iris-0.png)

![iris 1](../../../../assets/vllm/blog/serving/semantic-router-iris/02-iris-1.png)

![iris 2](../../../../assets/vllm/blog/serving/semantic-router-iris/03-iris-2.png)

![iris 3](../../../../assets/vllm/blog/serving/semantic-router-iris/04-iris-3.png)

![iris 4](../../../../assets/vllm/blog/serving/semantic-router-iris/05-iris-4.png)

![iris 7](../../../../assets/vllm/blog/serving/semantic-router-iris/06-iris-7.png)

![iris 6](../../../../assets/vllm/blog/serving/semantic-router-iris/07-iris-6.png)

![iris 5](../../../../assets/vllm/blog/serving/semantic-router-iris/08-iris-5.png)

![iris 8](../../../../assets/vllm/blog/serving/semantic-router-iris/09-iris-8.png)
