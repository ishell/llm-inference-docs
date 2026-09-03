---
source: https://vllm.ai/blog/2025-05-12-hardware-plugin
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Hardware Plugin：把卡从主干里请出去

英文对照：[en/vllm/blog/architecture/hardware-plugin.md](../../../../en/vllm/blog/architecture/hardware-plugin.md)  
原文：https://vllm.ai/blog/2025-05-12-hardware-plugin  
2025-05-12。Hardware Pluggable RFC，2024 年 12 月起和昇腾一起做。落地仓库：`vllm-ascend`、`vllm-spyre`。

后端一多，Executor / Worker / Runner / Attention 各写一份，非通用代码散落主干，维护的人一不在，新功能就加不进去。插件把 **Platform** 收成独立包：主干变干净，新后端少打补丁、可自己长。

底座两份 RFC：通用插件系统；device-agnostic 的 `platform` 子模块（还有 #6080）。Hardware Pluggable 把 Platform 做成插件，并重构 Executor、Worker、ModelRunner、AttentionBackend、Communicator。

## 开发

1. 新 Python 项目，`platform.py` 继承 `vllm.platforms.Platform`。
2. 按需实现 `WorkerBase`、`ModelRunnerBase`、`AttentionBackend`、`CommunicatorBase`。
3. `setup.py` 注册：

```python
entry_points={'vllm.platform_plugins': ["{name} = {module}:{register}"]}
```

`register` 返回 Platform 类路径。用户：`pip install vllm vllm-ascend`。日志里出现 `plugin ascend loaded` / `Platform plugin ascend is activated` 即生效。

下一步（当时）：V1 与 VLM、scheduler / graph / 自定义 op 的插件面、更稳的架构。Slack：`#sig-extensible-hardware`。

[torch.compile](torch-compile.md) 把优化从模型文件拿走；这篇把硬件从主干拿走。两件事都是为了让「又来一家卡」不必开一刀主干。
