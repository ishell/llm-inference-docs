---
source: https://vllm.ai/blog/2025-05-12-hardware-plugin
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Hardware Plugin：把卡从主干里请出去

英文对照：[en/vllm/blog/architecture/hardware-plugin.md](../../../../en/vllm/blog/architecture/hardware-plugin.md)  
原文：https://vllm.ai/blog/2025-05-12-hardware-plugin  
2025-05-12。署名 **The Ascend Team on vLLM**。原文没有机制图（只有 logo，本地不搬）。Hardware Pluggable RFC：[issue #11162](https://github.com/vllm-project/vllm/issues/11162)，自 **2024 年 12 月** 起和昇腾一起做。落地仓库：[vllm-ascend](https://github.com/vllm-project/vllm-ascend)、[vllm-spyre](https://github.com/vllm-project/vllm-spyre)。更宽的插件故事是半年后的 [plugin-system](plugin-system.md)。

RFC 要办的事：硬件用**解耦**的方式进 vLLM——快、按模块长，而不是在主干上打补丁。

## 为什么要插件

当时 vLLM 已经接了多家后端。后端一多，三件事同时坏：

- **代码变复杂。** 每家后端自己写一份 `Executor`、`Worker`、`Runner`、`Attention`。非通用代码散落整棵树。
- **维护贵。** 后端作者要付，社区也要付。后端维护的人一不在，社区那点人手加不进新功能。
- **谈不上扩展。** 分层（Executor / Worker / Runner / Attention）看起来干净，可**新卡**仍要侵入式改、打补丁，而不是动态注册。加一家后端像开刀。

他们要的解法：

- **代码解耦。** 后端住在自己的包里，主干变干净。
- **维护变轻。** 主干的人盯通用功能，不必被每家卡的脾气淹没。
- **接入更快、更独立。** 新后端少改主干，可以按自己的钟走。

## 它是什么（底下两份 RFC）

1. [[RFC] vLLM Plugin System](https://github.com/vllm-project/vllm/issues/7131)——自定义模型、executor、scheduler 等。2025 年 11 月那篇博客是这扇门后来的写法。
2. [[RFC] Make vLLM Device-Agnostic](https://github.com/vllm-project/vllm/issues/9268) 和 [PR #6080](https://github.com/vllm-project/vllm/pull/6080)——**`platform` 子模块**：硬件相关实现收拢一处，主干少写 `if device`，模块化的地基。

在这两份之上，[[RFC] Hardware Pluggable](https://github.com/vllm-project/vllm/issues/11162) 把 **`Platform` 自己做成插件**，并重构 `Executor`、`Worker`、`ModelRunner`、`AttentionBackend`、`Communicator`，好让插件把这些格子填上。

写这篇时，社区已经把 RFC 里的 Platform 模块落地，用上面两个仓库在 **Ascend NPU** 和 **IBM Spyre** 上核过。

## 怎样用插件接一家新后端

### 开发者

**第一步：新项目 + `Platform`。** 一个 Python 包，加 `platform.py`。从 `vllm.platforms` 引入 `Platform`，实现它要的属性和方法。他们点名的例子：[`vllm_ascend/platform.py`](https://github.com/vllm-project/vllm-ascend/blob/72a43a61d8d2193dddbfcc60578fd642008225a5/vllm_ascend/platform.py#L52)（commit `72a43a61`）。

**第二步：按需实现四块底座：**

```python
from vllm.worker.worker_base import WorkerBase
from vllm.worker.model_runner_base import ModelRunnerBase
from vllm.attention.backends.abstract import AttentionBackend
from vllm.distributed.device_communicators.base_communicator import CommunicatorBase
```

每一类在 vLLM 里都有对应的基类。整包例子看 [vllm-ascend](https://github.com/vllm-project/vllm-ascend/tree/main/vllm_ascend)。类名是当时的；后来 V1 可能挪过文件。

**第三步：在 `setup.py` 里用 Python entry point 注册：**

```python
setup(
    entry_points={
        "vllm.platform_plugins": [
            "{your_platform_name} = {code_path}:{register_function}"
        ]
    }
)
```

- `{your_platform_name}`：后端名字，可以随便起。
- `{code_path}`：主 Python 模块路径。
- `{register_function}`：注册函数，**返回第一步那个 `Platform` 类的路径**。

实践例子：昇腾的 [`setup.py`](https://github.com/vllm-project/vllm-ascend/blob/72a43a61d8d2193dddbfcc60578fd642008225a5/setup.py#L102)。

### 用户

先装 vanilla vLLM，再装插件：

```bash
pip install vllm vllm-ascend
```

启动日志里出现这些，就说明插件活了（原文时间戳 `02-06 15:49:01`）：

```
INFO 02-06 15:49:01 __init__.py:30] Available plugins for group vllm.platform_plugins:
INFO 02-06 15:49:01 __init__.py:32] name=ascend, value=vllm_ascend:register
…
INFO 02-06 15:49:01 __init__.py:44] plugin ascend loaded.
INFO 02-06 15:49:01 __init__.py:181] Platform plugin ascend is activated
```

## 下一步（文中当时）

四件还要继续的事：

1. 继续补 **V1** 和 **VLM**。
2. 把插件面扩到更多模块：**scheduler**、**graph mode**、**custom operators**。
3. 更好用、更快。
4. 把插件架构养稳，只给真正需要它的硬件平台。

试用；问题去 [vLLM Slack](https://slack.vllm.ai) 的 **`#sig-extensible-hardware`**。

## 致谢

vLLM 维护者（重构、讨论、快审）：[Kaichao You](https://github.com/youkaichao)、[Simon Mo](https://github.com/simon-mo)、[Cyrus Leung](https://github.com/DarkLight1337)、[Robert Shaw](https://github.com/robertgshaw2-redhat)、[Michael Goin](https://github.com/mgoin)、[Jie Li](https://github.com/jeejeelee)。

昇腾侧（机制设计与实现）：[Xiyuan Wang](https://github.com/wangxiyuan)、[Shanshan Shen](https://github.com/shen-shanshan)、[Chenguang Li](https://github.com/noemotiovon)、[Mengqing Cao](https://github.com/MengqingCao)。

Spyre 侧（可插拔 **scheduler**）：[Joe Runde](https://github.com/joerunde)、[Yannick Schnider](https://github.com/yannicks1)。

另外：[yancong](https://github.com/ice-tong)（可扩展量化）、[Aviv Keshet](https://github.com/akeshet)（可扩展 `SamplingParams`）。

[torch.compile](torch-compile.md) 把优化从模型文件拿走；这篇把硬件从主干拿走。两件事都是为了让「又来一家卡」不必开一刀主干。
