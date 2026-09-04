---
source: https://vllm.ai/blog/2025-11-20-vllm-plugin-system
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 插件系统：改 vLLM 不必养一座 fork

英文对照：[en/vllm/blog/architecture/plugin-system.md](../../../../en/vllm/blog/architecture/plugin-system.md)  
原文：https://vllm.ai/blog/2025-11-20-vllm-plugin-system  
2025-11-20。署名 **Dhruvil Bhatt (AWS SageMaker)**。先发在 [Medium](https://medium.com/@dhruvilbhattlm10/building-clean-maintainable-vllm-modifications-using-the-plugin-system-e80df0f62861)。架构图来源页上写的是 [vllm-ascend](https://github.com/vllm-project/vllm-ascend)。硬件那扇专门的门：[hardware-plugin](hardware-plugin.md)。后来走 `vllm.general_plugins` 的亲戚：[AFD](../serving/afd.md)。同样是「留门、别 fork」：[sleep-mode](sleep-mode.md)、[KV offload](../serving/kv-offload.md) / [Mooncake](../serving/mooncake.md)（`KVConnector`）、[RDT](../serving/rdt-weight-transfer.md) / [native-rl](../serving/native-rl.md)（`WeightTransferEngine`）。

适用：自定义调度、KV 行为、硬件、执行路径上的补丁。不适合：改引擎心脏、又想跟主线每一周对齐——那种还是该上游，或接受 fork 的税。

本地图（原文版权仍归原站；学习对照用）：

![vllm plugin system arch](../../../../assets/vllm/blog/architecture/plugin-system/01-vllm-plugin-system-arch.png)

## 概览

vLLM 已经是高吞吐、低延迟的 serving 引擎：continuous batching、调度、PagedAttention、生产级 API。可团队还是会想动内部——换调度、改 KV、塞专有优化、在模型执行里打补丁。三条旧路从这里开始不舒服。

## 问题：「我要改 vLLM……然后呢？」

改动简单、又对社区有用，答案仍是 **选项 A：上游**。补丁活在开源里，接受评审，跟着 vLLM 一起长。

现实常常是：**专有**、**领域特定**、**太实验**、**不够通用** 过不了上游，或 **内部时间表** 跟开源评审对不齐。那就得另找路。

### 选项 B：自己养一座 fork

第一反应：fork，把补丁加进去。小而慢的仓库可以。**vLLM 不是那种仓库。** 发版可以近到 **两周一次**，一周 **几百个 PR**。

长命 fork 意味着：不断 rebase / merge；在变得最剧烈的区域解冲突；手工再打一遍补丁；沉重的兼容测试；内部还要围着一份 **私有 vLLM 制品** 转开发流程。不久它就变成 **全职工作**。许多团队养不起。

### 选项 C：monkey patch

一个小 Python 包，在 vanilla vLLM 上动态打补丁，看起来很美：不必 fork、不跟主干分叉、动态生效、代码量小。原文把坑写死了：

- 常常 **整类、整模块替换**，十行改动也要 **复印大段源码**——包括你根本没改的部分。
- **每次升级都碎**，因为你换的是文件，不是那十行。
- **调试难受**：bug 在补丁里？在没改过的 vanilla 里？还是 monkey patch 把调用关系拧歪了？
- 运维成本会涨：每个发版都要 **diff、再同步复印出来的文件**——fork 的税，只是藏在 Python 包里。
- 有些模块（原文点名 **`Scheduler`**）monkey patch **经常无效**：它们跑在 **`EngineCore` 的另一个进程** 里。`EngineCore` 仍调用 **旧实现**。这是进程同步问题，不是 import 顺序能糊弄过去的。

表面问题解决了，长期维护跟 fork 一样难。

## 更干净的路：插件系统

作者走的是 vLLM 正在长的 [general_plugin 架构](https://docs.vllm.ai/en/stable/design/plugin_system.html)：把针对性改动注入引擎，**不改上游源码**。宣称的好处：结构化、模块化补丁；运行时启用；外科手术式覆盖；兼容性闸门；不必整文件复印；不必 monkey patch 杂技；不必养 fork。夹在「全部上游」和「整文件替换」中间。

> **原文注：** vLLM 提供 **四种** 插件组——**platform**、**engine**、**model**、**general**。这篇只讲 **general plugin**：它在 **每一个** vLLM 进程里加载，所以适合这种干净改法。分类见 [Types of Supported Plugins](https://docs.vllm.ai/en/latest/design/plugin_system/#types-of-supported-plugins)。platform 那条是 [hardware-plugin](hardware-plugin.md) 的故事。

## 用插件搭一套扩展框架

作者做了一个小扩展包，当所有自定义改动的容器。每一块补丁：

- 只装 **真正要改的片段或类**
- 运行时可以 **开、可以关**
- 可以声明 **最低 vLLM 版本**
- 可以 **休眠**，直到某个模型配置点名要它

插件在运行时生效，于是 **同一份容器镜像** 能伺候多个模型，按模型选择性开补丁。灵感来自 [ArcticInference](https://github.com/snowflakedb/ArcticInference)。

## 实现：一个 `general_plugins` 包

用 `vllm.general_plugins` 这个 entry point。文中的包名：`vllm-custom-patches`。

### 目录

```
vllm_custom_patches/
├── setup.py
├── vllm_custom_patches/
│   ├── __init__.py
│   ├── core.py              # Base patching infrastructure
│   └── patches/
│       ├── __init__.py
│       └── priority_scheduler.py
└── README.md
```

### 底座：`VLLMPatch` 和 `@min_vllm_version`

外科手术式地补类或模块。`PatchTarget = Union[Type, ModuleType]`。写成 `VLLMPatch[TargetClass]`；`__class_getitem__` 把目标记在 `_patch_target`。直接 `apply` 基类会 `TypeError`。

`VLLMPatch.apply()`：

- 在 `target._applied_patches` 上记账
- **同一属性禁止打两次**（`ValueError`）
- 跳过 `_` 开头的名字和 `apply` 本身
- 把 `MethodType` 的 classmethod 绑回 target
- 其余属性 `setattr` 上去

```python
# vllm_custom_patches/core.py
from types import MethodType, ModuleType
from typing import Type, Union
from packaging import version
import vllm

PatchTarget = Union[Type, ModuleType]

class VLLMPatch:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, '_patch_target'):
            raise TypeError(f"{cls.__name__} must be defined as VLLMPatch[Target]")

    @classmethod
    def __class_getitem__(cls, target: PatchTarget) -> Type:
        if not isinstance(target, (type, ModuleType)):
            raise TypeError(f"Can only patch classes or modules, not {type(target)}")
        return type(f"{cls.__name__}[{target.__name__}]", (cls,), {'_patch_target': target})

    @classmethod
    def apply(cls):
        if cls is VLLMPatch:
            raise TypeError("Cannot apply base VLLMPatch class directly")
        target = cls._patch_target
        if not hasattr(target, '_applied_patches'):
            target._applied_patches = {}
        for name, attr in cls.__dict__.items():
            if name.startswith('_') or name in ('apply',):
                continue
            if name in target._applied_patches:
                raise ValueError(
                    f"{target.__name__}.{name} already patched by {target._applied_patches[name]}"
                )
            target._applied_patches[name] = cls.__name__
            if isinstance(attr, MethodType):
                attr = MethodType(attr.__func__, target)
            setattr(target, name, attr)
```

版本闸：装上的 vLLM 太旧就 **警告并跳过**，不把进程打崩：

```python
def min_vllm_version(version_str: str):
    def decorator(cls):
        original_apply = cls.apply
        @classmethod
        def checked_apply(cls):
            current = version.parse(vllm.__version__)
            minimum = version.parse(version_str)
            if current < minimum:
                logger.warning(
                    f"Skipping {cls.__name__}: requires vLLM >= {version_str}, "
                    f"but found {vllm.__version__}"
                )
                return
            original_apply()
        cls.apply = checked_apply
        cls._min_version = version_str
        return cls
    return decorator
```

文中的例子：`@min_vllm_version("0.9.1")`。

### 例子：按优先级调度

当时的 import：`from vllm.core.scheduler import Scheduler`（V0 路径；后来 V1 可能挪过类）。

```python
# vllm_custom_patches/patches/priority_scheduler.py
from vllm.core.scheduler import Scheduler
from vllm_custom_patches.core import VLLMPatch, min_vllm_version

@min_vllm_version("0.9.1")
class PrioritySchedulerPatch(VLLMPatch[Scheduler]):
    def schedule_with_priority(self):
        output = self._schedule()
        if hasattr(output, 'scheduled_seq_groups'):
            output.scheduled_seq_groups.sort(
                key=lambda seq: getattr(seq, 'priority', 0),
                reverse=True,
            )
        return output
```

这段是 **新增** `schedule_with_priority`（按 metadata 里的 `priority` 降序），帖子里的片段并没有替换 `schedule()` 本身。声明兼容 **vLLM 0.9.1+**。

### 插件入口和 `PatchManager`

vLLM 调用的是 `register_patches()`。`PatchManager` 管 `available_patches` 和 `applied_patches`。`apply_from_env()` 读 **`VLLM_CUSTOM_PATCHES`**：逗号分隔，例如 `VLLM_CUSTOM_PATCHES="PatchOne,PatchTwo"`。空 / 未设置 → 不加自定义补丁。

```python
# vllm_custom_patches/__init__.py
manager = PatchManager()

def register_patches():
    from vllm_custom_patches.patches.priority_scheduler import PrioritySchedulerPatch
    manager.register('PriorityScheduler', PrioritySchedulerPatch)
    manager.apply_from_env()
```

`apply_patch(name)` 查出类、调用 `.apply()`、记入 `applied_patches`；失败打日志，不把进程掀翻。

### `setup.py`

```python
setup(
    name='vllm-custom-patches',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['vllm>=0.9.1', 'packaging>=20.0'],
    entry_points={
        'vllm.general_plugins': [
            'custom_patches = vllm_custom_patches:register_patches'
        ]
    },
    python_requires='>=3.11',
)
```

`vllm.general_plugins` 这一行才是官方挂钩。格式：`{name} = {module}:{function}`。

## 用法

### 安装

```bash
pip install -e .
```

### 运行（文中当时的 CLI）

vanilla（不开补丁）：

```bash
VLLM_CUSTOM_PATCHES="" python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2
```

打开优先级调度：

```bash
VLLM_CUSTOM_PATCHES="PriorityScheduler" python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B-Instruct
```

### Docker

底：`vllm/vllm-openai:latest`。拷进包、`pip install -e`，默认 `ENV VLLM_CUSTOM_PATCHES=""`。CMD：

```
python -m vllm.entrypoints.openai.api_server --model ${MODEL_NAME} --host 0.0.0.0 --port 8000
```

同一镜像、两套配置：

```bash
docker run \
    -e MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct \
    -e VLLM_CUSTOM_PATCHES="PriorityScheduler" \
    -p 8000:8000 \
    vllm-with-patches

docker run \
    -e MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2 \
    -e VLLM_CUSTOM_PATCHES="" \
    -p 8000:8000 \
    vllm-with-patches
```

**原文特意写的坑：** `VLLM_CUSTOM_PATCHES` **不是** vLLM 官方环境变量——只是这篇文章里的例子名字。自己的插件包里换什么名字都行。

## 它怎么生效：插件生命周期

**原文的关键洞察：** vLLM 是多进程（tensor / pipeline / 其他并行）。它在 **创建的每一个进程** 里调用 `load_general_plugins()`，而且是在 **那个进程开始任何正经工作之前**。

于是补丁会进： **主进程**；**所有 worker**；**GPU worker、CPU worker、以及任何辅助进程**。加载发生在 **模型初始化之前**、**scheduler 创建之前**、**推理之前**。

每个进程的启动顺序，原文列了十条：

1. 创建进程（main、worker、…）
2. 插件系统启动：先于其他 vLLM 工作调用 `load_general_plugins()`
3. 用 entry point 发现所有已注册的 `vllm.general_plugins`
4. 跑插件函数（`register_patches()`）
5. 向 manager 注册补丁
6. 读环境变量（`VLLM_CUSTOM_PATCHES`）
7. 按名单 `VLLMPatch.apply()`
8. 版本校验（`@min_vllm_version`）
9. 在目标类上外科手术式地加 / 换方法
10. 这才轮到加载模型、初始化 scheduler、serving

宣称的保证：补丁在 **vLLM 做任何事之前** 已经活着，跨进程行为一致，也躲掉 monkey patch 撞上的 `EngineCore` 旧类问题。

## 原文列的好处

1. **补丁极小、够外科。** 不必复印整文件。`VLLMPatch` 可以只加一个方法，不必抄整类。
2. **同一份 vLLM 构建伺候多个模型。** 不同进程 / 镜像用不同的 `VLLM_CUSTOM_PATCHES`。
3. **带版本闸。** `@min_vllm_version("0.9.1")` 升级时选择跳过，而不是给你一个惊喜。
4. **不必再 fork、同步、rebase。** 升级就是 `pip install --upgrade vllm`，再测自己的补丁。
5. **甩掉 monkey patch 的复杂度。** 改动可追踪，没有整文件默默碎掉。
6. **官方支持的挂钩。** 走的是 `general_plugins` entry point。

## 为什么这个模式要紧

推理引擎跑得很快。假选择是：改内部，**或者** 跟上游兼容。插件模型把这道二选一撤了。运维开销小，长期灵活性还在。原文说它从原型能长到多模型生产，作者也在生产环境用过。

## 收束 / 要点

先考虑 general plugin，再决定要不要 fork 或 monkey patch。

- 用 `VLLMPatch[TargetClass]` 做类级、外科手术式改动
- 在 `setup.py` 里注册 `vllm.general_plugins`
- 用环境变量（例如 `VLLM_CUSTOM_PATCHES`，只是例子名）控制开哪些补丁
- 用 `@min_vllm_version` 做版本闸
- 一份 Docker 镜像，多套配置

## 联系（原文）

- LinkedIn：https://www.linkedin.com/in/dhruvil-bhatt-uci/
- 网站：https://www.dhruvilbhatt.com/
- Email：dhruvilbhattlm10@gmail.com
