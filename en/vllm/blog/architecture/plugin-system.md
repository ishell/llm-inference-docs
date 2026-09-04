---
source: https://vllm.ai/blog/2025-11-20-vllm-plugin-system
lang: en
fetched: 2026-09-04
---

# Building Clean, Maintainable vLLM Modifications Using the Plugin System

Chinese: [zh/vllm/blog/architecture/plugin-system.md](../../../../zh/vllm/blog/architecture/plugin-system.md)

2025-11-20. Author on the page: **Dhruvil Bhatt (AWS SageMaker)**. First posted on [Medium](https://medium.com/@dhruvilbhattlm10/building-clean-maintainable-vllm-modifications-using-the-plugin-system-e80df0f62861). Figure source named on the page: [vllm-ascend](https://github.com/vllm-project/vllm-ascend). Study note. Hardware-specific door: [hardware-plugin.md](hardware-plugin.md). A later in-tree user of `vllm.general_plugins`: [afd.md](../serving/afd.md). Cousins that also leave a door instead of a fork: [sleep-mode.md](sleep-mode.md), [kv-offload.md](../serving/kv-offload.md) / [mooncake.md](../serving/mooncake.md) (`KVConnector`), [rdt-weight-transfer.md](../serving/rdt-weight-transfer.md) / [native-rl.md](../serving/native-rl.md) (`WeightTransferEngine`).

Fits: custom scheduling, KV-cache behavior, hardware integrations, model-execution patches. Does not fit: rewriting the engine heart while tracking main every week — that still wants upstream or a fork tax.

Local figures (copyright remains with the original site; study copies):

![vllm plugin system arch](../../../../assets/vllm/blog/architecture/plugin-system/01-vllm-plugin-system-arch.png)

## Overview

vLLM is the high-throughput, low-latency serving engine: continuous batching, scheduling, PagedAttention, a production API. Teams still want to change internals — custom scheduling, KV handling, proprietary optimizations, a patch in the execution flow. That is where the three old paths start to hurt.

## The problem: “I need to modify vLLM… what now?”

If the change is simple, or it helps the general community, the clean answer is still **Option A — upstream**. The change lives in the open, gets review, and moves with vLLM.

Reality is often: **proprietary**, **domain-specific**, **too experimental**, **not generalizable** enough for upstream, or **blocked by internal timelines** that do not match open-source review. Then you need another path.

### Option B — maintain your own fork

First instinct: fork and add the patch there. Fine for tiny, slow-moving trees. **vLLM is not one of those.** Releases as close as **two weeks** apart; **hundreds of PRs every week**.

A long-lived fork means: constant rebase/merge; conflicts in the fastest-moving regions; re-applying patches by hand; heavy compatibility testing; an internal developer workflow around a **custom vLLM artifact**. Before long it is a **full-time job**. Unsustainable for many teams.

### Option C — monkey patching

A small Python package that monkey-patches vanilla vLLM at build/runtime looks attractive: no fork, no divergence, dynamic patches, small footprint. The page’s caveats:

- You **copy large chunks of vLLM source** — including the parts you do not modify — because monkey patches typically **replace entire classes or modules** for a ten-line change.
- **Every upgrade breaks the patch**, because you replaced files, not the lines of interest.
- **Debugging is painful** — is the bug in your patch, in unchanged vanilla code, or in the rewiring?
- Operational cost grows: every release is a **diff and re-sync of copied files** — a fork, disguised inside a package.
- Monkey-patching some modules (the page names **`Scheduler`**) often **does not work**: they run inside `EngineCore` in a **separate process**. `EngineCore` can keep calling the **stale** implementation. Process-synchronization issues, not just import-order tricks.

Monkey patching solves the surface problem and recreates the fork tax.

## A cleaner alternative: the plugin system

The post’s path is vLLM’s [general_plugin architecture](https://docs.vllm.ai/en/stable/design/plugin_system.html): inject targeted modifications **without altering upstream code**. Claimed properties: structured modular patches; runtime activation; surgical overrides; compatibility safeguards; no full-file duplication; no monkey-patch gymnastics; no maintained fork. Middle ground between “upstream everything” and “replace entire files.”

> **Note from the page:** vLLM provides **four** plugin groups — **platform**, **engine**, **model**, and **general** plugins. This article is specifically the **general plugin** system, loaded in **all** vLLM processes, which is why it is the clean modification door here. Types: [Types of Supported Plugins](https://docs.vllm.ai/en/latest/design/plugin_system/#types-of-supported-plugins). Platform plugins are the [hardware-plugin](hardware-plugin.md) story.

## Building a clean extensions framework

Using plugins, the author built a small extensions package as a container for custom modifications. Each patch:

- contains **just the snippet or class** that must change
- can be **enabled or disabled at runtime**
- can declare a **minimum supported vLLM version**
- can stay **dormant unless a specific model config requests it**

Because plugins apply at runtime, one **unified container image** can serve multiple models and enable different patches per model. Inspired by [ArcticInference](https://github.com/snowflakedb/ArcticInference).

## Implementation: a `general_plugins` package

Walk-through using the `vllm.general_plugins` entry point. Package name in the post: `vllm-custom-patches`.

### Project structure

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

### Core: `VLLMPatch` and `@min_vllm_version`

Surgical class/module patches. `PatchTarget = Union[Type, ModuleType]`. Subclass as `VLLMPatch[TargetClass]`; `__class_getitem__` stores `_patch_target`. Applying the bare `VLLMPatch` base is a `TypeError`.

`VLLMPatch.apply()`:

- records applied names on `target._applied_patches`
- **refuses double-patch** of the same attribute (`ValueError`: already patched by …)
- skips names starting with `_` and the name `apply`
- rebinds `MethodType` classmethods onto the target
- `setattr` of each remaining attribute onto the target

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

Version gate — skip with a warning rather than crash if the installed vLLM is too old:

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

Example from the page: `@min_vllm_version("0.9.1")`.

### Example patch: priority-based scheduling

Then-current import in the post: `from vllm.core.scheduler import Scheduler` (V0-era path; later V1 may have moved the class).

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

The example **adds** `schedule_with_priority` (sort by a `priority` field on metadata, higher first). It does not, in the posted snippet, replace `schedule()` itself. Compatible with **vLLM 0.9.1+** as declared.

### Plugin entry point and `PatchManager`

`register_patches()` is the function vLLM calls. A `PatchManager` holds `available_patches` and `applied_patches`. `apply_from_env()` reads **`VLLM_CUSTOM_PATCHES`**: comma-separated names, e.g. `VLLM_CUSTOM_PATCHES="PatchOne,PatchTwo"`. Empty / unset → no custom patches.

```python
# vllm_custom_patches/__init__.py
manager = PatchManager()

def register_patches():
    from vllm_custom_patches.patches.priority_scheduler import PrioritySchedulerPatch
    manager.register('PriorityScheduler', PrioritySchedulerPatch)
    manager.apply_from_env()
```

`apply_patch(name)` looks up the class, calls `.apply()`, appends to `applied_patches`, logs failures instead of crashing the process.

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

That `vllm.general_plugins` line is the official hook. `{name} = {module}:{function}`.

## Usage

### Install

```bash
pip install -e .
```

### Run (then-current CLI in the post)

Vanilla (no patches):

```bash
VLLM_CUSTOM_PATCHES="" python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2
```

With the priority scheduler patch:

```bash
VLLM_CUSTOM_PATCHES="PriorityScheduler" python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B-Instruct
```

### Docker

Base: `vllm/vllm-openai:latest`. Copy the package, `pip install -e`, default `ENV VLLM_CUSTOM_PATCHES=""`. CMD:

```
python -m vllm.entrypoints.openai.api_server --model ${MODEL_NAME} --host 0.0.0.0 --port 8000
```

Same image, two configs:

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

**Caveat from the page:** `VLLM_CUSTOM_PATCHES` is **not** an official vLLM environment variable — it is an example name in this article. Pick any env var in your own plugin package.

## How it works: plugin lifecycle

**Critical insight on the page:** vLLM is multi-process (tensor / pipeline / other parallelism). It calls `load_general_plugins()` in **every process it creates**, **before that process starts any actual work**.

So patches load in: the **main** process; **all worker** processes; **GPU workers, CPU workers, and any auxiliary processes**. Loading happens **before model initialization**, **before scheduler creation**, **before inference**.

Startup sequence in each process, as listed:

1. Process creation (main, worker, …)
2. Plugin system activation: `load_general_plugins()` before other vLLM work
3. Entry-point discovery of all registered `vllm.general_plugins`
4. Plugin function runs (`register_patches()`)
5. Patch registration with the manager
6. Environment check (`VLLM_CUSTOM_PATCHES`)
7. Selective application via `VLLMPatch.apply()`
8. Version validation (`@min_vllm_version`)
9. Surgical add/replace on target classes
10. Only then: model loading, scheduler init, serving

Guarantee claimed: patches are active **before vLLM does anything**, so behavior is consistent across processes and you avoid the `EngineCore` stale-class race that monkey patches hit.

## Benefits listed

1. **Extremely small, surgical patch definitions.** No duplicated files. `VLLMPatch` can add a single method without copying the class.
2. **Multiple models on the same vLLM build.** Different `VLLM_CUSTOM_PATCHES` per process/image env.
3. **Version-aware safety.** `@min_vllm_version("0.9.1")` skips instead of surprising you on upgrade.
4. **No fork, sync, or rebase.** Upgrade is `pip install --upgrade vllm` plus testing your patches.
5. **No monkey-patch complexity.** Trackable modifications; no silent whole-file breakages.
6. **Officially supported.** Uses vLLM’s `general_plugins` entry point.

## Why the pattern matters

Inference engines move fast. The false choice is: modify internals **or** stay compatible with upstream. The plugin model **removes that trade-off**. Operational overhead stays small; long-term flexibility stays. The page says it scales from prototypes to multi-model production, and that the author has used it in production environments.

## Final thoughts / key takeaways

Prefer general plugins before a fork or a monkey-patch strategy.

- Use `VLLMPatch[TargetClass]` for surgical, class-level modifications
- Register via `vllm.general_plugins` in `setup.py`
- Control patches with an env var such as `VLLM_CUSTOM_PATCHES` (example name only)
- Version-guard with `@min_vllm_version`
- One Docker image, multiple configurations

## Contact (from the page)

- LinkedIn: https://www.linkedin.com/in/dhruvil-bhatt-uci/
- Website: https://www.dhruvilbhatt.com/
- Email: dhruvilbhattlm10@gmail.com
