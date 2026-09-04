---
source: https://vllm.ai/blog/2025-05-12-hardware-plugin
lang: en
fetched: 2026-09-04
---

# Introducing vLLM Hardware Plugin, Best Practice from Ascend NPU

Chinese: [zh/vllm/blog/architecture/hardware-plugin.md](../../../../zh/vllm/blog/architecture/hardware-plugin.md)

2025-05-12. Author on the page: **The Ascend Team on vLLM**. No figures in the original (logo only — not copied). Hardware Pluggable RFC: [#11162](https://github.com/vllm-project/vllm/issues/11162), joint work with the Ascend team **since December 2024**. Proof in-tree of the idea: [vllm-ascend](https://github.com/vllm-project/vllm-ascend), [vllm-spyre](https://github.com/vllm-project/vllm-spyre). The later, broader plugin story is [plugin-system.md](plugin-system.md).

The RFC's claim: hardware can join vLLM **decoupled** — rapid, modular support instead of patches in core.

## Why

vLLM already had several backends. As that list grew:

- **Code complexity.** Each backend owns its own `Executor`, `Worker`, `Runner`, and `Attention`. Non-generic code is scattered through the tree.
- **Maintenance cost.** Backend authors *and* the core community pay. When the backend maintainer is missing, community time is too thin to land new features on that path.
- **No real extensibility.** Layering via Executor / Worker / Runner / Attention is tidy, but a **new** device still needed invasive edits or patches, not dynamic registration.

What they wanted instead:

- **Decoupled codebase.** Backend lives in its own package; core stays generic.
- **Lower maintenance.** Core authors work on generic features, not every device's quirks.
- **Faster, more independent integration.** A new backend does less work in core and can evolve on its own clock.

## What it is (two RFCs underneath)

1. [[RFC] vLLM Plugin System](https://github.com/vllm-project/vllm/issues/7131) — plugins for custom models, executors, schedulers, and so on. (The November 2025 blog is the later write-up of this door.)
2. [[RFC] Make vLLM Device-Agnostic](https://github.com/vllm-project/vllm/issues/9268) and [PR #6080](https://github.com/vllm-project/vllm/pull/6080) — the **`platform`** submodule: hardware-specific code in one place, fewer `if device` branches in core, the modularization foundation.

On those, [[RFC] Hardware Pluggable](https://github.com/vllm-project/vllm/issues/11162) makes **`Platform` itself a plugin**, and refactors `Executor`, `Worker`, `ModelRunner`, `AttentionBackend`, and `Communicator` so a plugin can fill them.

By this post the community had the Platform module from the RFC, checked on Ascend NPU and IBM Spyre via the two repos above.

## How to plug a new backend

### Developer

**Step 1 — project + `Platform`.** New Python package, `platform.py`. Import `Platform` from `vllm.platforms`, implement the attributes and methods the class asks for. Example they point at: [`vllm_ascend/platform.py`](https://github.com/vllm-project/vllm-ascend/blob/72a43a61d8d2193dddbfcc60578fd642008225a5/vllm_ascend/platform.py#L52) (commit `72a43a61`).

**Step 2 — the four bases, as needed:**

```python
from vllm.worker.worker_base import WorkerBase
from vllm.worker.model_runner_base import ModelRunnerBase
from vllm.attention.backends.abstract import AttentionBackend
from vllm.distributed.device_communicators.base_communicator import CommunicatorBase
```

Each has a vLLM base class. Walk [vllm-ascend's package](https://github.com/vllm-project/vllm-ascend/tree/main/vllm_ascend) for a full example. Class names are of that vintage; later V1 may have moved files.

**Step 3 — register with a Python entry point** in `setup.py`:

```python
setup(
    entry_points={
        "vllm.platform_plugins": [
            "{your_platform_name} = {code_path}:{register_function}"
        ]
    }
)
```

- `{your_platform_name}` — arbitrary backend name.
- `{code_path}` — the Python module.
- `{register_function}` — returns the **path of the `Platform` class** from step 1.

Ascend's [`setup.py`](https://github.com/vllm-project/vllm-ascend/blob/72a43a61d8d2193dddbfcc60578fd642008225a5/setup.py#L102) is the practical example.

### User

Install vanilla vLLM plus the plugin:

```bash
pip install vllm vllm-ascend
```

Logs that mean it worked (timestamp from the post: `02-06 15:49:01`):

```
INFO 02-06 15:49:01 __init__.py:30] Available plugins for group vllm.platform_plugins:
INFO 02-06 15:49:01 __init__.py:32] name=ascend, value=vllm_ascend:register
…
INFO 02-06 15:49:01 __init__.py:44] plugin ascend loaded.
INFO 02-06 15:49:01 __init__.py:181] Platform plugin ascend is activated
```

## What's next (then)

They listed four follow-ons:

1. Keep hardening **V1** and **VLMs**.
2. More plugin surfaces: **scheduler**, **graph mode**, **custom operators**.
3. Better UX and performance.
4. Keep the plugin architecture **stable** for hardware that actually needs it.

Try it; questions go to [vLLM Slack](https://slack.vllm.ai) **`#sig-extensible-hardware`**.

## Acknowledgements

vLLM maintainers named for refactor, discussion, review: [Kaichao You](https://github.com/youkaichao), [Simon Mo](https://github.com/simon-mo), [Cyrus Leung](https://github.com/DarkLight1337), [Robert Shaw](https://github.com/robertgshaw2-redhat), [Michael Goin](https://github.com/mgoin), [Jie Li](https://github.com/jeejeelee).

Ascend team on vLLM (design + implementation): [Xiyuan Wang](https://github.com/wangxiyuan), [Shanshan Shen](https://github.com/shen-shanshan), [Chenguang Li](https://github.com/noemotiovon), [Mengqing Cao](https://github.com/MengqingCao).

Spyre team (pluggable **scheduler**): [Joe Runde](https://github.com/joerunde), [Yannick Schnider](https://github.com/yannicks1).

Also: [yancong](https://github.com/ice-tong) (extendable quantization), [Aviv Keshet](https://github.com/akeshet) (extendable `SamplingParams`).

[torch.compile](torch-compile.md) takes optimization out of model files; this takes hardware out of core. Both exist so “another accelerator” does not require a knife through trunk.
