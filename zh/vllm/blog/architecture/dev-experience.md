---
source: https://vllm.ai/blog/2025-01-10-dev-experience
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 安装与开发：nightly 按 commit 钉死，uv 比 pip 快两个数量级（他们那台机）

英文对照：[en/vllm/blog/architecture/dev-experience.md](../../../../en/vllm/blog/architecture/dev-experience.md)  
原文：https://vllm.ai/blog/2025-01-10-dev-experience  
2025-01-10。署名 **vLLM Team**。当时公开最新 **v0.6.6.post1**。uv 对 pip 的时间是 **他们** 那台 8 代 Intel CPU、缓存命中、干净 venv 上的演示。后来的发版 / CI 见 [production-quality](../performance/production-quality.md)。页上点名的下游：[openrlhf](../serving/openrlhf.md)。原生 RL API 更晚：[native-rl](../serving/native-rl.md)。

适用：按 **commit** 钉 nightly、只改 Python 不必编 CUDA、搞清 uv 和 pip 对 extra-index 的优先级。不适合：把 **约 200×** 当成普适的安装加速。

## TL;DR

- 稳定 PyPI 和 nightly wheel
- 只改 Python：`VLLM_USE_PRECOMPILED=1` 可编辑安装；C++ / CUDA 走编译缓存
- 生产用 **commit hash** bisect，不只跟标签

原文的说法：不只是一个包——模型和功能按周在变，要一套可追踪、能一起改的生态。

## 发版

PyPI：[vllm](https://pypi.org/project/vllm/)。

```sh
pip install vllm
```

[uv](https://github.com/astral-sh/uv)（环境搭法见 [文档](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html?device=cuda#create-a-new-python-environment)）：

```sh
uv pip install vllm
```

他们那台机（Intel 8 代 CPU），**缓存命中、干净 venv**：

```sh
# with cached packages, clean virtual environment
$ time pip install vllm
...
pip install vllm 59.09s user 3.82s system 83% cpu 1:15.68 total

# with cached packages, clean virtual environment
$ time uv pip install vllm
...
uv pip install vllm 0.17s user 0.57s system 193% cpu 0.383 total
```

约 **75 s** 对约 **0.38 s** ≈ **200×**，那一台。

## 从 `main` 装 nightly

`main` 上每个 commit 一只 wheel。

**pip**（要 `--pre` 才会搜预发版）：

```sh
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

**uv**：

```sh
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
```

## 开发

### Python

不必编 kernel：

```sh
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install -e .
```

`VLLM_USE_PRECOMPILED=1` 用预编译 CUDA kernel。面向 API、模型支持、集成。宣称笔记本也能跑。更多：[从源码编](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html?device=cuda#build-wheel-from-source)。

### C++ / kernel

编译缓存，少重编。同一页文档。

## 钉死某个 commit

`main` 每个 commit 都有 wheel——接口还在晃的时候好 bisect。点名的下游：OpenRLHF、veRL、open_instruct、LLaMA-Factory。

**uv**（推荐）。extra-index 上的包 [优先级高于默认 index](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)，开发中的 wheel 可以压过当时最新的公开发版（**v0.6.6.post1**）：

```sh
# use full commit hash from the main branch
export VLLM_COMMIT=72d9c316d3f6ede485146fe5aabd4e61dbc59069
uv pip install vllm --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}
```

**pip** 把 extra-index 和 PyPI 混在一起只挑「最新」，比发版旧的开发版不好装。要用完整 wheel URL：

```sh
# use full commit hash from the main branch
export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd
pip install https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```

## 收束 / 联系

信任、追踪改动、一起改。合作：[vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)。[GitHub](https://github.com/vllm-project/vllm)、[Slack](https://slack.vllm.ai/)。

## 致谢

[uv](https://docs.astral.sh/uv/) / [Charlie Marsh](https://github.com/charliermarsh)。[Kevin Luu](https://github.com/khluu)（Anyscale）、[Daniele Trifirò](https://github.com/dtrifiro)（Red Hat）、[Michael Goin](https://github.com/mgoin)（Neural Magic）。Berkeley 这边领的是 [Kaichao You](https://github.com/youkaichao) 和 [Simon Mo](https://github.com/simon-mo)。
