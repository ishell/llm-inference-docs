---
source: https://vllm.ai/blog/2025-01-10-dev-experience
lang: en
fetched: 2026-09-04
---

# Installing and Developing vLLM with Ease

Chinese: [zh/vllm/blog/architecture/dev-experience.md](../../../../zh/vllm/blog/architecture/dev-experience.md)

2025-01-10. **vLLM Team**. Study note; then-latest public **v0.6.6.post1**; uv vs pip timings are **their** 8th-gen CPU, cached, clean venv. Release/CI later: [production-quality.md](../performance/production-quality.md). Downstream named on the page: [openrlhf.md](../serving/openrlhf.md). Native RL APIs came later: [native-rl.md](../serving/native-rl.md).

Fits: pinning a nightly **by commit**, Python-only editable installs without CUDA compile, uv vs pip extra-index semantics. Does not fit: treating **~200×** as a universal installer speedup.

## TL;DR

- Stable PyPI and nightly wheels
- Python-only `VLLM_USE_PRECOMPILED=1` editable install; C++/CUDA uses a compile cache
- Production bisects by **commit hash**, not only tags

The post’s pitch: more than a package — a trackable, participatory ecosystem while models and features move weekly.

## Released versions

PyPI: [vllm](https://pypi.org/project/vllm/).

```sh
pip install vllm
```

[uv](https://github.com/astral-sh/uv) (env setup in the [docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html?device=cuda#create-a-new-python-environment)):

```sh
uv pip install vllm
```

Their box (Intel 8th-gen CPU), **cached packages, clean venv**:

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

~**75 s** vs ~**0.38 s** ≈ **200×** on that machine.

## Nightly from `main`

A wheel per commit on `main`.

**pip** (needs `--pre` so pre-releases are searched):

```sh
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

**uv**:

```sh
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
```

## Development

### Python

No kernel compile:

```sh
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install -e .
```

`VLLM_USE_PRECOMPILED=1` uses pre-compiled CUDA kernels. Aimed at API, model support, integration work. Claimed to run on a laptop. More: [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html?device=cuda#build-wheel-from-source).

### C++ / kernels

Compile cache to cut rebuild time. Same docs page.

## Pin a commit

Wheels for every `main` commit — bisect interfaces that were still moving. Downstream named: OpenRLHF, veRL, open_instruct, LLaMA-Factory.

**uv** (recommended). Extra-index packages have [higher priority than the default index](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes), so a developing wheel can beat the latest public release (then **v0.6.6.post1**):

```sh
# use full commit hash from the main branch
export VLLM_COMMIT=72d9c316d3f6ede485146fe5aabd4e61dbc59069
uv pip install vllm --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}
```

**pip** merges extra-index with PyPI and picks “latest”, so a developing version older than the release is hard. Full wheel URL:

```sh
# use full commit hash from the main branch
export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd
pip install https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```

## Close / contact

Trust, track changes, participate. Collab: [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu). [GitHub](https://github.com/vllm-project/vllm), [Slack](https://slack.vllm.ai/).

## Acknowledgements

[uv](https://docs.astral.sh/uv/) / [Charlie Marsh](https://github.com/charliermarsh). [Kevin Luu](https://github.com/khluu) (Anyscale), [Daniele Trifirò](https://github.com/dtrifiro) (Red Hat), [Michael Goin](https://github.com/mgoin) (Neural Magic). Led by [Kaichao You](https://github.com/youkaichao) and [Simon Mo](https://github.com/simon-mo) (UC Berkeley).
