---
source: https://vllm.ai/blog/2025-01-10-dev-experience
lang: en
fetched: 2026-09-01
---

# Install and develop: pin nightly by commit; uv two orders faster than pip on their box

Chinese: `../../zh/vllm/blog/architecture/dev-experience.md`  
Then-latest public: v0.6.6.post1. Demo timings.

`pip install vllm` / `uv pip install vllm`. Nightly: `--extra-index-url https://wheels.vllm.ai/nightly` (pip needs `--pre`). Python edits: `VLLM_USE_PRECOMPILED=1 pip install -e .` skips CUDA compile. Pin a commit: uv treats extra-index as higher priority; pip merges with PyPI and picks “latest”, so use the full wheel URL. Their 8th-gen CPU, cached: pip ~75s, uv ~0.38s (~**200×**). Production bisects by commit hash, not only PyPI tags.
