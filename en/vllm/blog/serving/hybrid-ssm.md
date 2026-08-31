---
source: https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg
lang: en
fetched: 2026-08-31
---

# Hybrid SSM P/D disaggregation

2026-04-21. `vllm>=0.20.0`. Nemotron-H-style: Mamba SSM interleaved with full attention. Additive on NIXL; transformers unchanged. Study note.

FA KV is per-token K/V. Mamba stores a collapsed conv + temporal SSM (no token axis; `block_size`≡1). HMA pools groups onto one physical tensor: same page, FA sees K/V, Mamba sees conv+SSM+pad. One uniform NIXL descriptor list cannot index both.

**Dual descriptors** over the same memory (FA K/V split for heterogeneous TP, then Mamba). **Physical vs logical blocks:** FlashInfer-style physical size applies to FA only; SSM keeps logical counts (P/D TP may differ). **3-descriptor conv:** heterogeneous TP cannot RDMA-gather interleaved SD-layout `(state_len, dim)` x|B|C. Require `VLLM_SSM_CONV_STATE_LAYOUT=DS`; contiguous x, B, C. Homogeneous TP can stay Conv+SSM.

PRs #36687 / #37416 / #37635 / #37310. Router P/D assumes one shape of memory; hybrids need two dictionaries on the same NIXL pipe.
