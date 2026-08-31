---
source: https://vllm.ai/blog/2025-12-15-vllm-epd
lang: en
fetched: 2026-08-31
---

# Encoder Disaggregation (EPD)

2025-12-15. **Encoder vs prefill/decode**, not the text P/D in the Router post. ViT is one-shot compute-bound; prefill is fat GEMMs; decode is memory-bound. Colocation: `[E PD]` cannot overlap; one image stalls mixed batches; one parallelism plan for three profiles.

Disagg: pipeline `E → PD`; text-only skips encoder; embeddings in remote **Encoder Cache**. Proxy fans out encoder jobs; PD loads by image hash via `ECConnector*` (`has_caches` / `save_caches` / `start_load_caches`).

**Goodput** = max QPS meeting P99 TTFT 20s and P99 TPOT 100 ms. 4×A100, Qwen3-VL-4B. Short text: 4 images **6→12 QPS**. Long text: **2–2.5×** goodput. Same shape on Ascend 910B.

Native since v0.11.1 (PR #25233). Earlier single-node trick: ViT DP + LM TP.
