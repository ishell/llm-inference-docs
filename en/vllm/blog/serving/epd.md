---
source: https://vllm.ai/blog/2025-12-15-vllm-epd
lang: en
fetched: 2026-08-31
---

# Encoder Disaggregation (EPD)

2025-12-15. **Encoder vs prefill/decode**, not the text P/D in the Router post. Two different “disaggregations.” Native in v0.11.1 (PR #25233). 

Before a VLM speaks, images go through a ViT. Encoder: one-shot, compute-bound, wants fat parallelism. Prefill: large GEMMs, bandwidth. Decode: memory-bound, long-lived. Colocate them and:

- The pipeline is `[E PD] → [E PD]`; encoder work cannot overlap someone else’s decode.
- Resolution / image count make encoder latency jump; one image in a mixed batch stalls text-only requests.
- One parallelism plan for three profiles. More images → you buy extra decode GPUs.

`mm_encoder_tp_mode="data"` in `optimization.md` is the single-node cousin (batch-level DP on a small encoder). EPD takes that knife to the cluster.

Disagg: pipeline `E → PD`; text-only skips the encoder; embeddings live in a remote **Encoder Cache** (logos / product shots computed once). Proxy fans out encoder jobs; workers write remote; PD nodes carry an image hash and a connector loads embeddings into the model runner.

API skeleton: `ECConnectorRole` (scheduler / worker), `has_caches` / `save_caches` / `start_load_caches`. Cousin of the text KVConnector: do not recompute intermediate state across machines.

**Goodput** = max QPS meeting P99 TTFT **20 s** and P99 TPOT **100 ms**. 4×A100, Qwen3-VL-4B, `vllm bench serve --dataset-name random-mm`. 1 encoder + 3 PD vs `--data-parallel-size 4`.

Short text (~400 tokens): 1 image 23→24 QPS; **4 images 6→12**. P99 often 20–50% lower. Without EPD, TPOT blows up around 12–14 QPS with many images.

Long text (~2000 tokens), decode-heavy: baseline 8 QPS (1 image) / 4 QPS (3–4 images); EPD holds **18 / 11 / 9 / 8**, about **2–2.5×**. Decode throughput +10–30%; P99 TTFT −30–50%, TPOT −20–40%. Same shape on Ascend 910B (Qwen2.5-VL-7B): +5–20% throughput, tighter tails — structure, not vendor.

Earlier trick: ViT DP + LM TP on one node (SGLang followed). Dynamo had an EPD-shaped split first. Papers: ModServe, encoder-decoder disaggregation. Processor / IPC multimodal caches avoid re-sending the same image; EPD moves the building.

Local figures (copyright remains with the original site; study copies):

![image](../../../../assets/vllm/blog/serving/epd/01-image.png)

![workflow](../../../../assets/vllm/blog/serving/epd/02-workflow.png)

![plot len400 epd vs non epd](../../../../assets/vllm/blog/serving/epd/03-plot_len400_epd_vs_non_epd.png)

![plot len2000 epd vs non epd](../../../../assets/vllm/blog/serving/epd/04-plot_len2000_epd_vs_non_epd.png)

![npu plot len400 epd vs non epd](../../../../assets/vllm/blog/serving/epd/05-npu_plot_len400_epd_vs_non_epd.png)

![npu plot len2000 epd vs non epd](../../../../assets/vllm/blog/serving/epd/06-npu_plot_len2000_epd_vs_non_epd.png)
