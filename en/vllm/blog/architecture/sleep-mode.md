---
source: https://vllm.ai/blog/2025-10-26-sleep-mode
lang: en
fetched: 2026-08-31
---

# Sleep Mode

2025-10-26. Two models that each fit, but not together: 2× VRAM, or 30–100+ s reload. Sleep hibernates the model and keeps the process.  Demo: vLLM 0.11.0.

**L1:** weights → CPU RAM, drop KV. Fastest wake. Needs RAM.  
**L2:** drop weights too; tiny buffers only. Wake reloads from disk.

Both claimed **18–200×** vs full reload; works with TP/PP/EP. Cold start still pays allocator, CUDA-graph recapture, kernel JIT, first-request warmup. Sleep keeps those. First inference after wake **61–88%** faster than cold — infrastructure, not memcpy.

```bash
VLLM_SERVER_DEV_MODE=1
vllm serve <model> --enable-sleep-mode --port 8001
curl -X POST 'localhost:8001/sleep?level=2'
curl -X POST 'localhost:8001/wake_up'
# L2 only: collective_rpc reload_weights + reset_prefix_cache
```

Admin endpoints (`/sleep`, `/wake_up`, `/collective_rpc`, `/reset_prefix_cache`) need `VLLM_SERVER_DEV_MODE=1`; trusted networks only.

A100 235B-FP8 TP4 ↔ 30B TP1: wake **18–20×** vs cold load. A4000 0.6B / Phi-3-vision: wake **0.1–0.8 s**, **58–203×**; five switches ~85 s vs 226 s. Same small pair on A100: no-sleep 357 s; L1 **112.6 s**; L2 **124.6 s**.

[torch-compile.md](torch-compile.md) made startup expensive. Sleep refuses to kill the process. Dense LoRA is the other multi-model path; this one swaps whole weights.

Local figures (copyright remains with the original site; study copies):

![sleepmode](../../../../assets/vllm/blog/architecture/sleep-mode/01-sleepmode.png)
