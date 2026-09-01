---
source: https://vllm.ai/blog/2026-08-10-nemotron-3-5-lightning-vllm
lang: en
fetched: 2026-09-01
---

# Nemotron 3.5 Lightning day-0

Chinese: `../../zh/vllm/blog/serving/nemotron-35-lightning.md`  
2026-08-10. Demo: image `vllm/vllm-openai:v0.27.1`. Figures on the original page.

Distilled from Nemotron 3 Ultra. Hybrid MoE **30B / 3B active**, 1M context, text-only. Same architecture as Nemotron 3 except weights and the spec stack. **Not a new engine.** Role: frontier model plans; this one runs the many small steps.

**Quant:** BF16, NVFP4. **HW:** Spark, Station, RTX PRO, Jetson, H100/H200/A100/L40S, B200/GB200, B300/GB300.

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --max-num-seqs 256 --max-num-batched-tokens 32768 \
  --enable-prefix-caching --async-scheduling \
  --mamba-backend flashinfer --moe-backend humming --linear-backend humming \
  --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```

Spec: **MTP**, **DFlash**, **DSpark**. Low-latency → DSpark; max throughput (then) → no spec. NVFP4 often `--kv-cache-dtype fp8` + `--moe-backend marlin`. Humming W4A16 ReLU2 MoE claimed ~**+20%** vs Marlin. ReplaySSM on Mamba2.

Demo: up to **4×** throughput vs similar-size open models. PinchBench: ~**30%** faster to finish 10k tasks at comparable accuracy. Pareto charts: Spark / H100, 32K prefix then 10× 2k/1k.
