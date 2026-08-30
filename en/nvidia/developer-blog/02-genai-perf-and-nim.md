---
source: https://developer.nvidia.com/blog/llm-performance-benchmarking-measuring-nvidia-nim-performance-with-genai-perf/
lang: en
fetched: 2026-08-30
---

# Part 2: LLM Inference Benchmarking with GenAI-Perf and NIM

Source: https://developer.nvidia.com/blog/llm-performance-benchmarking-measuring-nvidia-nim-performance-with-genai-perf/

**GenAI-Perf is being phased out; use AIPerf for new work.** The NIM guide chapter 4 is the AIPerf version: `en/nvidia/nim-benchmarking/04-quickstart.md`.

This post deploys Llama 3.1 8B Instruct with NIM and sweeps GenAI-Perf.

## Why

NIM ships optimized containers (TensorRT-LLM or vLLM backends). Numbers on the NIM Performance page were collected with GenAI-Perf. Reproduce on your hardware.

Run the client on the same host as NIM unless you want network in the measurement.

## Flow

1. `docker run` NIM with NGC API key and a local model cache.
2. Start Triton SDK container, mount a workdir, run `genai-perf profile` with ISL/OSL/concurrency, `ignore_eos:true`.
3. Sweep use cases (e.g. 200/200, 200/5, 1000/200, 200/1000) × concurrency `{1,2,5,10,50,100,250}`.
4. `--measurement-interval 30000` ms; raise for 70B / concurrency 250 (e.g. 100000 ms).
5. Artifacts under `artifacts/`; plot TTFT vs RPS. Example plot saturates around concurrency 50.

## LoRA

`-m adapter1 adapter2 adapter3 --model-selection-strategy random|round_robin`.
