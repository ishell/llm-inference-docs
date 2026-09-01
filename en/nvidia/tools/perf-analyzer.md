---
source: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/perf-analyzer-README.html
lang: en
fetched: 2026-09-01
---

# Triton Performance Analyzer

CLI for **classic models on Triton** (classify, detect, embed — one-shot in, one-shot out). You change an optimization; it tells you whether throughput and latency moved.

**LLMs / generative models use AIPerf** (formerly GenAI-Perf). That ruler knows tokens and streaming. Perf Analyzer reports infer/sec and client average batch latency. GenAI-Perf still calls it underneath; you do not need to swing this knife at GPT yourself.

The official page tracks an older pin in the Triton user guide (directory said 2.65.0 when fetched; current Triton releases move on). The flow has not.

## Load modes

- **Concurrency** — keep N in-flight requests. The lobby always holds that many people.
- **Request rate** — send consecutive requests at a set rate.
- **Custom interval** — send on a supplied interval sequence.

## Measurement modes

- **Time windows** — measure over a time interval until the run is judged stable.
- **Count windows** — same, but each window is N completed requests.

Also profiles sequence, ensemble, and decoupled models (outputs need not match inputs 1:1). Inputs can be auto-generated or supplied; outputs can be checked.

## Five-step quickstart

Image tag is `yy.mm`, e.g. `23.02`. The LLM toolchain often uses a newer SDK tag; this page follows the README.

**1. Triton container**

```bash
export RELEASE=<yy.mm>
docker pull nvcr.io/nvidia/tritonserver:${RELEASE}-py3
docker run --gpus all --rm -it --net host \
  nvcr.io/nvidia/tritonserver:${RELEASE}-py3
```

**2. Example model `simple`**

```bash
git clone --depth 1 https://github.com/triton-inference-server/server
mkdir model_repository
cp -r server/docs/examples/model_repository/simple model_repository
```

**3. Start the server**

```bash
tritonserver --model-repository $(pwd)/model_repository &> server.log &
curl -v localhost:8000/v2/health/ready   # expect HTTP/1.1 200 OK
# detach: CTRL-p CTRL-q
```

**4. SDK container (ships `perf_analyzer`)**

```bash
docker pull nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
docker run --gpus all --rm -it --net host \
  nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
```

**5. Run**

```bash
perf_analyzer -m simple
```

Remote: `-u host:8000`. Concurrency sweeps use `--concurrency-range` on the Triton tuning page.

How to read the output and the stability rule: official “full quick start.” Contributing and questions are at the bottom of that README; use an MCVE when you need code help.

Taking a trained model from “it loads” to “it can ship” is `triton-performance-tuning.md`: Perf Analyzer baseline, Model Analyzer search of `config.pbtxt`, then measure again.
