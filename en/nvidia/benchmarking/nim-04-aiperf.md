---
source: https://docs.nvidia.com/nim/benchmarking/llm/latest/quickstart.html
lang: en
fetched: 2026-08-30
---

# Using AIPerf to Benchmark

NVIDIA AIPerf is a client-side generative AI benchmarking tool that reports TTFT, ITL, TPS, RPS, and related metrics. It works with any OpenAI-compatible inference service, including NVIDIA NIM. This section walks through benchmarking a Llama-3 model with AIPerf.

Tool notes (install, formulas, scheduling, five workloads) live in `../tools/`: `aiperf.md`, `aiperf-metrics.md`, `aiperf-load-generator.md`, `aiperf-comprehensive.md`. For metric definitions, see Metrics. For parameter guidance, see Parameters and Best Practices.

## Set Up an OpenAI-Compatible Llama-3 Service with NVIDIA NIM

NVIDIA NIM is the fastest path to put an LLM into production. See NIM LLM getting started docs for hardware and NGC keys.

During startup the NIM container downloads resources and serves the model. Success looks like:

```
(APIServer pid=74) INFO:     Started server process [74]
(APIServer pid=74) INFO:     Waiting for application startup.
(APIServer pid=74) INFO:     Application startup complete.
```

Query the OpenAI-compatible API:

```python
from openai import OpenAI
client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="not-used")
prompt = "Once upon a time"
response = client.completions.create(
    model="meta/llama-3.1-8b-instruct",
    prompt=prompt,
    max_tokens=16,
    stream=False
)
print(response.choices[0].text)
```

> NVIDIA tests showed extra Docker flags can improve inference: `--security-opt seccomp=unconfined` or `--privileged`. Up to ~5% with NIM TensorRT-LLM v0.10.0 backend, up to ~20% with OSS vLLM backend (v0.4.3) / NIM vLLM 1.0.0, verified on DGX A100 and H100. These flags raise security risk. Use only after reviewing your security requirements.

## Set Up AIPerf and Warm Up

Run AIPerf on the **same host** as NIM unless you intentionally want network latency in the measurement.

```bash
export RELEASE="26.06"  # latest yy.mm
export WORKDIR=<YOUR_AI_PERF_WORKING_DIRECTORY>

docker run -it --net=host --gpus=all -v $WORKDIR:/workdir \
  nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
```

Inside the container:

```bash
pip install aiperf
pip install huggingface_hub
hf auth login   # Llama-3 tokenizer is gated
```

Warm-up load test:

```bash
export INPUT_SEQUENCE_LENGTH=200
export INPUT_SEQUENCE_STD=10
export OUTPUT_SEQUENCE_LENGTH=200
export CONCURRENCY=10
export REQUEST_COUNT=$(($CONCURRENCY * 3))
export MODEL=meta/llama-3.1-8b-instruct

cd /workdir
aiperf profile \
   -m $MODEL \
   --endpoint-type chat \
   --streaming \
   -u localhost:8000 \
   --synthetic-input-tokens-mean $INPUT_SEQUENCE_LENGTH \
   --synthetic-input-tokens-stddev $INPUT_SEQUENCE_STD \
   --concurrency $CONCURRENCY \
   --request-count $REQUEST_COUNT \
   --output-tokens-mean $OUTPUT_SEQUENCE_LENGTH \
   --extra-inputs min_tokens:$OUTPUT_SEQUENCE_LENGTH \
   --extra-inputs ignore_eos:true \
   --tokenizer meta-llama/llama-3.1-8b-instruct \
   --profile-export-file ${INPUT_SEQUENCE_LENGTH}_${OUTPUT_SEQUENCE_LENGTH}.json
```

This example sets ISL, OSL, and concurrency, and ignores EOS so output reaches the intended length.

## Sweep Use Cases

Sweep ISL/OSL combinations and concurrency. Warm up first (previous section).

```bash
declare -A useCases
useCases["Translation"]="200/200"
useCases["Text classification"]="200/5"
useCases["Text summary"]="1000/200"

runBenchmark() {
   local description="$1"
   local lengths="${useCases[$description]}"
   IFS='/' read -r inputLength outputLength <<< "$lengths"

   echo "Running AIPerf for $description with ISL=$inputLength OSL=$outputLength"
   for concurrency in 1 2 5 10 50 100 250; do
       local INPUT_SEQUENCE_LENGTH=$inputLength
       local OUTPUT_SEQUENCE_LENGTH=$outputLength
       local CONCURRENCY=$concurrency
       local REQUEST_COUNT=$(($CONCURRENCY * 3))
       local MODEL=meta/llama-3.1-8b-instruct

       aiperf profile \
           -m $MODEL \
           --endpoint-type chat \
           --streaming \
           -u localhost:8000 \
           --synthetic-input-tokens-mean $INPUT_SEQUENCE_LENGTH \
           --synthetic-input-tokens-stddev 0 \
           --concurrency $CONCURRENCY \
           --request-count $REQUEST_COUNT \
           --output-tokens-mean $OUTPUT_SEQUENCE_LENGTH \
           --extra-inputs min_tokens:$OUTPUT_SEQUENCE_LENGTH \
           --extra-inputs ignore_eos:true \
           --tokenizer meta-llama/llama-3.1-8b-instruct \
           --artifact-dir artifact/ISL${INPUT_SEQUENCE_LENGTH}_OSL${OUTPUT_SEQUENCE_LENGTH}/CON${CONCURRENCY}
   done
}

for description in "${!useCases[@]}"; do
   runBenchmark "$description"
done
```

`--request-count` is set to 3× concurrency for a stable sample. High concurrency + large model + large ISL/OSL can take a long time.

## Analyze Output

AIPerf writes under `artifact/`, organized by ISL/OSL and concurrency:

```
/workdir/artifact
 └── ISL200_OSL200
     ├── CON1
     |   ├── logs/aiperf.log
     |   ├── profile_export_aiperf.csv
     |   └── profile_export_aiperf.json
     ├── CON10
     ...
```

Main results are in `profile_export_aiperf.json`:

```python
import json
with open('artifact/ISL200_OSL5/CON1/profile_export_aiperf.json') as f:
    data = json.load(f)
```

Collect TPS and TTFT across concurrencies:

```python
ISL, OSL = 200, 5
concurrencies = [1, 2, 5, 10, 50, 100, 250]
TPS, TTFT = [], []
for con in concurrencies:
   with open(f'artifact/ISL{ISL}_OSL{OSL}/CON{con}/profile_export_aiperf.json') as f:
       data = json.load(f)
       TPS.append(data['output_token_throughput']['avg'])
       TTFT.append(data['time_to_first_token']['avg'])
```

Plot latency–throughput (each point is a concurrency):

```python
import matplotlib.pyplot as plt
plt.plot(TTFT, TPS, marker='o')
for i, c in enumerate(concurrencies):
    plt.text(TTFT[i], TPS[i], str(c), ha='center', va='bottom')
plt.xlabel("Single User: time to first token(s)")
plt.ylabel("Total System: tokens/s")
plt.grid(True)
plt.show()
```

## Interpret Results

X-axis = TTFT, Y-axis = system throughput, labels = concurrency.

- If you have a **latency budget**: pick max acceptable TTFT on X; the matching Y and concurrency are the highest throughput you can get within that budget.
- If you have a **target concurrency**: find that dot; X/Y are latency and throughput at that load.

The curve also shows concurrencies where latency jumps with little throughput gain (e.g. `concurrency=100` in NVIDIA’s sample plot). Similar plots can use ITL, e2e_latency, or TPS_per_user on X.
