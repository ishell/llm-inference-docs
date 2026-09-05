---
source: https://docs.nvidia.com/nim/benchmarking/llm/latest/benchmarking-lora.html
lang: en
fetched: 2026-08-30
---

# Benchmarking LoRA Models

Parameter-Efficient Fine-Tuning (PEFT) methods fine-tune large pretrained models cheaply. NIM supports LoRA. You can load and deploy multiple LoRA adapters. Follow the PEFT guide to load Hugging Face or NeMo adapters and pass the adapter directory to NIM via an environment variable.

After adapters are loaded, query a LoRA model like the base model by replacing the model ID:

```bash
curl -X 'POST' \
  'http://0.0.0.0:8000/v1/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
"model": "llama3-8b-instruct-lora_vhf-math-v1",
"prompt": "John buys 10 packs of magic cards. Each pack has 20 cards and 1/4 of those cards are uncommon. How many uncommon cards did he get?",
"max_tokens": 128
}'
```

Benchmark with AIPerf by passing adapter IDs with `-m`. `--model-selection-strategy {round_robin,random}` controls how adapters are chosen.

```bash
export INPUT_SEQUENCE_LENGTH=200
export OUTPUT_SEQUENCE_LENGTH=200
export CONCURRENCY=10
export REQUEST_COUNT=$(($CONCURRENCY * 3))
export MODEL=meta/llama-3.1-8b-instruct
aiperf profile \
   -m $MODEL \
   --endpoint-type chat \
   --streaming \
   -u localhost:8000 \
   --synthetic-input-tokens-mean $INPUT_SEQUENCE_LENGTH \
   --synthetic-input-tokens-stddev 0 \
   --concurrency $CONCURRENCY \
   --request-count $REQUEST_COUNT \
   --warmup-request-count 10 \
   --output-tokens-mean $OUTPUT_SEQUENCE_LENGTH \
   --extra-inputs max_tokens:$OUTPUT_SEQUENCE_LENGTH \
   --extra-inputs min_tokens:$OUTPUT_SEQUENCE_LENGTH \
   --extra-inputs ignore_eos:true \
   --tokenizer meta-llama/llama-3.1-8b-instruct \
   --artifact-dir artifact/ISL${INPUT_SEQUENCE_LENGTH}_OSL${OUTPUT_SEQUENCE_LENGTH}/CON${CONCURRENCY}
```

(To test multiple adapters, pass several IDs after `-m`, e.g. `-m adapter1 adapter2`, plus `--model-selection-strategy`.)

## Best Practices for Multi-LoRA Benchmarking

Depends on model size, adapter config, and load:

- **Base model:** 8B and 70B both work as LoRA bases. Smaller models often excel at classification; larger models at reasoning. LoRA lets you fine-tune a 70B model on a single H100 with 4-bit quantization.
- **Adapters:** Rank trades accuracy vs batching. Common ranks: 8, 16, 32, 64. Operators can standardize rank for better batching.
- **Output length:** set `ignore_eos` so generation continues to `max_tokens` (consistent OSL without task-specific data).
- **System load:** concurrency should match real usage and stay within effective batching. For an 8B model on one GPU, up to 250 concurrent clients is a realistic upper bound in NVIDIA’s guidance.
- **Task type:** cover generative and non-generative. ISL 200–2000 and OSL 1–2000 spans classification, summarization, translation, and code generation.
