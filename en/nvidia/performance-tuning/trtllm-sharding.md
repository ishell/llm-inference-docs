---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/deciding-model-sharding-strategy.html
lang: en
fetched: 2026-08-31
---

# Deciding Model Sharding Strategy

If the model does not fit on one GPU, how you shard it matters. Primer: `mastering-llm-techniques.md` (TP vs PP).


Local figures (copyright remains with the original site; study copies):

![Pipeline Parallel Vis](../../../assets/nvidia/performance-tuning/trtllm-sharding/01-Pipeline_Parallel_Vis.svg)

![Tensor Parallelism Vis](../../../assets/nvidia/performance-tuning/trtllm-sharding/02-Tensor_Parallelism_Vis.svg)

## Communication is the constraint

Sharding means GPUs exchange activations.

- **Pipeline parallel:** contiguous layer stacks per GPU. Communication is “send my outputs to the next GPU.”
- **Tensor parallel:** every layer is split. Each GPU holds a slice of every layer, so it needs an **All-Reduce** of the previous layer’s full outputs before the next layer can start. Heavier communication; smaller GEMMs per GPU.

If interconnects are fast (NVLink), smaller GEMMs often win. If they are slow (cross-node), PP usually wins.

Rules of thumb:

1. **Fits one GPU:** do not shard unless you have a specific reason. Best overhead is none.
2. **Fits one node:** start with TP if you have NVLink. If not, sanity-check PP. Measure.
3. **Multi-node:** intra-node links are usually much faster than inter-node. Start with **TP inside the node, PP between nodes**. Exception: Blackwell **NVL36 / NVL72** have multinode NVLink — TP is not bottlenecked by inter-node links while you stay inside that 36/72 GPU domain.

## How to set

`tensor_parallel_size * pipeline_parallel_size` must equal world size.

Two nodes × 16 GPUs: TP=8, PP=2:

```python
llm = LLM(
    model="/scratch/Llama-3.1-405B-Instruct",
    tensor_parallel_size=8,
    pipeline_parallel_size=2,
)
```

CLI: `convert_checkpoint.py --tp_size` and `--pp_size`. The official prose mistakenly repeats `--tp_size` for both; the command sample uses `--pp_size` for pipeline parallel.

```bash
python examples/models/core/llama/convert_checkpoint.py \
  --model_dir ./tmp/llama/405B/ \
  --output_dir ./tllm_checkpoint_16gpu_tp8_pp2 \
  --dtype float16 \
  --tp_size 8 \
  --pp_size 2
```

The rest of this handbook’s case study stays on one node: Llama-3.3-70B, TP=4, four NVLink H100s.
