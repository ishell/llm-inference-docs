---
source: https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/
lang: en
fetched: 2026-08-31
---

# LLM Inference Benchmarking: Performance Tuning with TensorRT-LLM

This is the third post in the large language model latency-throughput benchmarking series, which aims to instruct developers on how to benchmark LLM inference with TensorRT-LLM. See LLM Inference Benchmarking: Fundamental Concepts for background knowledge on common metrics for benchmarking and parameters. And read LLM Inference Benchmarking Guide: NVIDIA GenAI-Perf and NIM for tips on using GenAI-Perf and NVIDIA NIM for your applications. 

It’s important to consider inference performance when deploying, integrating, or benchmarking any large language model (LLM) framework. You need to be sure to tune your chosen framework and its features so it delivers on the performance metrics that are important to your application.  

TensorRT-LLM, NVIDIA’s open-source AI inference engine, allows you to deploy models with its native benchmarking and serving tools, and has a wide array of features you can tune against. In this post, we’ll provide a practical guide on how to tune a model with `trtllm-bench` and then deploy using `trtllm-serve`.

## How to benchmark with trtllm-bench

`trtllm-bench` is TensorRT-LLM’s Python-based utility for directly benchmarking models without the overhead of a full inference deployment. It makes it simple to quickly generate insights into model performance. `trtllm-bench` internally sets up the engine with optimal settings that generally provide good performance.

### Set up your GPU environment

Benchmarking begins with a properly configured GPU environment. To restore your GPUs to their default settings, run:

```

sudo nvidia-smi -rgc
sudo nvidia-smi -rmc

```

To query your GPU’s maximum use:

```

nvidia-smi -q -d POWER

```

If you’d like to set a specific power limit (or to set max), run:

```

nvidia-smi -i <gpu_id> -pl <wattage>

```

For more detail, see the trtllm-bench documentation.

### Prepare a dataset

You can prepare a synthetic dataset by using `prepare_dataset` or create a dataset of your own by using the format specified in our documentation. For a custom dataset, you can format a JSON Lines (jsonl) file with a payload configured on each line. An example of a single dataset entry is below:

```

{"task_id": 1, "prompt": "Generate infinitely: This is the song that never ends, it goes on and on", "output_tokens": 128}

```

For the purposes of this post, we provide example output based on a synthetic dataset with an ISL/OSL of 128/128.

### Run benchmarks

To run benchmarks using `trtllm-bench`, you can use the t`rtllm-bench throughput` subcommand. Running a benchmark using the PyTorch flow, you simply need to run the following command in an environment with TensorRT-LLM installed:

```

trtllm-bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset dataset.jsonl \
   --tp 1 \
   --backend pytorch \
   --report_json results.json
   --streaming \
   --concurrency $CONCURRENCY

```

The `throughput` command will automatically pull the checkpoint from HuggingFace (if not cached) and bootstrap TRT-LLM with the PyTorch flow. Results will be saved to `results.json` and printed to the terminal as follows once the run completes:

Note: This is only a sample of the output and does not represent performance claims.

```

===========================================================
= PYTORCH BACKEND
===========================================================
Model:                  meta-llama/Llama-3.1-8B-Instruct
Model Path:             None
TensorRT-LLM Version:   0.21.0rc0
Dtype:                  bfloat16
KV Cache Dtype:         None
Quantization:           None

===========================================================
= REQUEST DETAILS
===========================================================
Number of requests:             100
Number of concurrent requests:  94.6050
Average Input Length (tokens):  128.0000
Average Output Length (tokens): 128.0000
===========================================================
= WORLD + RUNTIME INFORMATION
===========================================================
TP Size:                1
PP Size:                1
EP Size:                None
Max Runtime Batch Size: 3840
Max Runtime Tokens:     7680
Scheduling Policy:      GUARANTEED_NO_EVICT
KV Memory Percentage:   90.00%
Issue Rate (req/sec):   1.0526E+15

===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Request Throughput (req/sec):                     86.5373
Total Output Throughput (tokens/sec):             11076.7700
Total Token Throughput (tokens/sec):              22153.5399
Total Latency (ms):                               1155.5715
Average request latency (ms):                     1093.2284
Per User Output Throughput [w/ ctx] (tps/user):   117.1544
Per GPU Output Throughput (tps/gpu):              11076.7700
Average time-to-first-token [TTFT] (ms):   162.6706
Average time-per-output-token [TPOT] (ms): 7.3272
Per User Output Speed (tps/user):          137.1475

-- Per-Request Time-per-Output-Token [TPOT] Breakdown (ms)

[TPOT] MINIMUM: 6.6450
[TPOT] MAXIMUM: 8.1306
[TPOT] AVERAGE: 7.3272
[TPOT] P50    : 7.6079
[TPOT] P90    : 8.1246
[TPOT] P95    : 8.1289
[TPOT] P99    : 8.1306

-- Per-Request Time-to-First-Token [TTFT] Breakdown (ms)

[TTFT] MINIMUM: 93.9210
[TTFT] MAXIMUM: 232.4339
[TTFT] AVERAGE: 162.6706
[TTFT] P50    : 159.7857
[TTFT] P90    : 220.0530
[TTFT] P95    : 226.9148
[TTFT] P99    : 232.4339

-- Per-Request Generation Throughput [GTPS] Breakdown (tps/user)

[GTPS] MINIMUM: 122.9921
[GTPS] MAXIMUM: 150.4894
[GTPS] AVERAGE: 137.1475
[GTPS] P50    : 131.4444
[GTPS] P90    : 150.4112
[GTPS] P95    : 150.4606
[GTPS] P99    : 150.4894

-- Request Latency Breakdown (ms) -----------------------

[Latency] P50    : 1091.7905
[Latency] P90    : 1130.7200
[Latency] P95    : 1133.0074
[Latency] P99    : 1137.6817
[Latency] MINIMUM: 1050.1519
[Latency] MAXIMUM: 1137.6817
[Latency] AVERAGE: 1093.2284

===========================================================
= DATASET DETAILS
===========================================================
Dataset Path:         /workspace/benchmark_toolkit/synthetic_data.jsonl
Number of Sequences:  100

-- Percentiles statistics ---------------------------------

        Input              Output           Seq. Length
-----------------------------------------------------------
MIN:   128.0000           128.0000           256.0000
MAX:   128.0000           128.0000           256.0000
AVG:   128.0000           128.0000           256.0000
P50:   128.0000           128.0000           256.0000
P90:   128.0000           128.0000           256.0000
P95:   128.0000           128.0000           256.0000
P99:   128.0000           128.0000           256.0000
===========================================================

```

### Analyze performance results

When running the command above, the primary statistics are displayed under the `PERFORMANCE OVERVIEW` section. Before we get into the details, here’s some useful terminology:

- `Output` in the context of the overview means all output tokens generated (including context tokens)

- `Total Token` means the total sequence length generated (ISL+OSL)

- `Per user`, `TTFT`, and `TPOT` take the perspective that every request is a “user;” these statistics are then used to form a distribution.

For more in-depth explanations, see LLM Inference Benchmarking: Fundamental Concepts, the first post in this series.

```

===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Request Throughput (req/sec):                 	86.5373
Total Output Throughput (tokens/sec):         	11076.7700
Total Token Throughput (tokens/sec):          	22153.5399
Total Latency (ms):                           	1155.5715
Average request latency (ms):                 	1093.2284
Per User Output Throughput [w/ ctx] (tps/user):   117.1544
Per GPU Output Throughput (tps/gpu):          	11076.7700
Average time-to-first-token [TTFT] (ms):   162.6706
Average time-per-output-token [TPOT] (ms): 7.3272
Per User Output Speed (tps/user):      	137.1475

```

You’ll also notice that `trtllm-bench` reports the maximum number of tokens and batch size.

```

===========================================================
= WORLD + RUNTIME INFORMATION
===========================================================
TP Size:            	1
PP Size:            	1
EP Size:            	None
Max Runtime Batch Size: 3840
Max Runtime Tokens: 	7680

```

These have a particular meaning in the context of TensorRT-LLM:

- The maximum number of tokens refers to the maximum number of tokens the engine itself can handle in one batched iteration. This limit includes the sum of all input tokens for all context requests and a single token for the sum of all generation requests in the batch.

- The maximum batch size is the maximum number of requests allowed in a batch. Let’s say your iteration contains a request with context of length 128, four generation requests (total 132 tokens), and you’ve set the max tokens to 512 with a max batch size of five requests. In this case, your engine will cap at the batch size even though it hasn’t satisfied the max tokens.

When analyzing results, it is helpful to know your priorities. Some common questions:

- Are you aiming for a high per-user token throughput?

- Are you crunching large amounts of text and need the highest throughput possible?

- Do you want the first token to return quickly?

The tuning you settle on highly depends on the scenario you want to prioritize. For this post, let’s focus on optimizing the per-user experience. We want to prioritize the `Per User Output Speed` metric or the speed that tokens are returned to the user after the context phase has completed. With `trtllm-bench`, you can specify the maximum number of outstanding requests using `--concurrency`, which enables you to narrow down the number of users your system can support. 

This option is useful for producing several different curves, which are crucial when searching for latency and throughput targets. Here is a set of curves based on NVIDIA’s Llama-3.1 8B FP8 and Meta’s Llama-3.1 8B FP16 generated for a 128/128 ISL/OSL scenario. Let’s say that we want to utilize the system as much as possible, but we still want a user to experience about 50 tokens/second of output speed (about 20ms between tokens). In order to assess the tradeoff between GPU performance and user experience, you can plot the per-GPU output throughput against per-user output speed.

			
				
			
		Figure 1. A comparison of per-user output speed and per-gpu output throughput. Based on our criteria, we can see that per-GPU throughput improves as concurrency increases (better system utilization); however, as the system becomes less saturated, the per-user output speed increases (better experience).

In Figure 1, we can see that Llama-3.1 8B FP16 can only handle about 256 concurrent users with approximately 72 tokens/sec/user before violating our 50 tokens/sec/user constraint. However, if we look at the Llama-3.1 8B FP8 optimized checkpoint, we see that TensorRT-LLM can handle 512 concurrent users at approximately 66 tokens/sec/user. We can conclude that the quantized model is able to serve more users within the same budget simply by sweeping both models with `trtllm-bench`.

With this data, you can consider the following:

- If you would like to force the engine to 512 entries, you could set the maximum batch size to 512; however, this risks increasing the time-to-first-token (TTFT) if traffic to this instance exceeds 512 (any requests beyond the 512 are queued).

- You can assess quality of service scenarios and models with other datasets using `trtllm-bench` and plot a variety of metrics. The tool allows you to make value assessments based on your priorities by adjusting the command line in a simple-to-use manner.

Note: In this scenario, we only explore a single GPU model—if you have a model that requires multiple GPUs, you can configure `trtllm-bench` using the `--tp`, `--pp`, and `--ep` options to find the best sharded/data parallel configuration. Additionally, if you’re a developer and need advanced features, you can use the `--extra_llm_api_options` argument.

## How to serve a large language model with trtllm-serve

TensorRT-LLM offers the ability to easily stand up an OpenAI-compatible endpoint using the `trtllm-serve` command. You can use the tuning from `trtllm-bench` above in order to spin up a tuned server. Unlike the benchmark, `trtllm-serve` makes no assumptions on the configuration aside from general settings. To tune the server based on our maximum throughput results above, you would need to provide the following command based on the output above:

```

trtllm-serve serve nvidia/Llama-3.1-8B-Instruct-FP8 --backend pytorch --max_num_tokens 7680 --max_batch_size 3840 --tp_size 1 --extra_llm_api_options llm_api_options.yml

```

The `--extra_llm_api_options` provides a mechanism to directly change the settings at the LLM API level. In order to match the settings from the benchmark, you will need the following in your `llm_api_options.yml`:

```

cuda_graph_config:
    max_batch_size: 3840
    padding_enabled: true 

```

Once configured and run, you should see a status update that the server is running:

```

INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)

```

With the server running, you can now benchmark the model using GenAI-Perf (similar to the second blog in this series), or you can use our ported version of benchmark_serving.py. These can help you verify the performance of your tuned server configuration. In future releases, we plan to augment `trtllm-bench` to be able to spin up an optimized server for benchmarking.

## Get started with benchmarking and performance tuning for LLMs

With `trtllm-bench`, TensorRT-LLM provides an easy way to benchmark a variety of configurations, tunings, concurrency, and features. The settings from `trtllm-bench` are directly translatable to TensorRT-LLM’s native serving solution, `trtllm-serve`. It enables you to seamlessly port your performance tuning to an OpenAI-compatible deployment.

For more detailed information on performance, model-specific tuning, and tuning/benchmarking TensorRT-LLM, see the following resources:

- For a more in-depth understanding of the performance options available, see the Performance Tuning Guide.

- For `trtllm-bench` documentation, see our Performance Benchmarking page.

- To see how to profile TensorRT-LLM, the Performance Analysis covers using Nsight System to profile model execution.

- For a deeper dive into performance tuning for DeepSeek-R1, check out the TensorRT-LLM Performance Tuning Guide for DeepSeek-R1.

Check out the following resources:

- To learn more about how platform architecture can impact TCO beyond just FLOPS, you can read the blog post “NVIDIA DGX Cloud Introduces Ready-To-Use Templates to Benchmark AI Platform Performance.” See also the collection of Performance Benchmarking Recipes (ready-to-use templates) available for download on NGC here.

- Learn how to lower your cost per token and maximize AI models with The IT Leader’s Guide to AI Inference and Performance.
