---
source: https://developer.nvidia.com/blog/measuring-generative-ai-model-performance-using-nvidia-genai-perf-and-an-openai-compatible-api/
lang: en
fetched: 2026-08-31
---

# Measuring Generative AI Model Performance Using NVIDIA GenAI-Perf and an OpenAI-Compatible API

NVIDIA offers tools like Perf Analyzer and Model Analyzer to assist machine learning engineers with measuring and balancing the trade-off between latency and throughput, crucial for optimizing ML inference performance. Model Analyzer has been embraced by leading organizations such as Snap to identify optimal configurations that enhance throughput and reduce deployment costs.

However, when serving generative AI models, particularly large language models (LLMs), performance measurement becomes more specialized.

For LLMs, latency and throughput metrics are further broken down into token-level metrics. The following list shows key metrics, but additional metrics such as request latency, request throughput, and number of output tokens are also important to track.

- Time to first token: Time between when a request is sent and when its first response is received, one value per request in the benchmark.

- Output token throughput: Total number of output tokens from benchmark divided by benchmark duration.

- Inter-token latency: Time between intermediate responses for a single request divided by the number of generated tokens of the latter response, one value per response per request in the benchmark. This metric is also known as time per output token.

When measuring LLMs, it is important to see results quickly and consistently across users and models. For many applications, time to first token is given the highest priority, followed by output token throughput and inter-token latency. However, a tool that can report all of these metrics can help define and measure what is most important for your specific system and use case.

Achieving optimal performance for LLM inference requires balancing these metrics effectively. There is a natural trade-off between output token throughput and inter-token latency: processing multiple user queries concurrently can increase throughput but may also lead to higher inter-token latency. Choosing the right balance to achieve total cost of ownership (TCO) savings can be challenging without specialized benchmarking tools tailored for generative AI.


Local figures (copyright remains with the original site; study copies):

![compare input output sequence length](../../../assets/nvidia/benchmarking/blog-genai-perf-openai/01-compare-input-output-sequence-length.jpg)

![compare time to first token](../../../assets/nvidia/benchmarking/blog-genai-perf-openai/02-compare-time-to-first-token.jpg)

## Introducing GenAI-Perf

The latest release of NVIDIA Triton now includes a new generative AI performance benchmarking tool, GenAI-Perf. This solution is designed to enhance the measurement and optimization of generative AI performance.

This solution empowers you in the following ways:

- Accurately measures the specific metrics crucial for generative AI, to determine optimal configurations that deliver peak performance and cost-effectiveness.

- Use industry-standard data sets such as OpenOrca and CNN_dailymail to evaluate model performance.

- Facilitate standardized performance evaluation across diverse inference engines through an OpenAI-compatible API.

GenAI-Perf serves as the default benchmarking tool for assessing performance across all NVIDIA generative AI offerings, including NVIDIA NIM, NVIDIA Triton Inference Server, and NVIDIA TensorRT-LLM. It facilitates easy comparisons among different serving solutions that support the OpenAI-compatible API.

In the following sections, we guide you through how GenAI-Perf can be used to measure the performance of models compatible with OpenAI endpoints.

## Currently supported endpoints

NVIDIA GenAI-Perf currently supports three OpenAI endpoint APIs:

- Chat

- Chat Completions

- Embeddings

New endpoints will be released as new model types become popular. GenAI-Perf is also open source and accepts community contributions.

## Running GenAI-Perf

For more information about getting started, see Installation in the GenAI-Perf GitHub repo. The easiest way to run GenAI-Perf is to install the latest Triton Inference Server SDK container. To do so, you can run the latest SDK container version on NVIDIA GPU Cloud or use version `YY.MM` which corresponds to the year and month (for example, 24.07 for July 2024).

To run the container, run the following command:

```

docker run -it --net=host --rm --gpus=all
nvcr.io/nvidia/tritonserver:YY.MM-py3-sdk

```

Then, you must get the server running. Each of the following examples has a different command to run that type of model.

Start the vLLM OpenAI server:

```

docker run -it --net=host --rm \
--gpus=all vllm/vllm-openai:latest \
--model gpt2 \
--dtype float16 --max-model-len 1024

```

The model changes with the endpoint:

- For chat and chat-completion, use gpt2.

- For embeddings, use `intfloat/e5-mistral-7b-instruct \`.

When the server is running, you can run GenAI-Perf with the GenAI-Perf command to get the results. You see results visually, which are also saved as CSV and JSON for easy parsing of the data. They are saved in the `/artifacts` folder with other generated artifacts, including graphs visualizing the model performance.

### Profiling OpenAI chat-compatible models

Run GenAI-Perf:

```

genai-perf \
  -m gpt2 \
  --service-kind openai \
  --endpoint-type chat \
  --tokenizer gpt2

```

Review the sample results (Table 1).

StatisticAvgMinMaxP99P90P75Request latency (ms)1679.30567.312929.262919.412780.702214.89Output sequence length453.43162.00784.00782.60744.00588.00Input sequence length318.48124.00532.00527.00488.00417.00Table 1. LLM metrics output from running GenAI -Perf to measure GPT2 performance for chat

The output shows LLM performance metrics such as request latency, output sequence length, and input sequence length computed from running GTP2 for chat.

- Output token throughput (per sec): 269.99

- Request throughput (per sec): 0.60

Review some of the generated graphs (Figure 1).

Figure 1. Running GenAI-Perf to compare input sequence length to output sequence length

### Profiling OpenAI chat completions-compatible models

Run GenAI-Perf:

```

genai-perf \
  -m gpt2 \
  --service-kind openai \
  --endpoint-type completions \
  --tokenizer gpt2 \
  --generate-plots

```

Review sample results (Table 2).

StatisticAvgMinMaxP99P90P75Request latency (ms)74.5630.0896.0893.4382.3474.81Output sequence length15.882.0017.0016.0016.0016.00Input sequence length311.6229.00570.00538.04479.40413.00Table 2. LLM metrics output from running GenAI-Perf to measure GPT2 performance for chat completion

- Output token throughput (per sec): 218.55

- Request throughput (per sec): 13.76

The output shows LLM performance metrics such as request latency, output sequence length, and input sequence length computed from running GTP2 for chat completion.

Review generated graphs (Figure 2).

Figure 2. Running GenAI-Perf to compare time to first token (TTFT) against the input sequence length

### Profiling OpenAI embeddings-compatible models

Create a compatible JSONL file with sample texts for embedding. You can generate this file with the following command on the Linux command line:

```

echo '{"text": "What was the first car ever driven?"}
{"text": "Who served as the 5th President of the United States of America?"}
{"text": "Is the Sydney Opera House located in Australia?"}
{"text": "In what state did they film Shrek 2?"}' > embeddings.jsonl

```

Run GenAI-Perf:

```

genai-perf \
-m intfloat/e5-mistral-7b-instruct \
--batch-size 2 \
--service-kind openai \
--endpoint-type embeddings \
--input-file embeddings.jsonl

```

Review the sample results (Table 3).

StatisticAvgMinMaxP99P90P75Request latency (ms)41.9628.16302.1955.2447.4642.57Table 3. Embedding metrics output from running GenAI-Perf to measure E5-Mistral-7b-Instruct performance

Request throughput (per sec): 23.78

## Conclusion

That is all there is to benchmarking your models with GenAI-Perf. You have everything you need to get started.

You can review the other CLI arguments to see how updating the inference parameters affects performance. For example, you can pass in different values for `--request-rate` to modify the number of requests sent per second. You can then see how those changes affect metrics like inter-token latency, request latency, and throughput.

GenAI-Perf is open source and available on GitHub.
