---
source: https://docs.nvidia.com/nim/benchmarking/llm/latest/overview.html
lang: en
fetched: 2026-08-30
---

# Overview — NVIDIA NIM LLMs Benchmarking

NVIDIA NIM LLMs Benchmarking 2.0.0

## Executive Summary

This guide helps large language model (LLM) application developers and system owners benchmark inference latency and throughput for NVIDIA NIM deployments. It covers key metrics, test parameters, and a hands-on walkthrough with AIPerf.

After reading this guide, you can answer:

- What metrics matter most for LLM inference latency and throughput?
- Which benchmarking tools are available, and how do their measurements differ?
- How do I use NVIDIA AIPerf to benchmark an LLM application?

This guide is organized as follows:

- **Metrics** — definitions for TTFT, ITL, TPS, RPS, and related metrics
- **Parameters and Best Practices** — concurrency, sequence lengths, and other test settings
- **Using AIPerf to Benchmark** — end-to-end AIPerf workflow with NVIDIA NIM
- **Benchmarking LoRA Models** — benchmarking multi-adapter LoRA deployments

## Introduction to LLM Inference Benchmarking

As LLM-based applications roll out across enterprises, teams need repeatable ways to compare serving solutions. Deployment cost depends on how many queries a system can handle while staying responsive. This guide focuses on performance measurement; accuracy evaluation is out of scope and should be validated separately for your use case.

You can benchmark LLM performance with general load-testing tools such as Locust and K6, or with LLM-specific clients such as NVIDIA AIPerf. These tools expose overlapping metrics but often define and calculate them differently. This guide clarifies those differences and walks through AIPerf, NVIDIA’s recommended benchmarking tool for generative AI inference.

Performance benchmarking and load testing answer different questions:

- **Load testing** (for example, with K6) simulates concurrent traffic to test scaling, autoscaling, network behavior, and resource limits.
- **Performance benchmarking** (for example, with AIPerf) measures model-level throughput, latency, and token-level behavior under controlled conditions.

This guide focuses on performance benchmarking—model efficiency, optimization, and configuration. Combine both approaches to understand end-to-end deployment behavior.

> **Note:** Server-side metrics are also available for NVIDIA NIM but are out of scope for this document. Refer to the Logging and Observability documentation.

## Background on How LLM Inference Works

Before interpreting benchmark metrics, understand the inference pipeline. For a typical LLM application, each request passes through these stages:

1. **Prompt submission** — the user provides a query.
2. **Queuing** — the request waits for an available inference slot.
3. **Prefill** — the model processes the full input prompt.
4. **Decode** — also known as generation, the model emits the response one token at a time.

Tokens are the basic unit of LLM text processing. Each model has a tokenizer that maps text to tokens. As a rough guide, one token is about 0.75 English words for many popular models.

Sequence lengths drive memory use and latency:

- **Input Sequence Length (ISL)** — tokens in the prompt, including system instructions, chat history, chain-of-thought content, and RAG context.
- **Output Sequence Length (OSL)** — tokens the model generates.
- **Context length** — total tokens visible at each generation step (input plus output generated so far), bounded by the model’s maximum context window.

For a deeper dive, refer to [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/).

Streaming returns partial outputs as tokens are generated. This improves perceived responsiveness in chat applications. In non-streaming mode, the client receives the full response after generation completes.
