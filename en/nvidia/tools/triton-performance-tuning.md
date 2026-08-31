---
source: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/performance_tuning.html
lang: en
fetched: 2026-08-31
---

# Triton: Deploy and Tune a Trained Model

Flow:

1. Backend compatible? ONNX/TensorRT can autocomplete config. Else Python backend or custom C++.
2. **Perf Analyzer** baseline: `perf_analyzer -m my_model` (sanity + throughput/latency).
3. **Model Analyzer** searches `config.pbtxt` (instance count, dynamic batching, max_batch_size). Copy the winning config back to the model repo.
4. Re-run Perf Analyzer. NVIDIA densenet_onnx example: default 168 infer/s → tuned 323 infer/s (~+92%) with 4 GPU instances + dynamic batching.

Cold start: ModelWarmup on load. Weak GPU speedup: framework GPU opts or convert to TensorRT; some models belong on CPU (OpenVINO).

Results are **machine-specific**. Profile on hardware that matches production.
