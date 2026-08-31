---
source: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/performance_tuning.html
lang: zh
fetched: 2026-08-31
---

# Triton：把训好的模型调到能上线

流程：

1. 后端是否支持？ONNX/TensorRT 可自动补全 config。否则 Python backend 或自写 C++。
2. **Perf Analyzer** 拿基线：`perf_analyzer -m my_model`。
3. **Model Analyzer** 搜 `config.pbtxt`（instance 数、dynamic batching、max_batch_size），把最优配置拷回模型仓库。
4. 再打一遍 Perf Analyzer。官方 densenet_onnx 例子：默认 168 infer/s → 调完 323（约 +92%），4 个 GPU instance + dynamic batching。

冷启动用 ModelWarmup。GPU 不明显加速时：框架 GPU 优化或转 TensorRT；有的模型更适合 CPU（OpenVINO）。

数字**跟机器绑定**，要在接近生产的硬件上 profile。
