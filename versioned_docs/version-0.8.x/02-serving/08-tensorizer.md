---
title: 使用 CoreWeave 的张量器加载模型
---


vLLM 支持使用 [CoreWeave's Tensorizer](https://docs.coreweave.com/coreweave-machine-learning-and-ai/inference/tensorizer) 加载模型。已序列化到磁盘、HTTP/HTTPS 端点或 S3 端点的 vLLM 模型张量可以在运行时非常快速地直接反序列化到 GPU，从而显著缩短 Pod 启动时间并减少 CPU 内存使用。张量加密也是支持的。


有关 CoreWeave 的 Tensorizer 的更多信息，请参阅 [CoreWeave's Tensorizer 文档](https://github.com/coreweave/tensorizer)。有关序列化 vLLM 模型的更多信息，以及将 Tensorizer 与 vLLM 结合使用的一般使用指南，请参阅 [vLLM 示例脚本](https://docs.vllm.ai/en/stable/getting_started/examples/tensorize_vllm_model.html)。

