---
title: 使用 CoreWeave 的 Tensorizer 加载模型
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM 支持使用 [CoreWeave 的 Tensorizer](https://docs.coreweave.com/coreweave-machine-learning-and-ai/inference/tensorizer) 加载模型。vLLM 模型张量可以被序列化到磁盘、HTTP/HTTPS 端点或 S3 端点，并在运行时极快地直接反序列化到 GPU，从而显著缩短 Pod 启动时间并减少 CPU 内存使用。同时，Tensorizer 还支持张量加密。

有关 CoreWeave 的 Tensorizer 的更多信息，请参阅 [CoreWeave 的 Tensorizer 文档](https://github.com/coreweave/tensorizer)。有关如何序列化 vLLM 模型、以及将 Tensorizer 与 vLLM 结合使用的通用指南，请参阅 [vLLM 示例脚本](https://docs.vllm.ai/en/stable/getting_started/examples/offline_inference/tensorize_vllm_model.html)。

> **注意：**
> 请注意，使用此功能需要安装 tensorizer，您可以通过运行 pip install vllm[tensorizer] 来完成安装。
