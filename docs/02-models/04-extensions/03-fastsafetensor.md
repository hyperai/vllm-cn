---
title: 使用 fastsafetensors 加载模型权重
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

使用 fastsafetensor 库可以通过利用 GPU 直接存储将模型权重加载到 GPU 内存。有关详细信息，请参阅 https://github.com/foundation-model-stack/fastsafetensors。要启用此功能，请将环境变量 USE_FASTSAFETENSOR 设置为 true。
