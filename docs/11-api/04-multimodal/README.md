---
title: 多模态支持
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM 通过 `vllm.multimodal` 包提供对多模态模型的实验性支持。

多模态输入可以与文本和 token 提示一起传递给[支持的模型](https://vllm.hyper.ai/docs/models/supported_models)，通过 `vllm.inputs.PromptType` 中的 `multi_modal_data` 字段传递。

想要添加自己的多模态模型？请按照[此处](https://vllm.hyper.ai/docs/contributing/model/multimodal)列出的说明操作。

## 模块内容

**vllm.multimodal.MULTIMODAL_REGISTRY = <vllm.multimodal.registry.MultiModalRegistry object>**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L101)

全局的 `MultiModalRegistry` 被模型运行器用于根据目标模型分派数据处理。

> **另请参阅** >[多模态数据处理](https://vllm.hyper.ai/docs/design/mm_processing)

## 子模块

- [输入定义](https://vllm.hyper.ai/docs/api/multimodal/inputs)
- [数据解析](https://vllm.hyper.ai/docs/api/multimodal/parse)
- [数据处理](https://vllm.hyper.ai/docs/api/multimodal/processing)
- [内存分析](https://vllm.hyper.ai/docs/api/multimodal/profiling)
- [注册表](https://vllm.hyper.ai/docs/api/multimodal/registry)
