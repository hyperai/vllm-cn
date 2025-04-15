---
title: 添加多模态插件
---


本文档将教您如何向 vLLM 添加一种新模态。


vLLM 中的每种模态均由 [MultiModalPlugin](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalPlugin) 表示，并注册到 [MULTIMODAL_REGISTRY](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MULTIMODAL_REGISTRY) 中。为了让 vLLM 识别新的模态类型，您必须创建一个新的插件，然后将其传递给 [register_plugin()](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalRegistry.register_plugin)。

本文档的其余部分详细介绍了如何定义自定义的 [MultiModalPlugin](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalPlugin)。


**注意：**

本文尚在编写中。


