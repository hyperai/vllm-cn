---
title: 编写单元测试
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

本页解释了如何编写单元测试以验证您的模型实现。

## 必需测试

这些测试是将您的 PR 合并到 vLLM 库中所必需的。没有它们，您的 PR 的 CI 将失败。

### 模型加载

在 [tests/models/registry.py](https://github.com/vllm-project/vllm/blob/main/tests/models/registry.py) 中包含您的模型的示例 HuggingFace 仓库。这启用了一个加载虚拟权重的单元测试，以确保可以在 vLLM 中初始化模型。

> **重要**
> 
> 每个部分中的模型列表应按字母顺序维护。

> **提示**
> 
> 如果您的模型需要 HF Transformers 的开发版本，您可以设置 `min_transformers_version` 以跳过 CI 中的测试，直到模型发布。

## 可选测试

这些测试是将您的 PR 合并到 vLLM 库中的可选测试。通过这些测试可以更有信心地认为您的实现是正确的，并有助于避免未来的回归。

### 模型正确性

这些测试将 vLLM 的模型输出与 [HF Transformers](https://github.com/huggingface/transformers) 进行比较。您可以在 [tests/models](https://github.com/vllm-project/vllm/tree/main/tests/models) 的子目录下添加新测试。

#### 生成模型

对于[生成模型](https://docs.vllm.ai/en/latest/models/generative_models.html#generative-models)，有两个级别的正确性测试，如 [tests/models/utils.py](https://github.com/vllm-project/vllm/blob/main/tests/models/utils.py) 中所定义的：

- 完全正确性 (`check_outputs_equal`)：vLLM 输出的文本应与 HF 输出的文本完全匹配。
- 对数概率相似性 (`check_logprobs_close`)：vLLM 输出的对数概率应在 HF 输出的 top-k 对数概率中，反之亦然。

#### 池化模型

对于[池化模型](https://docs.vllm.ai/en/latest/models/pooling_models.html#pooling-models)，我们只需检查余弦相似性，如 [tests/models/embedding/utils.py](https://github.com/vllm-project/vllm/blob/main/tests/models/embedding/utils.py) 中所定义的。

### 多模态处理

#### 通用测试

将您的模型添加到 [tests/models/multimodal/processing/test_common.py](https://github.com/vllm-project/vllm/blob/main/tests/models/multimodal/processing/test_common.py) 中，验证以下输入组合是否产生相同的输出：

- 文本 + 多模态数据
- Token + 多模态数据
- 文本 + 缓存的多模态数据
- Token + 缓存的多模态数据

#### 模型特定测试

您可以在 [tests/models/multimodal/processing](https://github.com/vllm-project/vllm/tree/main/tests/models/multimodal/processing) 下添加一个新文件，以运行仅适用于您的模型的测试。

例如，如果您的模型的 HF 处理器接受用户指定的关键字参数，您可以验证关键字参数是否正确应用，例如在 [tests/models/multimodal/processing/test_phi3v.py](https://github.com/vllm-project/vllm/blob/main/tests/models/multimodal/processing/test_phi3v.py) 中。
