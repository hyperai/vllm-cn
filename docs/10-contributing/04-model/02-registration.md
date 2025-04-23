---
title: 将模型注册到 vLLM
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM 依赖模型注册表来确定如何运行每个模型。预注册的架构列表可以在[此处](https://docs.vllm.ai/en/latest/models/supported_models.html#supported-models)找到。

如果您的模型不在此列表中，您必须将其注册到 vLLM。本页提供了如何执行此操作的详细说明。

## 内置模型

要将模型直接添加到 vLLM 库中，首先 fork 我们的 [GitHub 仓库](https://github.com/vllm-project/vllm)，然后[从源码构建](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#build-from-source)。这将使您能够修改代码库并测试您的模型。

在您实现模型后（参见[教程](https://docs.vllm.ai/en/latest/contributing/model/basic.html#new-model-basic)），将其放入 [vllm/model_executor/models](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models) 目录中。然后，将您的模型类添加到 [vllm/model_executor/models/registry.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py) 中的 `_VLLM_MODELS` 中，以便在导入 vLLM 时自动注册。最后，更新我们的[支持模型列表](https://docs.vllm.ai/en/latest/models/supported_models.html#supported-models)以推广您的模型！

> **重要\*\***信息\*\*
> 每个部分中的模型列表应按字母顺序维护。

## 外部模型

您可以使用插件加载外部模型，而无需修改 vLLM 代码库。

> **另请参阅** >[vLLM 的插件系统](https://docs.vllm.ai/en/latest/design/plugin_system.html#plugin-system)

要注册模型，请使用以下代码：

```plain
from vllm import ModelRegistry
from your_code import YourModelForCausalLM
ModelRegistry.register_model("YourModelForCausalLM", YourModelForCausalLM)
```

如果您的模型导入了初始化 CUDA 的模块，请考虑延迟导入，避免出现类似 `RuntimeError: Cannot re-initialize CUDA in forked subprocess` 的错误：

```plain
from vllm import ModelRegistry


ModelRegistry.register_model("YourModelForCausalLM", "your_code:YourModelForCausalLM")
```

> **重要\*\***信息\*\*
> 如果您的模型是多模态模型，请确保模型类实现了 `SupportsMultiModal` 接口。有关更多信息，请阅读[此处](https://docs.vllm.ai/en/latest/contributing/model/multimodal.html#supports-multimodal)。

> **注意**
> 虽然您可以直接将这些代码片段放入使用 `vllm.LLM` 的脚本中，但推荐的方式是将这些片段放入 vLLM 插件中。这确保了与各种 vLLM 功能（如分布式推理和 API 服务器）的兼容性。
