---
title: vLLM 的插件系统
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

社区经常请求能够通过自定义功能扩展 vLLM。为了便于实现这一点，vLLM 包含了一个插件系统，允许用户在不修改 vLLM 代码库的情况下添加自定义功能。本文档解释了 vLLM 中插件的工作原理以及如何为 vLLM 创建插件。

## vLLM 中插件的工作原理

插件是用户注册的代码，vLLM 会执行这些代码。鉴于 vLLM 的架构（参见[架构概述](https://docs.vllm.ai/en/latest/design/arch_overview.html#arch-overview)），可能会涉及多个进程，尤其是在使用分布式推理和各种并行技术时。为了成功启用插件，vLLM 创建的每个进程都需要加载插件。这是通过 `vllm.plugins` 模块中的 [load_general_plugins](https://github.com/vllm-project/vllm/blob/c76ac49d266e27aa3fea84ef2df1f813d24c91c7/vllm/plugins/__init__.py#L16) 函数完成的。该函数在 vLLM 创建的每个进程开始工作之前被调用。

## vLLM 如何发现插件

vLLM 的插件系统使用标准的 Python `entry_points` 机制。该机制允许开发者在他们的 Python 包中注册函数，以供其他包使用。以下是一个插件的示例：

```plain
# inside `setup.py` file
# 在 `setup.py` 文件中
from setuptools import setup


setup(name='vllm_add_dummy_model',
      version='0.1',
      packages=['vllm_add_dummy_model'],
      entry_points={
          'vllm.general_plugins':
          ["register_dummy_model = vllm_add_dummy_model:register"]
      })


# inside `vllm_add_dummy_model.py` file
# 在 `vllm_add_dummy_model.py` 文件中
def register():
    from vllm import ModelRegistry


    if "MyLlava" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("MyLlava",
                                        "vllm_add_dummy_model.my_llava:MyLlava")
```

有关如何向包中添加入口点的更多信息，请查看[官方文档](https://setuptools.pypa.io/en/latest/userguide/entry_point.html)。

每个插件包含 3 个部分：

1. **插件组**：入口点组的名称。vLLM 使用入口点组 `vllm.general_plugins` 来注册通用插件。这是 `setup.py` 文件中 `entry_points` 的键。对于 vLLM 的通用插件，始终使用 `vllm.general_plugins`。

2. **插件名称**：插件的名称。这是 `entry_points` 字典中的值。在上面的示例中，插件名称是 `register_dummy_model`。可以通过 `VLLM_PLUGINS` 环境变量按名称过滤插件。要仅加载特定插件，请将 `VLLM_PLUGINS` 设置为插件名称。

3. **插件值**：在插件系统中注册的函数的完全限定名称。在上面的示例中，插件值是 `vllm_add_dummy_model:register`，它指的是 `vllm_add_dummy_model` 模块中名为 `register` 的函数。

## 支持的插件类型

- **通用插件**（组名为 `vllm.general_plugins`）：这些插件的主要用例是将自定义的、不在树中的模型注册到 vLLM 中。这是通过在插件函数中调用 `ModelRegistry.register_model` 来注册模型实现的。
- **平台插件**（组名为 `vllm.platform_plugins`）：这些插件的主要用例是将自定义的、不在树中的平台注册到 vLLM 中。当当前环境不支持该平台时，插件函数应返回 `None`，或者当平台受支持时返回平台类的完全限定名称。

## 编写插件指南

- **可重入性**：在入口点中指定的函数应该是可重入的，这意味着它可以被多次调用而不会导致问题。这是必要的，因为该函数可能会在某些进程中被多次调用。

## 兼容性保证

vLLM 保证文档化的插件接口（例如 `ModelRegistry.register_model`）将始终可供插件注册模型。然而，插件开发者有责任确保他们的插件与他们所针对的 vLLM 版本兼容。例如，`"vllm_add_dummy_model.my_llava:MyLlava"` 应该与插件所针对的 vLLM 版本兼容。在 vLLM 的开发过程中，模型的接口可能会发生变化。
