---
title: 注册表 (Registry)
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

## 模块内容

**_class_ vllm.multimodal.registry.ProcessingInfoFactory(_\*args_, _\*\*kwargs_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L37)

从上下文中构建一个 `MultiModalProcessor` 实例。

**_class_ vllm.multimodal.registry.DummyInputsBuilderFactory(_\*args_, _\*\*kwargs_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L47)

从上下文中构建一个 `BaseDummyInputsBuilder` 实例。

**_class_ vllm.multimodal.registry.MultiModalProcessorFactory(_\*args_, _\*\*kwargs_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L56)

从上下文中构建一个 `MultiModalProcessor` 实例。

**_class_ vllm.multimodal.registry.MultiModalRegistry**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L101)

一个根据模型分派数据处理的注册表。

**get\_max\_tokens\_per\_item\_by\_modality(_model\_config: ModelConfig_) → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L98)[#](https://docs.vllm.ai/en/stable/api/multimodal/registry.html#vllm.multimodal.registry.MultiModalRegistry.get_max_tokens_per_item_by_modality "Permalink to this definition")

根据底层模型配置，从每种模式中获取每个数据项的最大 token 数。

**get\_max\_tokens\_per\_item\_by\_nonzero\_modality(_model\_config: ModelConfig_) → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L123)[#](https://docs.vllm.ai/en/stable/api/multimodal/registry.html#vllm.multimodal.registry.MultiModalRegistry.get_max_tokens_per_item_by_nonzero_modality "Permalink to this definition")

根据底层模型配置，从每个模态中获取每个数据项的最大令牌数，不包括用户通过 \_mm\_per\_prompt 显式禁用的模态。

> **注意**
> 
> 目前仅在 V1 中直接用于分析模型的内存使用情况。

**get\_max\_tokens\_by\_modality(_model\_config: ModelConfig_) → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L145)[#](https://docs.vllm.ai/en/stable/api/multimodal/registry.html#vllm.multimodal.registry.MultiModalRegistry.get_max_tokens_by_modality "Permalink to this definition")

从每种模态中获取用于分析模型的内存使用情况的最大 token 数。

有关更多详细信息，请参阅 `MultiModalPlugin.get_max_multimodal_tokens()`。

**get\_max\_multimodal\_tokens(_model\_config: ModelConfig_) → [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L163)[#](https://docs.vllm.ai/en/stable/api/multimodal/registry.html#vllm.multimodal.registry.MultiModalRegistry.get_max_multimodal_tokens "Permalink to this definition")

获取用于分析模型内存使用情况的多模态 token 的最大数量。

有关更多详细信息，请参阅 `MultiModalPlugin.get_max_multimodal_tokens()`。

**get\_mm\_limits\_per\_prompt(_model\_config: ModelConfig_) → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L182)[#](https://docs.vllm.ai/en/stable/api/multimodal/registry.html#vllm.multimodal.registry.MultiModalRegistry.get_mm_limits_per_prompt "Permalink to this definition")

获取模型类的每个提示允许的每种模态的最大多模态输入实例数。

**register\_processor(_processor: [MultiModalProcessorFactory](https://docs.vllm.ai/en/stable/api/multimodal/registry.html#vllm.multimodal.registry.MultiModalProcessorFactory "vllm.multimodal.registry.MultiModalProcessorFactory")\[\_I\]_, _\*_, _info: [ProcessingInfoFactory](https://docs.vllm.ai/en/stable/api/multimodal/registry.html#vllm.multimodal.registry.ProcessingInfoFactory "vllm.multimodal.registry.ProcessingInfoFactory")\[\_I\]_, _dummy\_inputs: [DummyInputsBuilderFactory](https://docs.vllm.ai/en/stable/api/multimodal/registry.html#vllm.multimodal.registry.DummyInputsBuilderFactory "vllm.multimodal.registry.DummyInputsBuilderFactory")\[\_I\]_)**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L197)[#](https://docs.vllm.ai/en/stable/api/multimodal/registry.html#vllm.multimodal.registry.MultiModalRegistry.register_processor "Permalink to this definition")

将多模态处理器注册到模型类。处理器是惰性构造的，因此应该传递一个工厂方法。

当模型接收到多模态数据时，将调用提供的函数以将数据转换为模型输入的字典。

**create\_processor(_model\_config: ModelConfig_, _\*_, _tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | TokenizerBase | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _disable\_cache: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [BaseMultiModalProcessor](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.BaseMultiModalProcessor "vllm.multimodal.processing.BaseMultiModalProcessor")\[[BaseProcessingInfo](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.BaseProcessingInfo "vllm.multimodal.processing.BaseProcessingInfo")\]**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L246)[#](https://docs.vllm.ai/en/stable/api/multimodal/registry.html#vllm.multimodal.registry.MultiModalRegistry.create_processor "Permalink to this definition")

为特定模型和分词器创建多模态处理器。

**get\_decoder\_dummy\_data(_model\_config: ModelConfig_, _seq\_len: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_, _mm\_counts: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [DummyDecoderData](https://docs.vllm.ai/en/stable/api/multimodal/profiling.html#vllm.multimodal.profiling.DummyDecoderData "vllm.multimodal.profiling.DummyDecoderData")**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L275)[#](https://docs.vllm.ai/en/stable/api/multimodal/registry.html#vllm.multimodal.registry.MultiModalRegistry.get_decoder_dummy_data "Permalink to this definition")

创建虚拟数据以分析模型的内存使用情况。

模型由 `model_config` 标识。

**get\_encoder\_dummy\_data(_model\_config: ModelConfig_, _seq\_len: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_, _mm\_counts: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [DummyEncoderData](https://docs.vllm.ai/en/stable/api/multimodal/profiling.html#vllm.multimodal.profiling.DummyEncoderData "vllm.multimodal.profiling.DummyEncoderData")**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L299)[#](https://docs.vllm.ai/en/stable/api/multimodal/registry.html#vllm.multimodal.registry.MultiModalRegistry.get_encoder_dummy_data "Permalink to this definition")

创建虚拟数据以分析模型的内存使用情况。

模型由 `model_config` 标识。
