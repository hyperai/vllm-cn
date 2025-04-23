---
title: 注册表 (Registry)
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

## 模块内容

**\*class\*\*\***vllm.multimodal.registry.\***\*ProcessingInfoFactory\*\***(\***\*\*\*\*\*\*\***args**\***,**\*\***\***\*\***kwargs**\***)\*\*

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L37)

从上下文中构建一个 `MultiModalProcessor` 实例。

**\*class\*\*\***vllm.multimodal.registry.\***\*DummyInputsBuilderFactory\*\***(\***\*\*\*\*\*\*\***args**\***,**\*\***\***\*\***kwargs**\***)\*\*

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L47)

从上下文中构建一个 `BaseDummyInputsBuilder` 实例。

**\*class\*\*\***vllm.multimodal.registry.\***\*MultiModalProcessorFactory\*\***(\***\*\*\*\*\*\*\***args**\***,**\*\***\***\*\***kwargs**\***)\*\*

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L56)

从上下文中构建一个 `MultiModalProcessor` 实例。

**\*class\*\*\***vllm.multimodal.registry.\***\*MultiModalRegistry\*\***(\***\*\*\*\*\*\***,**\***plugins:**_[Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)_**[MultiModalPlugin]**\*\***=**\* \***DEFAULT_PLUGINS**\***)\*\*

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L101)

一个根据模型分派数据处理的注册表。

**register_plugin\*\***(**\***plugin: MultiModalPlugin**\***)\***\*→**[None](https://docs.python.org/3/library/constants.html#None)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L124)

注册一个多模态插件，以便 vLLM 可以识别它。

**register_input_mapper\*\***(**\***data*type_key:**\*[str](https://docs.python.org/3/library/stdtypes.html#str)**,**\***mapper:\*\**[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)_**[[InputContext,\***[object](https://docs.python.org/3/library/functions.html#object)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[object](https://docs.python.org/3/library/functions.html#object)**\*]],**_[MultiModalKwargs](https://docs.vllm.ai/en/latest/api/multimodal/inputs.html#vllm.multimodal.inputs.MultiModalKwargs)_**] |**_[None](https://docs.python.org/3/library/constants.html#None) \_**=**\* \***None**\*\*\*)\*\*

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L146)

为特定模态注册一个输入映射器到模型类。

有关更多详细信息，请参阅 `MultiModalPlugin.register_input_mapper()`。

**register_image_input_mapper\*\***(**\***mapper:**_[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)_**[[InputContext,***[object](https://docs.python.org/3/library/functions.html#object)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[object](https://docs.python.org/3/library/functions.html#object)***]],**_[MultiModalKwargs](https://docs.vllm.ai/en/latest/api/multimodal/inputs.html#vllm.multimodal.inputs.MultiModalKwargs)_**] |**_[None](https://docs.python.org/3/library/constants.html#None) _**=**\* \***None**\***)\*\*

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L158)

为图像数据注册一个输入映射器到模型类。

有关更多详细信息，请参阅 `MultiModalPlugin.register_input_mapper()`。

**map_input\*\***(**\***model*config: ModelConfig**\***,**\***data:\*\**[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)_**[**_[str](https://docs.python.org/3/library/stdtypes.html#str)_**,**_[Any](https://docs.python.org/3/library/typing.html#typing.Any)_**|**_[list](https://docs.python.org/3/library/stdtypes.html#list)_**[**_[Any](https://docs.python.org/3/library/typing.html#typing.Any)_**]]**\***,**\***mm_processor_kwargs:**_[dict](https://docs.python.org/3/library/stdtypes.html#dict)_**[**_[str](https://docs.python.org/3/library/stdtypes.html#str)_**,**_[Any](https://docs.python.org/3/library/typing.html#typing.Any)_**] |**_[None](https://docs.python.org/3/library/constants.html#None) \_**=**\* \***None**\***)\***\*→\*\*[MultiModalKwargs](https://docs.vllm.ai/en/latest/api/multimodal/inputs.html#vllm.multimodal.inputs.MultiModalKwargs)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L169)

将输入映射器应用于传递给模型的数据。

每个模态的数据都会传递给对应的插件，然后该插件通过为该模型注册的输入映射器 (input mapper)，将数据转换为关键字参数。

有关更多详细信息，请参阅 `MultiModalPlugin.map_input()`。

> **注意**
> 应在 `init_mm_limits_per_prompt()` 之后调用此方法。

**create_input_mapper\*\***(**\***model_config: ModelConfig**\***)\*\*

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L212)

为特定模型创建一个输入映射器（参见 `map_input()`）。

**register_max_multimodal_tokens\*\***(**\***data*type_key:**\*[str](https://docs.python.org/3/library/stdtypes.html#str)**,**\***max_mm_tokens:\*\**[int](https://docs.python.org/3/library/functions.html#int)_**|**_[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)_**[[InputContext],**_[int](https://docs.python.org/3/library/functions.html#int)_**] |**_[None](https://docs.python.org/3/library/constants.html#None) \_**=**\* \***None**\*\*\*)\*\*

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L227)

为某个模型类别注册一个最大 token 数，这个 token 数对应于属于特定模态的单个多模态数据实例传递给语言模型的 token 数量。

**register_max_image_tokens\*\***(**\***max*mm_tokens:\*\**[int](https://docs.python.org/3/library/functions.html#int)_**|**_[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)_**[[InputContext],**_[int](https://docs.python.org/3/library/functions.html#int)_**] |**_[None](https://docs.python.org/3/library/constants.html#None) \_**=**\* \***None**\*\*\*)\*\*

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L240)

为某个模型类别注册一个最大图像 token 数，这个 token 数对应于单张图像传递给语言模型的 token 数量。

**get_max_tokens_per_item_by_modality\*\***(**\***model_config: ModelConfig**\***)\***\*→**[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[int](https://docs.python.org/3/library/functions.html#int)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L250)

根据底层模型配置获取每个模态的每个数据项的最大 token 数。

**get_max_tokens_per_item_by_nonzero_modality\*\***(**\***model_config: ModelConfig**\***)\***\*→**[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[int](https://docs.python.org/3/library/functions.html#int)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L273)

根据底层模型配置获取每个模态的每个数据项的最大 token 数，排除用户通过 `limit_mm_per_prompt` 显式禁用的模态。

> **注意**
> 目前仅在 V1 中直接用于分析模型的内存使用情况。

**get_max_tokens_by_modality\*\***(**\***model_config: ModelConfig**\***)\***\*→**[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[int](https://docs.python.org/3/library/functions.html#int)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L295)

获取每个模态的最大 token 数，用于分析模型的内存使用情况。

有关更多详细信息，请参阅 `MultiModalPlugin.get_max_multimodal_tokens()` 。

> **注意**
> 应在 `init_mm_limits_per_prompt()` 之后调用此方法。

**get_max_multimodal_tokens\*\***(**\***model_config: ModelConfig**\***)\***\*→**[int](https://docs.python.org/3/library/functions.html#int)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L316)

获取用于分析模型内存使用的多模态 token 的最大数量。

有关更多详细信息，请参阅 `MultiModalPlugin.get_max_multimodal_tokens()`。

> **注意**
> 应在 `init_mm_limits_per_prompt()` 之后调用此方法。

**init_mm_limits_per_prompt\*\***(**\***model_config: ModelConfig**\***)\***\*→**[None](https://docs.python.org/3/library/constants.html#None)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L328)

初始化每个提示允许的每个模态的多模态输入实例的最大数量。

**get_mm_limits_per_prompt\*\***(**\***model_config: ModelConfig**\***)\***\*→**[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[int](https://docs.python.org/3/library/functions.html#int)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L364)

获取每个提示允许的每个模态的多模态输入实例的最大数量。

> **注意**
> 应在 `init_mm_limits_per_prompt()` 之后调用此方法。

**register_processor\*\***(**\***processor:**_[MultiModalProcessorFactory](https://docs.vllm.ai/en/latest/api/multimodal/registry.html#vllm.multimodal.registry.MultiModalProcessorFactory)_**[_I]**\***,\***\*\*\*\*\*\***,**\***info:**_[ProcessingInfoFactory](https://docs.vllm.ai/en/latest/api/multimodal/registry.html#vllm.multimodal.registry.ProcessingInfoFactory)_**[_I]**\***,**\***dummy*inputs:\*\**[DummyInputsBuilderFactory](https://docs.vllm.ai/en/latest/api/multimodal/registry.html#vllm.multimodal.registry.DummyInputsBuilderFactory)\_**[_I]**\*\*\*)\*\*

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L385)

为模型类注册一个多模态处理器。处理器是延迟构建的，因此应传递一个工厂方法。

当模型接收到多模态数据时，调用提供的函数将数据转换为模型输入的字典。

> **另请参阅** >[多模态数据处理](https://docs.vllm.ai/en/latest/design/mm_processing.html#mm-processing)

**has_processor\*\***(**\***model_config: ModelConfig**\***)\***\*→**[bool](https://docs.python.org/3/library/functions.html#bool)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L427)

测试是否为特定模型定义了多模态处理器。

> **另请参阅** >[多模态数据处理](https://docs.vllm.ai/en/latest/design/mm_processing.html#mm-processing)

**create_processor\*\***(**\***model*config: ModelConfig**\***,**\***tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | TokenizerBase**\***,\***\*\*\*\*\*\***,**\***disable_cache:\*\**[bool](https://docs.python.org/3/library/functions.html#bool)_**|**_[None](https://docs.python.org/3/library/constants.html#None) \_**=**\* \***None**\***)\***\*→**[BaseMultiModalProcessor](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.BaseMultiModalProcessor)**[**[BaseProcessingInfo](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.BaseProcessingInfo)**]\*\*

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L436)

为特定模型和分词器创建一个多模态处理器。

> **另请参阅** >[多模态数据处理](https://docs.vllm.ai/en/latest/design/mm_processing.html#mm-processing)
