---
title: 多模态
---


vLLM 通过 [vllm.multimodal](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#module-vllm.multimodal) 包为多模态模型提供实验性支持。


多模态输入可以通过 [vllm.inputs.PromptInputs](https://docs.vllm.ai/en/latest/dev/offline_inference/llm_inputs.html#vllm.inputs.PromptInputs) 中的 `multi_modal_data` 字段将文本和 token 提示一起传到[支持的模型](https://docs.vllm.ai/en/latest/models/supported_models.html#supported-vlms)中。


目前，vLLM 仅内置对图像数据的支持。您可以按照[该指南](https://docs.vllm.ai/en/latest/dev/multimodal/adding_multimodal_plugin.html#adding-multimodal-plugin)扩展 vLLM 处理其他模态。


想要添加您自己的多模态模型吗？请按照[此处](https://docs.vllm.ai/en/latest/models/enabling_multimodal_inputs.html#enabling-multimodal-inputs)列出的说明进行操作。

## 

## 指南

[添加多模态插件](https://docs.vllm.ai/en/latest/dev/multimodal/adding_multimodal_plugin.html)

## 模块内容

### 注册

>vllm.multimodal.MULTIMODAL_REGISTRY = <vllm.multimodal.registry.MultiModalRegistry object> 

模型运行器 (model runners) 使用全局 [MultiModalRegistry](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalRegistry) 根据数据的模态和目标模型来调度数据处理。


另见 [Input Processing Pipeline](https://docs.vllm.ai/en/latest/dev/input_processing/input_processing_pipeline.html#input-processing-pipeline)。


>class vllm.multimodal.MultiModalRegistry(*, plugins: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[MultiModalPlugin](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalPlugin)] = DEFAULT_PLUGINS) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/registry.html#MultiModalRegistry)

一个将数据处理分派给每种模态的 [MultiModalPlugin](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalPlugin) 的注册表。


>create_input_mapper(model_config: ModelConfig)  
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/registry.html#MultiModalRegistry.create_input_mapper)

为特定模型创建 1 个输入映射器（请参阅 [map_input()](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalRegistry.map_input)）。


>get_max_multimodal_tokens(model_config: ModelConfig) → [int](https://docs.python.org/3/library/functions.html#int) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/registry.html#MultiModalRegistry.get_max_multimodal_tokens)

获取用于分析模型内存使用情况的最大多模态 tokens 数量。

请参阅 [MultiModalPlugin.get_max_multimodal_tokens()](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalPlugin.get_max_multimodal_tokens) 了解更多详细信息。


**注意：**

这应该在调用了 [init_mm_limits_per_prompt()](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalRegistry.init_mm_limits_per_prompt) 之后进行。


>get_mm_limits_per_prompt(model_config: ModelConfig) → [Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)] 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/registry.html#MultiModalRegistry.get_mm_limits_per_prompt)

获取一个模型类针对每个模态，在每个提示中所允许的最大多模态输入实例数量。

**注意：**

这应该在调用了 [init_mm_limits_per_prompt()](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalRegistry.init_mm_limits_per_prompt) 之后进行。


>init_mm_limits_per_prompt(model_config: ModelConfig) → [None](https://docs.python.org/3/library/constants.html#None) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/registry.html#MultiModalRegistry.init_mm_limits_per_prompt)

为每种模态设置每个提示所允许的多模态输入实例的最大数量，针对特定模型类别进行初始化。


>map_input(model_config: ModelConfig, data: [MultiModalDataBuiltins](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalDataBuiltins) | [Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [object](https://docs.python.org/3/library/functions.html#object) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[object](https://docs.python.org/3/library/functions.html#object)]], mm_processor_kwargs: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [MultiModalInputs](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalInputs) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/registry.html#MultiModalRegistry.map_input)


将输入映射器应用于传递给模型的数据。


相应插件接收来自每种模态的数据，并通过为该模型注册的输入映射器，将这些数据转换成关键字参数。


有关更多详细信息，请参阅 [MultiModalPlugin.map_input()](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalPlugin.map_input)。


**注意：**

这应该在调用了 [init_mm_limits_per_prompt()](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalRegistry.init_mm_limits_per_prompt) 之后进行。


>register_image_input_mapper(mapper: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[InputContext](https://docs.vllm.ai/en/latest/dev/input_processing/model_inputs_index.html#vllm.inputs.registry.InputContext), [object](https://docs.python.org/3/library/functions.html#object) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[object](https://docs.python.org/3/library/functions.html#object)]], [MultiModalInputs](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalInputs)] | [None](https://docs.python.org/3/library/constants.html#None) = None) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/registry.html#MultiModalRegistry.register_image_input_mapper)

为图像数据注册一个输入映射器到模型类别。


请参阅 [MultiModalPlugin.register_input_mapper()](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalPlugin.register_input_mapper) 了解更多详细信息。


>register_input_mapper(data_type_key: [str](https://docs.python.org/3/library/stdtypes.html#str), mapper: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[InputContext](https://docs.vllm.ai/en/latest/dev/input_processing/model_inputs_index.html#vllm.inputs.registry.InputContext), [object](https://docs.python.org/3/library/functions.html#object) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[object](https://docs.python.org/3/library/functions.html#object)]], [MultiModalInputs](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalInputs)] | [None](https://docs.python.org/3/library/constants.html#None) = None) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/registry.html#MultiModalRegistry.register_input_mapper)

为特定模态注册一个输入映射器到模型类别。


请参阅 [MultiModalPlugin.register_input_mapper()](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalPlugin.register_input_mapper) 了解更多详细信息。


>register_max_image_tokens(max_mm_tokens: [int](https://docs.python.org/3/library/functions.html#int) | [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[InputContext](https://docs.vllm.ai/en/latest/dev/input_processing/model_inputs_index.html#vllm.inputs.registry.InputContext)], [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/registry.html#MultiModalRegistry.register_max_image_tokens)

为特定模型类别注册最高数量的图像 token，这些标记对应于单个图像，并被传递给语言模型。


>register_max_multimodal_tokens(data_type_key: [str](https://docs.python.org/3/library/stdtypes.html#str), max_mm_tokens: [int](https://docs.python.org/3/library/functions.html#int) | [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[InputContext](https://docs.vllm.ai/en/latest/dev/input_processing/model_inputs_index.html#vllm.inputs.registry.InputContext)], [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/registry.html#MultiModalRegistry.register_max_multimodal_tokens)

注册一个模型类别中，传递给语言模型的、对应于特定模态下单一多模态数据实例的最大 token 数量。


>register_plugin(plugin: [MultiModalPlugin](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalPlugin)) → [None](https://docs.python.org/3/library/constants.html#None) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/registry.html#MultiModalRegistry.register_plugin)

注册一个多模态插件，使得 vLLM 可以识别它。


另见 [Adding a Multimodal Plugin](https://docs.vllm.ai/en/latest/dev/multimodal/adding_multimodal_plugin.html#adding-multimodal-plugin)。


### 基础类 (Base Classes)

>vllm.multimodal.NestedTensors 
内部 API 的核心部分。


这表示类型参数「origin」的一个通用版本，它包含了类型参数「params」。这些别名分为 2 种类型：用户定义的和特殊的。特殊别名是用于内置集合和 collections.abc 中的 ABCs 的包装器。这些特殊别名必须始终设置「name」。如果「inst」设置为 False，则无法实例化该别名，例如 Typing.List 和 Typing.Dict 就是这种情况。


别名是：[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedTensors](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.NestedTensors)], [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)], [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)]


>vllm.multimodal.BatchedTensorInputs 
内部 API 的核心部分。


这表示类型参数「origin」的一个通用版本，它包含了类型参数「params」。这些别名分为 2 种：用户定义的和特殊的。特殊别名是用于内置集合和 collections.abc 中的 ABCs 的包装器。这些特殊别名必须始终设置「name」。如果「inst」设置为 False，则无法实例化该别名，例如 Typing.List 和 Typing.Dict 就是这种情况。


别名为：[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedTensors](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.NestedTensors)], [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)], [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)]]


>final class vllm.multimodal.MultiModalDataBuiltins(*args, **kwargs) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/base.html#MultiModalDataBuiltins)

基类 (Bases)：[dict](https://docs.python.org/3/library/stdtypes.html#dict)


由 vLLM 预定义的模态类型。


>audio: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float)] | [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), [int](https://docs.python.org/3/library/functions.html#int) | [float](https://docs.python.org/3/library/functions.html#float)]] 
输入音频项和相应的采样率。


>image: [Image](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[Image](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image)] 
输入图像。


>vllm.multimodal.MultiModalDataDict 
内部API的核心部分。


这表示类型参数「origin」的一个通用版本，它包含了类型参数「params」。这些别名分为 2 种：用户定义的和特殊的。特殊别名是用于内置集合和 collections.abc 中的 ABCs 的包装器。这些特殊别名必须始终设置「name」。如果「inst」设置为 False，则无法实例化该别名，例如 Typing.List 和 Typing.Dict 就是这种情况。


别名为：[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[MultiModalDataBuiltins](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalDataBuiltins), [Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[object](https://docs.python.org/3/library/functions.html#object), [List](https://docs.python.org/3/library/typing.html#typing.List)[[object](https://docs.python.org/3/library/functions.html#object)]]]]


>class vllm.multimodal.MultiModalInputs(dict=None, /, **kwargs) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/base.html#MultiModalInputs)

基类 (Bases)：_MultiModalInputsBase


一个表示传递给 [forward()](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward) 方法的关键字参数的字典。


>static batch(inputs_list: [List](https://docs.python.org/3/library/typing.html#typing.List)[[MultiModalInputs](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalInputs)]) → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedTensors](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.NestedTensors)] | [List](https://docs.python.org/3/library/typing.html#typing.List)[[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)] | [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)] | [List](https://docs.python.org/3/library/typing.html#typing.List)[[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)] | [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)] 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/base.html#MultiModalInputs.batch)


将多个输入一起批处理到 1 个字典中。


生成的字典具有与输入相同的键。如果每个输入的对应值是一个张量并且它们都具有相同的形状，那么输出值就是一个单一的批处理张量；否则，输出值是 1 个包含每个输入的原始值的列表。


>class vllm.multimodal.MultiModalPlugin 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/base.html#MultiModalPlugin)

基类 (Bases)：[ABC](https://docs.python.org/3/library/abc.html#abc.ABC)


定义特定模态的数据处理逻辑的基类。


具体来说，我们采用注册模式来根据所使用的模型分派数据处理（考虑到不同模型可能会以不同方式处理相同数据）。这个注册机制反过来被 [MultiModalRegistry](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalRegistry) 使用，它在更高层次上发挥作用（即数据的模态）。


另见 [Adding a Multimodal Plugin](https://docs.vllm.ai/en/latest/dev/multimodal/adding_multimodal_plugin.html#adding-multimodal-plugin)。


>abstract get_data_key() → [str](https://docs.python.org/3/library/stdtypes.html#str) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/base.html#MultiModalPlugin.get_data_key)

获取与模态对应的数据键。


>get_max_multimodal_tokens(model_config: ModelConfig) → [int](https://docs.python.org/3/library/functions.html#int) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/base.html#MultiModalPlugin.get_max_multimodal_tokens)


获取用于分析模型内存使用情况的最大多模态 token 数量。


如果此注册不适用于该模型，则返回 0。


该模型通过 `model_config` 标识。


另见 [Enabling Multimodal Inputs](https://docs.vllm.ai/en/latest/models/enabling_multimodal_inputs.html#enabling-multimodal-inputs)。


>map_input(model_config: ModelConfig, data: [object](https://docs.python.org/3/library/functions.html#object) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[object](https://docs.python.org/3/library/functions.html#object)]) → [MultiModalInputs](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalInputs) 

[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/base.html#MultiModalPlugin.map_input)


使用为该模型注册的输入映射器将数据转换为模型输入的字典。


该模型通过 `model_config` 标识。


**报错：**

[TypeError](https://docs.python.org/3/library/exceptions.html#TypeError) – 如果不支持数据类型。


另见

* [Input Processing Pipeline](https://docs.vllm.ai/en/latest/dev/input_processing/input_processing_pipeline.html#input-processing-pipeline)

* [Enabling Multimodal Inputs](https://docs.vllm.ai/en/latest/models/enabling_multimodal_inputs.html#enabling-multimodal-inputs)


>register_input_mapper(mapper: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[InputContext](https://docs.vllm.ai/en/latest/dev/input_processing/model_inputs_index.html#vllm.inputs.registry.InputContext), [object](https://docs.python.org/3/library/functions.html#object) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[object](https://docs.python.org/3/library/functions.html#object)]], [MultiModalInputs](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalInputs)] | [None](https://docs.python.org/3/library/constants.html#None) = None) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/base.html#MultiModalPlugin.register_input_mapper)

将 1 个输入映射器注册到模型类。


当模型接收到与此插件提供的模态相匹配的输入数据时（请参阅 [get_data_key()](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalPlugin.get_data_key)），会调用提供的函数来将数据转换为模型输入的字典。


如果没有提供，则使用默认的输入映射器。


另见 

* [Input Processing Pipeline](https://docs.vllm.ai/en/latest/dev/input_processing/input_processing_pipeline.html#input-processing-pipeline)

* [Enabling Multimodal Inputs](https://docs.vllm.ai/en/latest/models/enabling_multimodal_inputs.html#enabling-multimodal-inputs)


>register_max_multimodal_tokens(max_mm_tokens: [int](https://docs.python.org/3/library/functions.html#int) | [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[InputContext](https://docs.vllm.ai/en/latest/dev/input_processing/model_inputs_index.html#vllm.inputs.registry.InputContext)], [int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/base.html#MultiModalPlugin.register_max_multimodal_tokens)

为特定模型类别注册一个单一多模态数据实例所对应的最大 token 数量，这些 token 将被传递给语言模型。


如果没有提供，则使用默认的计算方式。


另见 [Enabling Multimodal Inputs](https://docs.vllm.ai/en/latest/models/enabling_multimodal_inputs.html#enabling-multimodal-inputs)。


### 图像类

>class vllm.multimodal.image.ImagePlugin 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/image.html#ImagePlugin)

基类 (Bases)：[MultiModalPlugin](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalPlugin)


图像数据插件。


>get_data_key() → [str](https://docs.python.org/3/library/stdtypes.html#str) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/multimodal/image.html#ImagePlugin.get_data_key)


获取与模态对应的数据键。

