---
title: 输入处理
---


每个模型都可以通过 [INPUT_REGISTRY](https://docs.vllm.ai/en/latest/dev/input_processing/model_inputs_index.html#vllm.inputs.INPUT_REGISTRY) 和 [MULTIMODAL_REGISTRY](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MULTIMODAL_REGISTRY) 覆盖 vLLM 的输入处理管道 [input_processing_pipeline](https://docs.vllm.ai/en/latest/dev/input_processing/input_processing_pipeline.html#input-processing-pipeline) 的部分内容。


目前，这种机制仅在多模态模型中用于预处理[多模态](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#multi-modality)输入数据以及输入提示，但如有需要，也可以扩展到仅处理文本的语言模型。


## 指南

[输入处理管道](https://docs.vllm.ai/en/latest/dev/input_processing/input_processing_pipeline.html)


## 模块内容

### LLM 引擎输入

>vllm.inputs.DecoderOnlyInputs
 `TokenInputs` 的别名


### 注册

>vllm.inputs.INPUT_REGISTRY =  <vllm.inputs.registry.InputRegistry object>
[LLMEngine](https://docs.vllm.ai/en/latest/dev/engine/llm_engine.html#vllm.LLMEngine) 使用全局 `InputRegistry` 来根据目标模型调度数据处理。

另见 [Input Processing Pipeline](https://docs.vllm.ai/en/latest/dev/input_processing/input_processing_pipeline.html#input-processing-pipeline)。


>class vllm.inputs.registry.DummyDataFactory(*args, **kwargs) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/inputs/registry.html#DummyDataFactory)

基类：[Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol)


>class vllm.inputs.registry.InputContext(model_config: ModelConfig) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/inputs/registry.html#InputContext)

包含有关可用于修改输入的模型信息。


>get_hf_config(hf_config_type: [Type](https://docs.python.org/3/library/typing.html#typing.Type)[C] = PretrainedConfig) → C 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/inputs/registry.html#InputContext.get_hf_config)

获取模型的 HuggingFace 配置(`transformers.PretrainedConfig`)，同时检查其类型。


**报错：**

[TypeError](https://docs.python.org/3/library/exceptions.html#TypeError) – 如果模型不属于指定类型。


>get_hf_image_processor_config() → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/inputs/registry.html#InputContext.get_hf_image_processor_config)

获取模型的 HuggingFace 图像处理器配置。


>model_config: ModelConfig 
模型的配置。


>vllm.inputs.registry.InputProcessor 
预处理模型的输入。

别名：[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[InputContext](https://docs.vllm.ai/en/latest/dev/input_processing/model_inputs_index.html#vllm.inputs.registry.InputContext), [LLMInputs](https://docs.vllm.ai/en/latest/dev/input_processing/model_inputs_index.html#vllm.inputs.LLMInputs)], [LLMInputs](https://docs.vllm.ai/en/latest/dev/input_processing/model_inputs_index.html#vllm.inputs.LLMInputs)]


>class vllm.inputs.registry.InputRegistry 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/inputs/registry.html#InputRegistry)

一个根据目标模型调度数据处理的注册表。


>create_input_processor(model_config: ModelConfig) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/inputs/registry.html#InputRegistry.create_input_processor)

为特定模型创建输入处理器（请参阅 `_process_input()`）。


>dummy_data_for_profiling(model_config: ModelConfig, seq_len: [int](https://docs.python.org/3/library/functions.html#int), mm_registry: [MultiModalRegistry](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalRegistry), is_encoder_data: [bool](https://docs.python.org/3/library/functions.html#bool) = False) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[SequenceData, MultiModalDataDict | [None](https://docs.python.org/3/library/constants.html#None)] 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/inputs/registry.html#InputRegistry.dummy_data_for_profiling)


创建虚拟数据来分析模型的内存使用情况。


该模型由 `model_config` 标识。


另见 [Enabling Multimodal Inputs](https://docs.vllm.ai/en/latest/models/enabling_multimodal_inputs.html#enabling-multimodal-inputs)。


**注意：**

这应该在调用了 `init_mm_limits_per_prompt()` 之后进行。


>process_input(model_config: ModelConfig, inputs: TokenInputs) →  TokenInputs
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/inputs/registry.html#InputRegistry.process_input)


将输入处理器应用于模型输入的实例。


该模型由 `model_config` 标识。


另见 [Input Processing Pipeline](https://docs.vllm.ai/en/latest/dev/input_processing/input_processing_pipeline.html#input-processing-pipeline)。


>register_dummy_data(factory: [DummyDataFactory](https://docs.vllm.ai/en/latest/dev/input_processing/model_inputs_index.html#vllm.inputs.registry.DummyDataFactory)) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/inputs/registry.html#InputRegistry.register_dummy_data)


为模型类注册一个虚拟数据工厂。


在进行内存分析期间，将调用提供的函数来创建要输入到模型中的虚拟数据。由此生成的内存使用量应该是模型在推理时使用的内存使用量的上限。


>**register_dummy_encoder_data****(*****factory:***[DummyDataFactory](https://docs.vllm.ai/en/latest/dev/input_processing/model_inputs_index.html#vllm.inputs.registry.DummyDataFactory)**)**
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/inputs/registry.html#InputRegistry.register_dummy_encoder_data)

为模型类注册一个虚拟编码器数据工厂。


这与 `register_dummy_data()`，类似，但针对编码器输入。


>register_input_processor(processor: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[InputContext](https://docs.vllm.ai/en/latest/dev/input_processing/model_inputs_index.html#vllm.inputs.registry.InputContext), [LLMInputs](https://docs.vllm.ai/en/latest/dev/input_processing/model_inputs_index.html#vllm.inputs.LLMInputs)], [LLMInputs](https://docs.vllm.ai/en/latest/dev/input_processing/model_inputs_index.html#vllm.inputs.LLMInputs)]) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/inputs/registry.html#InputRegistry.register_input_processor)

为模型类注册一个输入处理器。


在模型的每个输入上调用所提供的函数。这发生在 `map_input()` 之前。


另见 [Input Processing Pipeline](https://docs.vllm.ai/en/latest/dev/input_processing/input_processing_pipeline.html#input-processing-pipeline)。

