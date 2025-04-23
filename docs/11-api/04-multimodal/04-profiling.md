---

title: 内存分析

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


## 模块内容

***class*****vllm.multimodal.profiling.****ProcessorInputs****(*****prompt_text: str, mm_data: ~collections.abc.Mapping[str, ~typing.Any | list[typing.Any]], hf_processor_mm_kwargs: ~collections.abc.Mapping[str, object] = <factory>*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/profiling.py#L23)

表示 `vllm.multimodal.processing.BaseMultiModalProcessor.apply()` 的关键词参数。


***class*****vllm.multimodal.profiling.****BaseDummyInputsBuilder****(*****info: _I*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/profiling.py#L37)

为分析多模态模型而构造虚拟数据的抽象基类。

***abstract*****get_dummy_processor_inputs****(*****seq_len:***[int](https://docs.python.org/3/library/functions.html#int)**,*****mm_counts:***[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[int](https://docs.python.org/3/library/functions.html#int)***]*****)****→**[ProcessorInputs](https://docs.vllm.ai/en/latest/api/multimodal/profiling.html#vllm.multimodal.profiling.ProcessorInputs)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/profiling.py#L48)

在处理后构建输入，结果位于 `self.info.get_mm_max_tokens_per_item()` 占位符 token 中。


***class*****vllm.multimodal.profiling.****MultiModalProfiler****(*****processor:***[BaseMultiModalProcessor](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.BaseMultiModalProcessor)***[_I]*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/profiling.py#L91)

包含运行多模态模型的内存分析的代码。


