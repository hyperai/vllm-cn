---
title: 内存分析
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

## 模块内容

**_class_ vllm.multimodal.profiling.ProcessorInputs(_prompt\_text: str, mm\_data: ~collections.abc.Mapping\[str, ~typing.Any | list\[typing.Any\]\], hf\_processor\_mm\_kwargs: ~collections.abc.Mapping\[str, object\] \= <factory\>_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/profiling.py#L23)

表示 `vllm.multimodal.processing.BaseMultiModalProcessor.apply()` 的关键词参数。

**_class_ vllm.multimodal.profiling.DummyEncoderData(_prompt\_token\_ids: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/profiling.py#L37)

为分析多模态模型而构造虚拟数据的抽象基类。

**prompt\_token\_ids_: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/profiling.py#L35)[#](https://docs.vllm.ai/en/stable/api/multimodal/profiling.html#vllm.multimodal.profiling.DummyEncoderData.prompt_token_ids "Permalink to this definition")

字段编号 0 的别名。

**_class_ vllm.multimodal.profiling.DummyDecoderData(_prompt\_token\_ids: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_, _multi\_modal\_data: [MultiModalKwargs](https://docs.vllm.ai/en/stable/api/multimodal/inputs.html#vllm.multimodal.inputs.MultiModalKwargs "vllm.multimodal.inputs.MultiModalKwargs")_, _multi\_modal\_placeholders: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[PlaceholderRange](https://docs.vllm.ai/en/stable/api/multimodal/inputs.html#vllm.multimodal.inputs.PlaceholderRange "vllm.multimodal.inputs.PlaceholderRange")\]\]_)**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/profiling.py#L41)[#](https://docs.vllm.ai/en/stable/api/multimodal/profiling.html#vllm.multimodal.profiling.DummyDecoderData "Permalink to this definition")

用于分析的虚拟数据。

**prompt\_token\_ids_: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/profiling.py#L41)[#](https://docs.vllm.ai/en/stable/api/multimodal/profiling.html#vllm.multimodal.profiling.DummyDecoderData.prompt_token_ids "Permalink to this definition")

字段编号 0 的别名。

  **multi\_modal\_data_: [MultiModalKwargs](https://docs.vllm.ai/en/stable/api/multimodal/inputs.html#vllm.multimodal.inputs.MultiModalKwargs "vllm.multimodal.inputs.MultiModalKwargs")**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/profiling.py#L41)[#](https://docs.vllm.ai/en/stable/api/multimodal/profiling.html#vllm.multimodal.profiling.DummyDecoderData.multi_modal_data "Permalink to this definition")

字段编号 1 的别名。

**multi\_modal\_placeholders_: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[PlaceholderRange](https://docs.vllm.ai/en/stable/api/multimodal/inputs.html#vllm.multimodal.inputs.PlaceholderRange "vllm.multimodal.inputs.PlaceholderRange")\]\]**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/profiling.py#L41)[#](https://docs.vllm.ai/en/stable/api/multimodal/profiling.html#vllm.multimodal.profiling.DummyDecoderData.multi_modal_placeholders "Permalink to this definition")

字段编号 2 的别名。

**_class_ vllm.multimodal.profiling.BaseDummyInputsBuilder(_info: \_I_)**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/profiling.py#L52)[#](https://docs.vllm.ai/en/stable/api/multimodal/profiling.html#vllm.multimodal.profiling.BaseDummyInputsBuilder "Permalink to this definition")

抽象基类，用于构造虚拟数据以分析多模态模型。

**get\_dummy\_text(_mm\_counts: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_) → [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/profiling.py#L64)[#](https://docs.vllm.ai/en/stable/api/multimodal/profiling.html#vllm.multimodal.profiling.BaseDummyInputsBuilder.get_dummy_text "Permalink to this definition")

构建与 `mm_counts` 对应的文本输入。

**get\_dummy\_mm\_data(_seq\_len: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_, _mm\_counts: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_) → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [Any](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[Any](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")\]\]**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/profiling.py#L81)[#](https://docs.vllm.ai/en/stable/api/multimodal/profiling.html#vllm.multimodal.profiling.BaseDummyInputsBuilder.get_dummy_mm_data "Permalink to this definition")

构建多模式输入，该输入在处理后会产生最大可能的占位符标记数。

**get\_dummy\_processor\_inputs(_seq\_len: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_, _mm\_counts: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_) → [ProcessorInputs](https://docs.vllm.ai/en/stable/api/multimodal/profiling.html#vllm.multimodal.profiling.ProcessorInputs "vllm.multimodal.profiling.ProcessorInputs")**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/profiling.py#L92)[#](https://docs.vllm.ai/en/stable/api/multimodal/profiling.html#vllm.multimodal.profiling.BaseDummyInputsBuilder.get_dummy_processor_inputs "Permalink to this definition")

构建输入，该输入在处理后会产生最大可能的占位符 token 数。

**_class_ vllm.multimodal.profiling.MultiModalProfiler(_processor: [BaseMultiModalProcessor](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.BaseMultiModalProcessor "vllm.multimodal.processing.BaseMultiModalProcessor")\[\_I\]_)**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/profiling.py#L143)[#](https://docs.vllm.ai/en/stable/api/multimodal/profiling.html#vllm.multimodal.profiling.MultiModalProfiler "Permalink to this definition")

包含用于为多模态模型运行内存分析的代码。
