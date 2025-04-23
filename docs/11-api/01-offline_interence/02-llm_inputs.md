---

title: LLM Inputs

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


**vllm.inputs.****PromptType**

[[source]](https://github.com/vllm-project/vllm/blob/main/#L1588)

`Union`[`str`, `TextPrompt`, `TokensPrompt`, `ExplicitEncoderDecoderPrompt`] 的别名

***class*****vllm.inputs.****TextPrompt**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/inputs/data.py#L17)

Bases: `TypedDict`

文本提示的模式。


**prompt*****:***[str](https://docs.python.org/3/library/stdtypes.html#str)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/inputs/data.py#L17)

传递给模型之前需要分词处理的输入文本。


**multi_modal_data*****:***[NotRequired](https://docs.python.org/3/library/typing.html#typing.NotRequired)***[MultiModalDataDict]***

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/inputs/data.py#L17)

可选的多模态数据，如果模型支持，可以传递给模型。


**mm_processor_kwargs*****:***[NotRequired](https://docs.python.org/3/library/typing.html#typing.NotRequired)***[***[dict](https://docs.python.org/3/library/stdtypes.html#dict)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[Any](https://docs.python.org/3/library/typing.html#typing.Any)***]]***

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/inputs/data.py#L17)

可选的多模态处理器参数，将传递给多模态输入映射器和处理器。注意，如果多个模态为当前模型注册了映射器等，会尝试将`mm_processor_kwargs`传递给每个模态。


***class*****vllm.inputs.****TokensPrompt**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/inputs/data.py#L38)

Bases: `TypedDict`

分词后提示的模式。


**prompt_token_ids*****:***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***]***

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/inputs/data.py#L38)

传递给模型的 token ID 列表。


**token_type_ids*****:***[NotRequired](https://docs.python.org/3/library/typing.html#typing.NotRequired)***[***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***]]***

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/inputs/data.py#L38)

传递给交叉编码器模型的 token 类型 ID 列表。


**multi_modal_data*****:***[NotRequired](https://docs.python.org/3/library/typing.html#typing.NotRequired)***[MultiModalDataDict]***

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/inputs/data.py#L38)

可选的多模态数据，如果模型支持，可以传递给模型。


**mm_processor_kwargs*****:***[NotRequired](https://docs.python.org/3/library/typing.html#typing.NotRequired)***[***[dict](https://docs.python.org/3/library/stdtypes.html#dict)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[Any](https://docs.python.org/3/library/typing.html#typing.Any)***]]***

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/inputs/data.py#L38)

可选的多模态处理器参数，将传递给多模态输入映射器和处理器。注意，如果多个模态为当前模型注册了映射器等，会尝试将`mm_processor_kwargs`传递给每个模态。


