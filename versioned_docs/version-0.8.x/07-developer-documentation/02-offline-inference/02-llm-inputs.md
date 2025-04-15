---
title: LLM Inputs
---

## vllm.inputs.PromptType

内部 API 的核心部分。


这表示具有类型参数「params」的「origin」类型的通用版本。这类别名有两种：用户定义的和特殊的。特殊的是 collections.abc 中关于内置容器合与容器中 ABCs 的封装。它们必须始终设置「name」。如果「inst」为 False，则该别名无法实例化，例如可以使用别名： Typing.List 和 Typing.Dict。

是 [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[str](https://docs.python.org/3/library/stdtypes.html#str), [TextPrompt](https://docs.vllm.ai/en/latest/dev/offline_inference/llm_inputs.html#vllm.inputs.TextPrompt), [TokensPrompt](https://docs.vllm.ai/en/latest/dev/offline_inference/llm_inputs.html#vllm.inputs.TokensPrompt), ExplicitEncoderDecoderPrompt] 的别名。

# class vllm.inputs.TextPrompt

[[source]](https://docs.vllm.ai/en/latest/_modules/vllm/inputs/data.html#TextPrompt)

基类： [TypedDict](https://typing-extensions.readthedocs.io/en/latest/index.html#typing_extensions.TypedDict)


文本提示的架构。

>prompt: [str](https://docs.python.org/3/library/stdtypes.html#str)
在传递到模型之前要标记化的输入文本。


>multi_modal_data: typing_extensions.NotRequired[MultiModalDataDict]
如果模型支持，可传递给模型的可选多模态数据。


>mm_processor_kwargs: typing_extensions.NotRequired[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]
将被转发到多模态输入映射器和处理器的可选多模态处理器关键字参数。请注意，如果有多种模态已为正在考虑的模型注册了映射器等，我们会尝试将多模态处理器关键字参数传递给它们中的每一个。


# class vllm.inputs.TokensPrompt

[[source]](https://docs.vllm.ai/en/latest/_modules/vllm/inputs/data.html#TokensPrompt)

基类：[TypedDict](https://typing-extensions.readthedocs.io/en/latest/index.html#typing_extensions.TypedDict)


标记化提示的架构。

>prompt_token_ids: [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]
一个要传递给模型的 token ID 列表。

>multi_modal_data: typing_extensions.NotRequired[MultiModalDataDict]
如果模型支持，可传递给模型的可选多模态数据。

>mm_processor_kwargs: typing_extensions.NotRequired[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]
将被转发至多模态输入映射器和处理器的可选多模态处理器关键参数。需要注意的是，如果有多种模态已经为正在考量的模型注册了映射器等相关内容，我们会尝试将这些多模态处理器关键参数传递给它们中的每一个。

