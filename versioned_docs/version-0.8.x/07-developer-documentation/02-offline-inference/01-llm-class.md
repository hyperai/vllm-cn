---
title: LLM 类
---

>class vllm.LLM(model: [str](https://docs.python.org/3/library/stdtypes.html#str), tokenizer: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, tokenizer_mode: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'auto', skip_tokenizer_init: [bool](https://docs.python.org/3/library/functions.html#bool) = False, trust_remote_code: [bool](https://docs.python.org/3/library/functions.html#bool) = False, tensor_parallel_size: [int](https://docs.python.org/3/library/functions.html#int) = 1, dtype: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'auto', quantization: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, revision: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, tokenizer_revision: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, seed: [int](https://docs.python.org/3/library/functions.html#int) = 0, gpu_memory_utilization: [float](https://docs.python.org/3/library/functions.html#float) = 0.9, swap_space: [float](https://docs.python.org/3/library/functions.html#float) = 4, cpu_offload_gb: [float](https://docs.python.org/3/library/functions.html#float) = 0, enforce_eager: [bool](https://docs.python.org/3/library/functions.html#bool) | [None](https://docs.python.org/3/library/constants.html#None) = None, max_context_len_to_capture: [int](https://docs.python.org/3/library/functions.html#int) | [None](https://docs.python.org/3/library/constants.html#None) = None, max_seq_len_to_capture: [int](https://docs.python.org/3/library/functions.html#int) = 8192, disable_custom_all_reduce: [bool](https://docs.python.org/3/library/functions.html#bool) = False, disable_async_output_proc: [bool](https://docs.python.org/3/library/functions.html#bool) = False, **kwargs) 
[[source]](https://docs.vllm.ai/en/latest/_modules/vllm/entrypoints/llm.html#LLM)  


1 个用于根据给定提示和采样参数生成文本的大语言模型。


该类包括 1 个 tokenizer、1 个语言模型（可能分布在多个 GPU 上）以及为中间状态分配的 GPU 内存空间（也称为 KV 缓存）。给定一批提示和采样参数，该类使用智能批处理机制和高效的内存管理从模型中生成文本。


## 参数：

* **model** – HuggingFace Transformers 模型的名称或路径。

* **tokenizer** – HuggingFace Transformers tokenizer 的名称或路径。

* **tokenizer_mode** –  tokenizer 模式。「auto」将在可用时使用快速 tokenizer，而「slow」将始终使用慢速 tokenizer。

* **skip_tokenizer_init** – 如果为 true，则跳过 tokenizer 和去 tokenizer 的初始化。输入中的提示需要有效的 Prompt_token_ids 和 None 。

* **trust_remote_code** – 下载模型和 tokenizer 时信任远程代码（例如，来自 HuggingFace）。

* **tensor_parallel_size** – 用于张量并行进行分布式执行的 GPU 数量。

* **dtype** – 模型权重和激活的数据类型。目前，我们支持 float32、float16 和 bfloat16。如果是「auto」，我们使用模型配置文件中指定的 torch_dtype 属性。但是，如果配置中的 torch_dtype 是 float32，我们将使用 float16。

* **quantization** – 用于量化模型权重的方法。目前，我们支持「awq」、「gptq」和「fp8」（实验性）。如果为 None，我们首先检查模型配置文件中的 quantization_config 属性。如果为该属性 None，我们假设模型权重未被量化，并使用 dtype 来确定权重的数据类型。

* **revision** – 要使用的特定模型版本。它可以是分支名称、标签名称或提交 ID。

* **tokenizer_revision** – 要使用的特定 tokenizer 版本。它可以是分支名称、标签名称或提交 ID。

* **seed** – 用于初始化采样随机数生成器的种子。

* **gpu_memory_utilization** – 为模型权重、激活值和 KV 缓存保留的 GPU 内存比率（介于 0 和 1 之间）。较高的值将增加 KV 缓存大小，从而提高模型的吞吐量。但是，如果该值太高，可能会导致内存不足 (OOM) 错误。

* **swap_space** – 每个 GPU 用作交换空间的 CPU 内存大小（以 GiB 为单位）。当请求的 best_of 采样参数大于 1 时，可用于临时存储请求的状态。如果所有请求的 best_of=1，您可以安全地将其设置为 0。否则，太小的值可能会导致内存不足 (OOM) 错误。

* **cpu_offload_gb** – 用于卸载模型权重的 CPU 内存大小（以 GiB 为单位）。这实际上增加了可用于保存模型权重的 GPU 内存空间，但每次前向传播都需要进行 CPU - GPU 数据传输。

* **enforce_eager** – 是否强制急切执行 (Eager Execution)。如果为 True，我们将禁用 CUDA 图并始终以 eager 模式执行模型。如果为 False，我们将混合使用 CUDA 图和 Eager Execution。

* **max_context_len_to_capture** – CUDA 图覆盖的最大上下文长度。当序列的上下文长度大于此值时，我们会回退到 eager 模式（已弃用。请改用 max_seq_len_to_capture）。

* **max_seq_len_to_capture** – CUDA 图覆盖的最大序列长度。当序列的上下文长度大于此值时，我们会回退到 eager 模式。此外，对于编码器-解码器模型，如果编码器输入的序列长度大于此值，我们也会回退到 eager 模式。

* **disable_custom_all_reduce** – 请参阅并行配置。

* ****kwargs** –  `EngineArgs` 的参数。（参阅 [Engine Arguments](https://docs.vllm.ai/en/latest/models/engine_args.html#engine-args)）   


**注意：**

该类旨在用于离线推理。对于在线服务，请改用 [AsyncLLMEngine](https://docs.vllm.ai/en/latest/models/engine_args.html#engine-args) 类代替。


>DEPRECATE_INIT_POSARGS: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)[[bool](https://docs.python.org/3/library/functions.html#bool)] = True
一个用于切换是否弃用 `LLM.init()` 中的位置参数的标志。


>DEPRECATE_LEGACY: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)[[bool](https://docs.python.org/3/library/functions.html#bool)] = False   
一个用于切换是否弃用旧版 generate/encode API 的标志。


>beam_search(prompts: [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]], params: BeamSearchParams) → [List](https://docs.python.org/3/library/typing.html#typing.List)[BeamSearchOutput]
[[source]](https://docs.vllm.ai/en/latest/_modules/vllm/entrypoints/llm.html#LLM.beam_search)

使用束搜索生成序列。


**参数：**

* **prompts：**提示列表。每个提示可以是一个字符串或一个标记 ID 列表。

* **params：**束搜索参数。

待办事项：束搜索如何与长度惩罚、频率惩罚和停止标准等协同工作？


>chat(messages: [List](https://docs.python.org/3/library/typing.html#typing.List)[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam | ChatCompletionToolMessageParam | ChatCompletionFunctionMessageParam | CustomChatCompletionMessageParam] | [List](https://docs.python.org/3/library/typing.html#typing.List)[[List](https://docs.python.org/3/library/typing.html#typing.List)[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam | ChatCompletionToolMessageParam | ChatCompletionFunctionMessageParam | CustomChatCompletionMessageParam]],, sampling_params: [SamplingParams](https://docs.vllm.ai/en/latest/dev/sampling_params.html#vllm.SamplingParams) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[SamplingParams](https://docs.vllm.ai/en/latest/dev/sampling_params.html#vllm.SamplingParams)] | [None](https://docs.python.org/3/library/constants.html#None) = None, use_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool) = True, lora_request: LoRARequest | [None](https://docs.python.org/3/library/constants.html#None) = None, chat_template: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, add_generation_prompt: [bool](https://docs.python.org/3/library/functions.html#bool) = True, nue_final_message: bool = False, tools: [List](https://docs.python.org/3/library/typing.html#typing.List)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [List](https://docs.python.org/3/library/typing.html#typing.List)[RequestOutput] 
[[source]](https://docs.vllm.ai/en/latest/_modules/vllm/entrypoints/llm.html#LLM.chat)  


生成聊天对话的响应。


使用 tokenizer 将聊天对话转换为文本提示，并调用 [generate()](https://docs.vllm.ai/en/latest/dev/offline_inference/llm.html#vllm.LLM.generate)方法来生成响应。


多模态输入的传递方式与将它们传递给 OpenAI API 的方式相同。


## 参数：

* **messages** – 单个对话表示为消息列表。每个对话表示为一个消息列表。每个消息是一个带有「role」（角色）和「content」（内容）键的字典。

* **sampling_params** – 用于文本生成的采样参数。如果为 None，则使用默认的采样参数。当它是单个值时，它将应用于每个提示。当它是列表时，列表必须与提示的长度相同，并且与提示一一配对。

* **use_tqdm** – 是否使用 tqdm 显示进度条。

* **lora_request** – 如果有，则为用于生成的 LoRA 请求。

* **chat_template** – 用于构建聊天的模板。如果未提供，将使用模型的默认聊天模板。

* **add_generation_prompt** – 如果为 True，则向每条消息添加生成模板。

* **continue_final_message**– 如果为 True，则继续对话中的最后一条消息，而不是开始新的一条。如果「add_generation_prompt」也为 True，则此参数不能为 True。

* **mm_processor_kwargs****–** 此聊天请求的多模态处理器关键字参数覆盖。仅用于离线请求。


## 返回

一个包含 `RequestOutput` 对象的列表，这些对象包含生成的响应，其顺序与输入消息的顺序相同。

>encode(prompts: [str](https://docs.python.org/3/library/stdtypes.html#str), pooling_params: PoolingParams | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[PoolingParams] | [None](https://docs.python.org/3/library/constants.html#None) = None, prompt_token_ids: [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, use_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool) = True, lora_request: [List](https://docs.python.org/3/library/typing.html#typing.List)[LoRARequest] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None) = None) → [List](https://docs.python.org/3/library/typing.html#typing.List)[EmbeddingRequestOutput] 
[[source]](https://docs.vllm.ai/en/latest/_modules/vllm/entrypoints/llm.html#LLM.encode)  

>encode(prompts: [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)], pooling_params: PoolingParams | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[PoolingParams] | [None](https://docs.python.org/3/library/constants.html#None) = None, prompt_token_ids: [List](https://docs.python.org/3/library/typing.html#typing.List)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, use_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool) = True, lora_request: [List](https://docs.python.org/3/library/typing.html#typing.List)[LoRARequest] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None) = None) → [List](https://docs.python.org/3/library/typing.html#typing.List)[EmbeddingRequestOutput]
>encode(prompts: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, pooling_params: PoolingParams | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[PoolingParams] | [None](https://docs.python.org/3/library/constants.html#None) = None, *, prompt_token_ids: [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], use_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool) = True, lora_request: [List](https://docs.python.org/3/library/typing.html#typing.List)[LoRARequest] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None) = None) → [List](https://docs.python.org/3/library/typing.html#typing.List)[EmbeddingRequestOutput]
>encode(prompts: [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, pooling_params: PoolingParams | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[PoolingParams] | [None](https://docs.python.org/3/library/constants.html#None) = None, *, prompt_token_ids: [List](https://docs.python.org/3/library/typing.html#typing.List)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]], use_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool) = True, lora_request: [List](https://docs.python.org/3/library/typing.html#typing.List)[LoRARequest] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None) = None) → [List](https://docs.python.org/3/library/typing.html#typing.List)[EmbeddingRequestOutput]
>encode(prompts: [None](https://docs.python.org/3/library/constants.html#None), pooling_params: [None](https://docs.python.org/3/library/constants.html#None), prompt_token_ids: [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)] | [List](https://docs.python.org/3/library/typing.html#typing.List)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]], use_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool) = True, lora_request: [List](https://docs.python.org/3/library/typing.html#typing.List)[LoRARequest] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None) = None) → [List](https://docs.python.org/3/library/typing.html#typing.List)[EmbeddingRequestOutput]
>encode(prompts: [PromptType](https://docs.vllm.ai/en/latest/dev/offline_inference/llm_inputs.html#vllm.inputs.PromptType) | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[PromptType](https://docs.vllm.ai/en/latest/dev/offline_inference/llm_inputs.html#vllm.inputs.PromptType)], /, *, pooling_params: PoolingParams | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[PoolingParams] | [None](https://docs.python.org/3/library/constants.html#None) = None, use_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool) = True, lora_request: [List](https://docs.python.org/3/library/typing.html#typing.List)[LoRARequest] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None) = None) → [List](https://docs.python.org/3/library/typing.html#typing.List)[EmbeddingRequestOutput] 
生成输入提示的补全。


这个类会自动对给定的提示进行批处理，同时考虑内存限制。为了获得最佳性能，请将所有提示放入一个列表中并将其传递给此方法。


## 参数：

* **prompts** – 给语言模型的提示。您可以传递一系列提示以进行批量推理。有关每个提示的格式的更多详细信息，请参阅 [PromptType](https://docs.vllm.ai/en/latest/dev/offline_inference/llm_inputs.html#vllm.inputs.PromptType)。

* **pooling_params** – 用于池化的池化参数。如果为 None，则使用默认的池化参数。

* **use_tqdm** – 是否使用 tqdm 显示进度条。

* **lora_request** – 如果有，则为用于生成的 LoRA 请求。

* **prompt_adapter_request** – 如果有，则为用于生成的提示适配器请求。


## 返回

一个包含 *EmbeddingRequestOutput* 对象的列表，这些对象包含生成的嵌入，其顺序与输入提示的顺序相同。


**注意：**

使用 `prompts` 和 `prompt_token_ids` 作为关键字参数的做法已经过时，将来可能会被弃用。您应该改为通过 `inputs` 参数传递它们。


>generate(prompts: [str](https://docs.python.org/3/library/stdtypes.html#str), sampling_params: [SamplingParams](https://docs.vllm.ai/en/latest/dev/sampling_params.html#vllm.SamplingParams) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[SamplingParams](https://docs.vllm.ai/en/latest/dev/sampling_params.html#vllm.SamplingParams)] | [None](https://docs.python.org/3/library/constants.html#None) = None, prompt_token_ids: [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, use_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool) = True, lora_request: [List](https://docs.python.org/3/library/typing.html#typing.List)[LoRARequest] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None) = None) → [List](https://docs.python.org/3/library/typing.html#typing.List)[RequestOutput] 
[[source]](https://docs.vllm.ai/en/latest/_modules/vllm/entrypoints/llm.html#LLM.generate)

>generate(prompts: [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)], sampling_params: [SamplingParams](https://docs.vllm.ai/en/latest/dev/sampling_params.html#vllm.SamplingParams) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[SamplingParams](https://docs.vllm.ai/en/latest/dev/sampling_params.html#vllm.SamplingParams)] | [None](https://docs.python.org/3/library/constants.html#None) = None, prompt_token_ids: [List](https://docs.python.org/3/library/typing.html#typing.List)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]] | [None](https://docs.python.org/3/library/constants.html#None) = None, use_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool) = True, lora_request: [List](https://docs.python.org/3/library/typing.html#typing.List)[LoRARequest] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None) = None) → [List](https://docs.python.org/3/library/typing.html#typing.List)[RequestOutput]
>generate(prompts: [str](https://docs.python.org/3/library/stdtypes.html#str) | [None](https://docs.python.org/3/library/constants.html#None) = None, sampling_params: [SamplingParams](https://docs.vllm.ai/en/latest/dev/sampling_params.html#vllm.SamplingParams) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[SamplingParams](https://docs.vllm.ai/en/latest/dev/sampling_params.html#vllm.SamplingParams)] | [None](https://docs.python.org/3/library/constants.html#None) = None, *, prompt_token_ids: [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], use_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool) = True, lora_request: [List](https://docs.python.org/3/library/typing.html#typing.List)[LoRARequest] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None) = None) → [List](https://docs.python.org/3/library/typing.html#typing.List)[RequestOutput]
>generate(prompts: [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, sampling_params: [SamplingParams](https://docs.vllm.ai/en/latest/dev/sampling_params.html#vllm.SamplingParams) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[SamplingParams](https://docs.vllm.ai/en/latest/dev/sampling_params.html#vllm.SamplingParams)] | [None](https://docs.python.org/3/library/constants.html#None) = None, *, prompt_token_ids: [List](https://docs.python.org/3/library/typing.html#typing.List)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]], use_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool) = True, lora_request: [List](https://docs.python.org/3/library/typing.html#typing.List)[LoRARequest] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None) = None) → [List](https://docs.python.org/3/library/typing.html#typing.List)[RequestOutput]
>generate(prompts: [None](https://docs.python.org/3/library/constants.html#None), sampling_params: [None](https://docs.python.org/3/library/constants.html#None), prompt_token_ids: [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)] | [List](https://docs.python.org/3/library/typing.html#typing.List)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]], use_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool) = True, lora_request: [List](https://docs.python.org/3/library/typing.html#typing.List)[LoRARequest] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None) = None) → [List](https://docs.python.org/3/library/typing.html#typing.List)[RequestOutput]
>generate(prompts: [PromptType](https://docs.vllm.ai/en/latest/dev/offline_inference/llm_inputs.html#vllm.inputs.PromptType) | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[PromptType](https://docs.vllm.ai/en/latest/dev/offline_inference/llm_inputs.html#vllm.inputs.PromptType)], /, *, sampling_params: [SamplingParams](https://docs.vllm.ai/en/latest/dev/sampling_params.html#vllm.SamplingParams) | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[SamplingParams](https://docs.vllm.ai/en/latest/dev/sampling_params.html#vllm.SamplingParams)] | [None](https://docs.python.org/3/library/constants.html#None) = None, use_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool) = True, lora_request: [List](https://docs.python.org/3/library/typing.html#typing.List)[LoRARequest] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None) = None) → [List](https://docs.python.org/3/library/typing.html#typing.List)[RequestOutput] 

为输入的提示生成补全内容。


这个类会自动对给定的提示进行批处理，同时考虑内存限制。为了获得最佳性能，请将你所有的提示放入一个单独的列表中，并将其传递给这个方法。


## 参数：

* **prompts** – 给 LLM 的提示。您可以传递一系列提示以进行批量推理。有关每个提示的格式的更多详细信息，请参阅 [PromptType](https://docs.vllm.ai/en/latest/dev/offline_inference/llm_inputs.html#vllm.inputs.PromptType)。

* **sampling_params** – 用于文本生成的采样参数。如果为 None，则使用默认的采样参数。当它是单个值时，它将应用于每个提示。当它是一个列表时，该列表必须与提示长度相同，并且与提示一一配对。

* **use_tqdm** – 是否使用 tqdm 显示进度条。

* **lora_request** – 如果有，则为用于生成的 LoRA 请求。

* **prompt_adapter_request** – 如果有，则为用于生成的提示适配器请求。


## 返回

一个包含 `RequestOutput` 对象的列表，这些对象包含生成的补全内容，其顺序与输入提示的顺序相同。


**注意：**

使用 `prompts` 和 `prompt_token_ids` 作为关键字参数的做法已经过时，将来可能会被弃用。您应该改为通过 `inputs` 参数传递它们。

