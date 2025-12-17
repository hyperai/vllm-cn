---
title: LLM Class
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

**class vllm.LLM(model:[str](https://docs.python.org/3/library/stdtypes.html#str), tokenizer:[str](https://docs.python.org/3/library/stdtypes.html#str)_|_[None](https://docs.python.org/3/library/constants.html#None) _= None, tokenizer_mode:[str](https://docs.python.org/3/library/stdtypes.html#str) = 'auto', skip_tokenizer_init:[bool](https://docs.python.org/3/library/functions.html#bool) = False, trust_remote_code: [bool](https://docs.python.org/3/library/functions.html#bool) = False, allowed_local_media_path:[str](https://docs.python.org/3/library/stdtypes.html#str) ='', tensor_parallel_size: [int](https://docs.python.org/3/library/functions.html#int) = 1, dtype: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'auto', quantization:[str](https://docs.python.org/3/library/stdtypes.html#str) \ [None](https://docs.python.org/3/library/constants.html#None) = None, revision: [str](https://docs.python.org/3/library/stdtypes.html#str) \ [None](https://docs.python.org/3/library/constants.html#None) = None, tokenizer_revision : [str](https://docs.python.org/3/library/stdtypes.html#str) \ [None](https://docs.python.org/3/library/constants.html#None) = None, seed:[int](https://docs.python.org/3/library/functions.html#int) \ [None](https://docs.python.org/3/library/constants.html#None) = None, gpu_memory_utilization: [float](https://docs.python.org/3/library/functions.html#float) = 0.9, swap_space: [float](https://docs.python.org/3/library/functions.html#float) = 4, cpu_offload_gb:[float](https://docs.python.org/3/library/functions.html#float) = 0, enforce_eager: [bool](https://docs.python.org/3/library/functions.html#bool) \ [None](https://docs.python.org/3/library/constants.html#None) = None, max_seq_len_to_capture: [int](https://docs.python.org/3/library/functions.html#int) = 8192, disable_custom_all_reduce : [bool](https://docs.python.org/3/library/functions.html#bool) = False,disable_async_output_proc: [bool](https://docs.python.org/3/library/functions.html#bool) = False, hf_overrides: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[transformers.PretrainedConfig], transformers.PretrainedConfig] | None = None, mm_processor_kwargs: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, task: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['auto', 'generate', 'embedding', 'embed', 'classify', 'score', 'reward', 'transcription'] = 'auto', override_pooler_config: PoolerConfig | [None](https://docs.python.org/3/library/constants.html#None) = None, compilation_config: [int](https://docs.python.org/3/library/functions.html#int) | [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None, ** kwargs)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py#L53)

用于根据给定提示和采样参数生成文本的 LLM。

该类包括 tokenizer、语言模型（可能分布在多个 GPU 上）以及为中间状态分配的 GPU 内存空间（也称为 KV 缓存）。给定一批提示和采样参数，该类使用智能批处理机制和高效的内存管理从模型中生成文本。

**参数：**

- **model** – HuggingFace Transformers 模型的名称或路径。
- **tokenizer** – HuggingFace Transformers 分词器的名称或路径。
- **tokenizer_mode** – 分词器模式。「auto」将使用快速分词器（如果可用），而 "slow" 将始终使用慢速分词器。
- **skip_tokenizer_init** – 如果为 true，则跳过分词器和反分词器的初始化。期望输入中的 `prompt_token_ids` 有效，并且 `prompt` 为 None。
- **trust_remote_code** – 在下载模型和分词器时信任远程代码（例如来自 HuggingFace）。
- **allowed_local_media_path** – 允许 API 请求从服务器文件系统指定的目录中读取本地图像或视频。这是一个安全风险。仅在受信任的环境中启用。
- **tensor_parallel_size** – 用于张量并行分布式执行的 GPU 数量。
- **dtype** – 模型权重和激活的数据类型。目前，我们支持 `float32`、`float16` 和 `bfloat16`。如果为 `auto`，则使用模型配置文件中指定的 `torch_dtype` 属性。但是，如果配置文件中的 `torch_dtype` 为 `float32`，我们将使用 `float16`。
- **quantization** – 用于量化模型权重的方法。目前，我们支持「awq」、「gptq」和 "fp8"（实验性）。如果为 None，我们首先检查模型配置文件中的 `quantization_config` 属性。如果为 None，我们假设模型权重未量化，并使用 `dtype` 确定权重的数据类型。
- **revision** – 要使用的特定模型版本。可以是分支名称、标签名称或提交 ID。
- **tokenizer_revision** – 要使用的特定分词器版本。可以是分支名称、标签名称或提交 ID。
- **seed** – 用于初始化随机数生成器的种子，用于采样。
- **gpu_memory_utilization** – 为模型权重、激活和 KV 缓存保留的 GPU 内存比例（介于 0 和 1 之间）。较高的值将增加 KV 缓存大小，从而提高模型的吞吐量。但是，如果值过高，可能会导致内存不足（OOM）错误。
- **swap_space** – 每个 GPU 用于交换空间的 CPU 内存大小（GiB）。当请求的 `best_of` 采样参数大于 1 时，可以用于临时存储请求的状态。如果所有请求的 `best_of=1`，则可以安全地将其设置为 0。请注意，`best_of` 仅在 V0 中支持。否则，值过小可能会导致内存不足（OOM）错误。
- **cpu_offload_gb** – 用于卸载模型权重的 CPU 内存大小（GiB）。这实际上增加了可用于保存模型权重的 GPU 内存空间，但代价是每次前向传递时 CPU-GPU 数据传输。
- **enforce_eager** – 是否强制启用 eager 执行。如果为 True，我们将禁用 CUDA 图并始终以 eager 模式执行模型。如果为 False，我们将混合使用 CUDA 图和 eager 执行。
- **max_seq_len_to_capture** – CUDA 图覆盖的最大序列长度。当序列的上下文长度超过此值时，我们将回退到 eager 模式。此外，对于编码器-解码器模型，如果编码器输入的序列长度超过此值，我们也将回退到 eager 模式。
- **disable_custom_all_reduce** – 参见 `ParallelConfig`。
- **disable_async_output_proc** – 禁用异步输出处理。这可能会导致性能下降。
- **hf_overrides** – 如果是字典，则包含要转发到 HuggingFace 配置的参数。如果是可调用对象，则调用它来更新 HuggingFace 配置。
- **compilation_config** – 可以是整数或字典。如果是整数，则用作编译优化级别。如果是字典，则可以指定完整的编译配置。
- \***\*kwargs** – `EngineArgs` 的参数。（参见 [引擎参数](https://docs.vllm.ai/en/latest/serving/engine_args.html#engine-args)）

> **注意**
> 
> 该类旨在用于离线推理。对于在线服务，请改用 `AsyncLLMEngine` 类。

**DEPRECATE_INIT_POSARGS:_[ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)_[_[bool](https://docs.python.org/3/library/functions.html#bool)_] = True**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py#L53)

用于切换是否弃用 `LLM.__init__()` 中的位置参数的标志。

**DEPRECATE_LEGACY:_[ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)_[_[bool](https://docs.python.org/3/library/functions.html#bool)_] = True**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py#L53)

用于切换是否弃用旧版 generate/encode API 的标志。

**apply_model(func:_[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)_[[_[torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)_], _R] → N[list](https://docs.python.org/3/library/stdtypes.html#list)[_R]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py#L506)

在每个工作线程中直接运行模型上的函数，并返回每个工作线程的结果。

**beam_search(prompts:[list](https://docs.python.org/3/library/stdtypes.html#list)_[_[Union](https://docs.python.org/3/library/typing.html#typing.Union)_[_[vllm.inputs.data.TokensPrompt](https://docs.vllm.ai/en/latest/api/offline_inference/llm_inputs.html#vllm.inputs.TokensPrompt)_,_[vllm.inputs.data.TextPrompt](https://docs.vllm.ai/en/latest/api/offline_inference/llm_inputs.html#vllm.inputs.TextPrompt)_]],params: BeamSearchParams → [list](https://docs.python.org/3/library/stdtypes.html#list)[vllm.beam_search.BeamSearchOutput]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py#L514)

使用束搜索生成序列。

**参数：**

- **prompts** – 提示列表。每个提示可以是字符串或 token ID 列表。
- **params** – 束搜索参数。

TODO：束搜索如何与长度惩罚、频率惩罚和停止条件等一起工作？

**chat(messages:_[list](https://docs.python.org/3/library/stdtypes.html#list)_[[Union](https://docs.python.org/3/library/typing.html#typing.Union)_
[openai.types.chat.chat_completion_developer_message_param.ChatCompletionDeveloperMessageParam, 
openai.types.chat.chat_completion_system_message_param.ChatCompletionSystemMessageParam, 
openai.types.chat.chat_completion_user_message_param.ChatCompletionUserMessageParam, 
openai.types.chat.chat_completion_assistant_message_param.ChatCompletionAssistantMessageParam, 
openai.types.chat.chat_completion_tool_message_param.ChatCompletionToolMessageParam, 
openai.types.chat.chat_completion_function_message_param.ChatCompletionFunctionMessageParam, 
vllm.entrypoints.chat_utils.CustomChatCompletionMessageParam]] |  
[list](https://docs.python.org/3/library/stdtypes.html#list)_[_[list](https://docs.python.org/3/library/stdtypes.html#list)_[_[Union](https://docs.python.org/3/library/typing.html#typing.Union)_
[openai.types.chat.chat_completion_developer_message_param.ChatCompletionDeveloperMessageParam, 
openai.types.chat.chat_completion_system_message_param.ChatCompletionSystemMessageParam, 
openai.types.chat.chat_completion_user_message_param.ChatCompletionUserMessageParam, 
openai.types.chat.chat_completion_assistant_message_param.ChatCompletionAssistantMessageParam, 
openai.types.chat.chat_completion_tool_message_param.ChatCompletionToolMessageParam, 
openai.types.chat.chat_completion_function_message_param.ChatCompletionFunctionMessageParam, 
vllm.entrypoints.chat_utils.CustomChatCompletionMessageParam]]],
sampling*params:[SamplingParams](https://docs.vllm.ai/en/latest/api/inference_params.html#vllm.SamplingParams)__[list](https://docs.python.org/3/library/stdtypes.html#list)_[_[vllm.sampling_params.SamplingParams](https://docs.vllm.ai/en/latest/api/inference_params.html#vllm.SamplingParams)_] |_[None](https://docs.python.org/3/library/constants.html#None) = None, use_tqdm:[bool](https://docs.python.org/3/library/functions.html#bool) = True, lora_request: LoRARequest | [None](https://docs.python.org/3/library/constants.html#None) = None, chat_template:[str](https://docs.python.org/3/library/stdtypes.html#str)_|_[None](https://docs.python.org/3/library/constants.html#None) = None, chat_template_content_format:[Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['auto', 'string', 'openai'] = 'auto', add_generation_prompt:_[bool](https://docs.python.org/3/library/functions.html#bool) = True, continue_final_message:[bool](https://docs.python.org/3/library/functions.html#bool) = False, tools:[list](https://docs.python.org/3/library/stdtypes.html#list)_[_[dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str)_, [Any](https://docs.python.org/3/library/typing.html#typing.Any)]] |[None](https://docs.python.org/3/library/constants.html#None) = None, mm_processor_kwargs: [dict](https://docs.python.org/3/library/stdtypes.html#dict)_[_[str](https://docs.python.org/3/library/stdtypes.html#str)_,_[Any](https://docs.python.org/3/library/typing.html#typing.Any)_] |_[None](https://docs.python.org/3/library/constants.html#None)  = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[vllm.outputs.RequestOutput]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py#L624)

为聊天对话生成响应。

聊天对话使用分词器转换为文本提示，并调用 `generate()` 方法生成响应。

多模态输入可以以与 OpenAI API 相同的方式传递。

**参数：**

**messages** –

- 对话列表或单个对话。

  - 每个对话由消息列表表示。

  - 每条消息是一个包含「role」和「content」键的字典。

- **sampling_params** – 文本生成的采样参数。如果为 None，则使用默认采样参数。当它为单个值时，将应用于每个提示。当它为列表时，列表长度必须与提示数量相同，并逐个与提示配对。
- **use_tqdm** – 是否使用 tqdm 显示进度条。
- **lora_request** – 用于生成的 LoRA 请求（如果有）。
- **chat_template** – 用于构建对话的模板。如果未提供，则使用模型的默认对话模板。
- 消息内容的渲染格式。

  - 「string」将内容渲染为字符串。例如：`"Who are you?"`

  - 「openai」将内容渲染为字典列表，类似于 OpenAI 的模式。例如：`[{"type": "text", "text": "Who are you?"}]`

- **add_generation_prompt** – 如果为 True，则为每条消息添加生成模板。
- **continue_final_message** – 如果为 True，则继续对话中的最后一条消息，而不是开始新消息。如果`add_generation_prompt`也为`True`，则不能为`True`。
- **mm_processor_kwargs** – 此对话请求的多模态处理器参数覆盖。仅用于离线请求。

**返回：**

包含生成响应的`RequestOutput`对象列表，顺序与输入消息相同。

**classify(prompts: [str](https://docs.python.org/3/library/stdtypes.html#str) | [TextPrompt](https://docs.vllm.ai/en/v0.8.4_a/api/offline_inference/llm_inputs.html#vllm.inputs.TextPrompt) | [TokensPrompt](https://docs.vllm.ai/en/v0.8.4_a/api/offline_inference/llm_inputs.html#vllm.inputs.TokensPrompt) | ExplicitEncoderDecoderPrompt | [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[str](https://docs.python.org/3/library/stdtypes.html#str) | [TextPrompt](https://docs.vllm.ai/en/v0.8.4_a/api/offline_inference/llm_inputs.html#vllm.inputs.TextPrompt) | [TokensPrompt](https://docs.vllm.ai/en/v0.8.4_a/api/offline_inference/llm_inputs.html#vllm.inputs.TokensPrompt) | ExplicitEncoderDecoderPrompt], /, * , use_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool) = True, lora_request: [list](https://docs.python.org/3/library/stdtypes.html#list)[vllm.lora.request.LoRARequest] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None) = None, prompt_adapter_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[vllm.outputs.ClassificationRequestOutput]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py#L966)

为每个提示生成分类 logits。

此类会自动批处理给定的提示，并考虑内存限制。为了获得最佳性能，请将所有提示放入单个列表中传递给此方法。

**参数：**

- **prompts** – 传递给LLM的提示。您可以传递一系列提示以进行批量推理。有关每个提示的格式的更多详细信息，请参见`PromptType`。
- **use_tqdm** – 是否使用 tqdm 显示进度条。
- **lora_request** – 用于生成的 LoRA 请求（如果有）。
- **prompt_adapter_request** – 用于生成的提示适配器请求（如果有）。

**返回：**

包含嵌入向量的 `ClassificationRequestOutput` 对象列表，顺序与输入提示相同。

**collective_rpc(method: [str](https://docs.python.org/3/library/stdtypes.html#str) | [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[...], _R], timeout: [float](https://docs.python.org/3/library/functions.html#float) | [None](https://docs.python.org/3/library/constants.html#None) = None, args: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple) = (), kwargs: [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [list](https://docs.python.org/3/library/stdtypes.html#list)[_R]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py#L476)

在所有工作节点上执行 RPC 调用。

**参数：**

- **method** –
要执行的工作节点方法的名称，或一个可调用对象，该对象将被序列化并发送到所有工作节点执行。
- 如果方法是可调用对象，则除了传递给 args 和 kwargs 的参数外，还应接受一个额外的 self 参数。self 参数将是工作节点对象。
- **timeout** – 等待执行的最大时间（秒）。超时后引发`TimeoutError`。None表示无限期等待。
- **args** – 传递给工作节点方法的位置参数。
- **kwargs** – 传递给工作节点方法的关键字参数。

**返回：**

包含每个工作节点结果的列表。

> **注意**
> 
> 建议仅使用此 API 传递控制消息，并设置数据平面通信传递数据。

**embed(prompts: str | TextPrompt | TokensPrompt | ExplicitEncoderDecoderPrompt | Sequence[str | TextPrompt | TokensPrompt | ExplicitEncoderDecoderPrompt], /, * , use_tqdm: bool = True, pooling_params: PoolingParams | Sequence[PoolingParams] | None = None, lora_request: list[vllm.lora.request.LoRARequest] | LoRARequest | None = None, prompt_adapter_request: PromptAdapterRequest | None = None) → list[vllm.outputs.EmbeddingRequestOutput]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py#L926)

为每个提示生成嵌入向量。

此类会自动批处理给定的提示，考虑内存限制。为了获得最佳性能，请将所有提示放入单个列表并传递给此方法。

**参数：**

- **prompts** – 传递给 LLM 的提示。您可以传递一系列提示以进行批量推理。有关每个提示的格式的更多详细信息，请参见`PromptType`。
- **use_tqdm** – 是否使用 tqdm 显示进度条。
- **lora_request** – 用于生成的 LoRA 请求（如果有）。
- **prompt_adapter_request** – 用于生成的提示适配器请求（如果有）。

**返回：**

包含嵌入向量的`EmbeddingRequestOutput`对象列表，顺序与输入提示相同。

**encode(_prompts: PromptType | Sequence\[PromptType\]_, _/_, _pooling\_params: [PoolingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.PoolingParams "vllm.PoolingParams") | Sequence\[[PoolingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.PoolingParams "vllm.PoolingParams")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _\*_, _use\_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_, _lora\_request: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[LoRARequest\] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_adapter\_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[PoolingRequestOutput\][\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/utils.py#L854)[#](https://docs.vllm.ai/en/v0.8.4_a/api/offline_inference/llm.html#vllm.LLM.encode "Permalink to this definition")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/utils.py#L844)

**encode(_prompts: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _pooling\_params: [PoolingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.PoolingParams "vllm.PoolingParams") | Sequence\[[PoolingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.PoolingParams "vllm.PoolingParams")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_token\_ids: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _use\_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_, _lora\_request: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[LoRARequest\] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_adapter\_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[PoolingRequestOutput\]**

**encode(_prompts: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")\]_, _pooling\_params: [PoolingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.PoolingParams "vllm.PoolingParams") | Sequence\[[PoolingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.PoolingParams "vllm.PoolingParams")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_token\_ids: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _use\_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_, _lora\_request: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[LoRARequest\] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_adapter\_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[PoolingRequestOutput\]**

**encode(_prompts: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _pooling\_params: [PoolingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.PoolingParams "vllm.PoolingParams") | Sequence\[[PoolingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.PoolingParams "vllm.PoolingParams")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _\*_, _prompt\_token\_ids: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_, _use\_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_, _lora\_request: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[LoRARequest\] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_adapter\_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[PoolingRequestOutput\]**

**encode(_prompts: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _pooling\_params: [PoolingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.PoolingParams "vllm.PoolingParams") | Sequence\[[PoolingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.PoolingParams "vllm.PoolingParams")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _\*_, _prompt\_token\_ids: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]\]_, _use\_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_, _lora\_request: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[LoRARequest\] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_adapter\_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[PoolingRequestOutput\]**

**encode(_prompts: [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")_, _pooling\_params: [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")_, _prompt\_token\_ids: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]\]_, _use\_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_, _lora\_request: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[LoRARequest\] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_adapter\_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[PoolingRequestOutput\]**

对输入提示的隐藏状态进行池化。

此类会自动批处理给定的提示，并考虑内存限制。为了获得最佳性能，请将所有提示放入单个列表中传递给此方法。

**参数：**

- **prompts** – 传递给 LLM 的提示。您可以传递一系列提示以进行批量推理。有关每个提示的格式的更多详细信息，请参见`PromptType`。
- **pooling_params** – 池化参数。如果为 None，则使用默认池化参数。
- **use_tqdm** – 是否使用 tqdm 显示进度条。
- **lora_request** – 用于生成的 LoRA 请求（如果有）。
- **prompt_adapter_request** – 用于生成的提示适配器请求（如果有）。

**返回：**

包含池化隐藏状态的`PoolingRequestOutput`对象列表，顺序与输入提示相同。

> **注意**
> 
> 使用`prompts`和`prompt_token_ids`作为关键字参数为遗留用法，未来可能会被弃用。您应通过`inputs`参数传递它们。

**generate(_prompts: PromptType | Sequence\[PromptType\]_, _/_, _sampling\_params: [SamplingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.SamplingParams "vllm.SamplingParams") | Sequence\[[SamplingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.SamplingParams "vllm.SamplingParams")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _\*_, _use\_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_, _lora\_request: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[LoRARequest\] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_adapter\_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _guided\_options\_request: LLMGuidedOptions | GuidedDecodingRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[RequestOutput\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/utils.py#L377)

**generate(_prompts: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _sampling\_params: [SamplingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.SamplingParams "vllm.SamplingParams") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[SamplingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.SamplingParams "vllm.SamplingParams")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_token\_ids: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _use\_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_, _lora\_request: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[LoRARequest\] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_adapter\_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _guided\_options\_request: LLMGuidedOptions | GuidedDecodingRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[RequestOutput\]**

**generate(_prompts: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")\]_, _sampling\_params: [SamplingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.SamplingParams "vllm.SamplingParams") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[SamplingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.SamplingParams "vllm.SamplingParams")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_token\_ids: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _use\_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_, _lora\_request: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[LoRARequest\] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_adapter\_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _guided\_options\_request: LLMGuidedOptions | GuidedDecodingRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[RequestOutput\]**

**generate(_prompts: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _sampling\_params: [SamplingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.SamplingParams "vllm.SamplingParams") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[SamplingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.SamplingParams "vllm.SamplingParams")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _\*_, _prompt\_token\_ids: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_, _use\_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_, _lora\_request: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[LoRARequest\] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_adapter\_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _guided\_options\_request: LLMGuidedOptions | GuidedDecodingRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[RequestOutput\]**

**generate(_prompts: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _sampling\_params: [SamplingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.SamplingParams "vllm.SamplingParams") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[SamplingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.SamplingParams "vllm.SamplingParams")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _\*_, _prompt\_token\_ids: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]\]_, _use\_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_, _lora\_request: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[LoRARequest\] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_adapter\_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _guided\_options\_request: LLMGuidedOptions | GuidedDecodingRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[RequestOutput\]**

**generate(_prompts: [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")_, _sampling\_params: [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")_, _prompt\_token\_ids: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]\]_, _use\_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_, _lora\_request: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[LoRARequest\] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_adapter\_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _guided\_options\_request: LLMGuidedOptions | GuidedDecodingRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[RequestOutput\]**

生成输入提示的补全。

此类会自动批处理给定的提示，并考虑内存限制。为了获得最佳性能，请将所有提示放入单个列表中传递给此方法。

**参数：**

- **prompts** – 传递给 LLM 的提示。您可以传递一系列提示以进行批量推理。有关每个提示的格式的更多详细信息，请参见`PromptType`。
- **sampling_params** – 文本生成的采样参数。如果为 None，则使用默认采样参数。当它为单个值时，将应用于每个提示。当它为列表时，列表长度必须与提示数量相同，并逐个与提示配对。
- **use_tqdm** – 是否使用 tqdm 显示进度条。
- **lora_request** – 用于生成的 LoRA 请求（如果有）。
- **prompt_adapter_request** – 用于生成的提示适配器请求（如果有）。
- **priority** – 请求的优先级（如果有）。仅在启用优先级调度策略时适用。

**返回：**

包含生成补全的`RequestOutput`对象列表，顺序与输入提示相同。

> **注意**
> 
> 使用`prompts`和`prompt_token_ids`作为关键字参数为遗留用法，未来可能会被弃用。您应通过`inputs`参数传递它们。

**score(_text\_1: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [TextPrompt](https://docs.vllm.ai/en/v0.8.4_a/api/offline_inference/llm_inputs.html#vllm.inputs.TextPrompt "vllm.inputs.data.TextPrompt") | [TokensPrompt](https://docs.vllm.ai/en/v0.8.4_a/api/offline_inference/llm_inputs.html#vllm.inputs.TokensPrompt "vllm.inputs.data.TokensPrompt") | [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [TextPrompt](https://docs.vllm.ai/en/v0.8.4_a/api/offline_inference/llm_inputs.html#vllm.inputs.TextPrompt "vllm.inputs.data.TextPrompt") | [TokensPrompt](https://docs.vllm.ai/en/v0.8.4_a/api/offline_inference/llm_inputs.html#vllm.inputs.TokensPrompt "vllm.inputs.data.TokensPrompt")\]_, _text\_2: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [TextPrompt](https://docs.vllm.ai/en/v0.8.4_a/api/offline_inference/llm_inputs.html#vllm.inputs.TextPrompt "vllm.inputs.data.TextPrompt") | [TokensPrompt](https://docs.vllm.ai/en/v0.8.4_a/api/offline_inference/llm_inputs.html#vllm.inputs.TokensPrompt "vllm.inputs.data.TokensPrompt") | [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [TextPrompt](https://docs.vllm.ai/en/v0.8.4_a/api/offline_inference/llm_inputs.html#vllm.inputs.TextPrompt "vllm.inputs.data.TextPrompt") | [TokensPrompt](https://docs.vllm.ai/en/v0.8.4_a/api/offline_inference/llm_inputs.html#vllm.inputs.TokensPrompt "vllm.inputs.data.TokensPrompt")\]_, _/_, _\*_, _truncate\_prompt\_tokens: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _use\_tqdm: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_, _lora\_request: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[vllm.lora.request.LoRARequest\] | LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_adapter\_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[vllm.outputs.ScoringRequestOutput\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py#L1092)

为所有`<text,text_pair>`对生成相似度分数。

输入可以是`1 -> 1`、`1 -> N`或`N -> N`。在`1 - N`的情况下，`text_1`句子将被复制`N`次以与`text_2`句子配对。输入对用于构建交叉编码器模型的提示列表。此类会自动批处理提示，并考虑内存限制。为了获得最佳性能，请将所有文本放入单个列表中传递给此方法。

**参数：**

- **text_1** – 可以是单个提示或提示列表，如果是列表，则必须与`text_2`列表长度相同。
- **text_2** – 与查询配对以形成 LLM 输入的文本。有关每个提示的格式的更多详细信息，请参见`PromptType`。
- **use_tqdm** – 是否使用 tqdm 显示进度条。
- **lora_request** – 用于生成的 LoRA 请求（如果有）。
- **prompt_adapter_request** – 用于生成的提示适配器请求（如果有）。

**返回：**

包含生成分数的`ScoringRequestOutput`对象列表，顺序与输入提示相同。

**sleep(_level: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") \= 1_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py#L1202)

使引擎进入睡眠状态。引擎不应处理任何请求。调用者应确保在调用 wake_up 之前没有请求正在处理。

**参数：**

**level** – 睡眠级别。级别 1 睡眠将卸载模型权重并丢弃 kv 缓存。kv 缓存的内容将被遗忘。级别1睡眠适用于睡眠和唤醒引擎以再次运行相同的模型。模型权重备份在 CPU 内存中。请确保有足够的 CPU 内存来存储模型权重。级别 2 睡眠将丢弃模型权重和kv缓存。模型权重和 kv 缓存的内容都将被遗忘。级别 2 睡眠适用于睡眠和唤醒引擎以运行不同的模型或更新模型，其中不需要先前的模型权重。它减少了 CPU 内存压力。

wake\_up(_tags: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py#L1223)

从睡眠模式唤醒引擎。有关更多详细信息，请参见`sleep()`方法。

参数：

**tags** – 用于为特定内存分配重新分配引擎内存的可选标签列表。值必须在 （“weights”， “kv_cache”，） 中。如果为 None，则重新分配所有内存。在再次使用引擎之前，应使用所有标记（或 None）调用 wake_up。
