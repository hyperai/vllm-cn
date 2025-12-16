---
title: AsyncLLMEngine
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

**class_ vllm.AsyncLLMEngine(_\*args_, _log\_requests: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_, _start\_engine\_loop: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_, _\*\*kwargs_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L574)

基类：`EngineClient`

`LLMEngine` 的异步封装类。

该类用于封装 `LLMEngine` 类，使其变为异步。它使用 asyncio 创建一个后台循环，持续处理传入的请求。当等待队列中有请求时，`generate` 方法会触发 `LLMEngine` 进行处理。`generate` 方法将 `LLMEngine` 的输出结果返回给调用者。

**参数:**

- **log_requests** – 是否记录请求。
- **start_engine_loop** – 如果为 True，生成调用时将自动启动运行引擎的后台任务。
- **\*args** – `LLMEngine` 的参数。
- \***\*kwargs** – `LLMEngine` 的参数。

**async_ abort(_request\_id: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_) → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L1110)

中止请求。

中止已提交的请求。如果请求已完成或未找到，此方法将不执行任何操作。

**参数:**

**request_id** – 请求的唯一标识符。

**async_ add\_lora(_lora\_request: LoRARequest_) → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L1234)[#](https://docs.vllm.ai/en/v0.8.4_a/api/engine/async_llm_engine.html#vllm.AsyncLLMEngine.add_lora "Permalink to this definition")

将新的 LoRA 适配器加载到引擎中，以供后续请求使用。

**async_ check\_health() → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L1167)

如果引擎运行状况不佳，则会引发错误。

**async_ encode(_prompt: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [TextPrompt](https://docs.vllm.ai/en/v0.8.4_a/api/offline_inference/llm_inputs.html#vllm.inputs.TextPrompt "vllm.inputs.data.TextPrompt") | [TokensPrompt](https://docs.vllm.ai/en/v0.8.4_a/api/offline_inference/llm_inputs.html#vllm.inputs.TokensPrompt "vllm.inputs.data.TokensPrompt") | ExplicitEncoderDecoderPrompt_, _pooling\_params: [PoolingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.PoolingParams "vllm.pooling_params.PoolingParams")_, _request\_id: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _lora\_request: LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _trace\_headers: [Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _priority: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") \= 0_) → [AsyncGenerator](https://docs.python.org/3/library/typing.html#typing.AsyncGenerator "(in Python v3.13)")\[PoolingRequestOutput, [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L1025)

为池化模型的请求生成输出。

为请求生成输出。此方法是一个协程。它将请求添加到 `LLMEngine` 的等待队列中，并将 `LLMEngine` 的输出流式传输给调用者。

**参数：**

- **prompt** – 输入给 LLM 的提示。有关每种输入格式的更多详细信息，请参阅 `PromptType`。
- **pooling_params** – 请求的池化参数。
- **request_id** – 请求的唯一标识符。
- **lora_request** – 用于生成的 LoRA 请求（如果有）。
- **trace_headers** – OpenTelemetry 跟踪头。
- **priority** – 请求的优先级。仅适用于优先级调度。

**生成:**

`LLMEngine` 为请求生成的 `PoolingRequestOutput` 对象。

**详细信息：**

- 如果引擎未运行，则启动后台循环，该循环会迭代调用 `engine_step()` 来处理等待的请求。
- 将请求添加到引擎的 `RequestTracker` 中。在下一次后台循环中，该请求将被发送到底层引擎。同时，将创建一个相应的 `AsyncStream`。
- 等待来自 `AsyncStream` 的请求输出并生成它们。

**示例**

```plain
>>> # 请参考 entrypoints/api_server.py 获取完整示例。
>>>
>>> # 初始化引擎和示例输入
>>> # 注意这里的 engine_args 是 AsyncEngineArgs 实例
>>> engine = AsyncLLMEngine.from_engine_args(engine_args)
>>> example_input = {
>>>     "input": "What is LLM?",
>>>     "request_id": 0,
>>> }
>>>
>>> # 开始生成
>>> results_generator = engine.encode(
>>>    example_input["input"],
>>>    PoolingParams(),
>>>    example_input["request_id"])
>>>
>>> # 获取结果
>>> final_output = None
>>> async for request_output in results_generator:
>>>     if await request.is_disconnected():
>>>         # 如果客户端断开连接，则终止请求。
>>>         await engine.abort(request_id)
>>>         # 返回一个错误，或者抛出一个错误。
>>>         ...
>>>     final_output = request_output
>>>
>>> # 处理并返回最终输出
>>> ...
```

**async_ engine\_step(_virtual\_engine: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_) → [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L730)

触发引擎处理等待中的请求。

如果有正在处理的请求，则返回 True。

**classmethod_ from\_engine\_args(_engine\_args: AsyncEngineArgs_, _start\_engine\_loop: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_, _usage\_context: UsageContext \= UsageContext.ENGINE\_CONTEXT_, _stat\_loggers: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), StatLoggerBase\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [AsyncLLMEngine](https://docs.vllm.ai/en/v0.8.4_a/api/engine/async_llm_engine.html#vllm.AsyncLLMEngine "vllm.engine.async_llm_engine.AsyncLLMEngine")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L632)

根据引擎参数创建异步 LLM 引擎。

**classmethod_ from\_vllm\_config(_vllm\_config: VllmConfig_, _start\_engine\_loop: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_, _usage\_context: UsageContext \= UsageContext.ENGINE\_CONTEXT_, _stat\_loggers: [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), vllm.engine.metrics\_types.StatLoggerBase\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _disable\_log\_requests: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= False_, _disable\_log\_stats: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= False_) → [AsyncLLMEngine](https://docs.vllm.ai/en/v0.8.4_a/api/engine/async_llm_engine.html#vllm.AsyncLLMEngine "vllm.engine.async_llm_engine.AsyncLLMEngine")**

从 EngineArgs 创建一个 AsyncLLMEngine。

**async_ generate(_prompt: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [TextPrompt](https://docs.vllm.ai/en/v0.8.4_a/api/offline_inference/llm_inputs.html#vllm.inputs.TextPrompt "vllm.inputs.data.TextPrompt") | [TokensPrompt](https://docs.vllm.ai/en/v0.8.4_a/api/offline_inference/llm_inputs.html#vllm.inputs.TokensPrompt "vllm.inputs.data.TokensPrompt") | ExplicitEncoderDecoderPrompt_, _sampling\_params: [SamplingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.SamplingParams "vllm.sampling_params.SamplingParams")_, _request\_id: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _lora\_request: LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _trace\_headers: [Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_adapter\_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _priority: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") \= 0_) → [AsyncGenerator](https://docs.python.org/3/library/typing.html#typing.AsyncGenerator "(in Python v3.13)")\[RequestOutput, [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L934)

为请求生成输出。

为请求生成输出。此方法是一个协程。它将请求添加到 `LLMEngine` 的等待队列中，并将 `LLMEngine` 的输出流式传输给调用者。

**参数：**

- **prompt** – 输入给 LLM 的提示。有关每种输入格式的更多详细信息，请参阅 `PromptType`。
- **sampling_params** – 请求的采样参数。
- **request_id** – 请求的唯一标识符。
- **lora_request** – 用于生成的 LoRA 请求（如果有）。
- **trace_headers** – OpenTelemetry 跟踪头。
- **prompt_adapter_request** – 用于生成的提示适配器请求（如果有）。
- **priority** – 请求的优先级。仅适用于优先级调度。

**生成：**

`LLMEngine`为请求生成的 RequestOutput 对象。

**详细信息：**

- 如果引擎未运行，则启动后台循环，该循环会迭代调用 `engine_step()` 来处理等待的请求。
- 将请求添加到引擎的 RequestTracker 中。在下一次后台循环中，该请求将被发送到底层引擎。同时，将创建一个相应的 AsyncStream。
- 等待来自 AsyncStream 的请求输出并生成它们。

**示例**

```plain
>>> # 请参考 entrypoints/api_server.py 获取完整示例。
>>>
>>> # 初始化引擎和示例输入
>>> # 注意这里的 engine_args 是 AsyncEngineArgs 实例
>>> engine = AsyncLLMEngine.from_engine_args(engine_args)
>>> example_input = {
>>>     "prompt": "What is LLM?",
>>>     "stream": False, # assume the non-streaming case
>>>     "temperature": 0.0,
>>>     "request_id": 0,
>>> }
>>>
>>> # 开始生成
>>> results_generator = engine.generate(
>>>    example_input["prompt"],
>>>    SamplingParams(temperature=example_input["temperature"]),
>>>    example_input["request_id"])
>>>
>>> # 获取结果
>>> final_output = None
>>> async for request_output in results_generator:
>>>     if await request.is_disconnected():
>>>         # Abort the request if the client disconnects.
>>>         await engine.abort(request_id)
>>>         # Return or raise an error
>>>         ...
>>>     final_output = request_output
>>>
>>> # 处理并返回最终输出
>>> ...
```

**async_ get\_decoding\_config() → DecodingConfig**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L1149)

获取 vLLM 引擎的解码配置。

**async_ get\_input\_preprocessor() → InputPreprocessor**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L691)

获取 vLLM 引擎的输入预处理器。

**async_ get\_lora\_config() → LoRAConfig**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L1157)

获取 vLLM 引擎的 LoRA 配置。

**async_ get\_model\_config() → ModelConfig**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L1141)

获取 vLLM 引擎的模型配置。

**async_ get\_parallel\_config() → ParallelConfig**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L1145)

获取 vLLM 引擎的并行配置。

**async_ get\_scheduler\_config() → SchedulerConfig**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L1153)

获取 vLLM 引擎的调度配置。

**async_ get\_tokenizer(_lora\_request: LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | TokenizerBase**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L694)

获取请求的适当分词器。

**async_ is\_sleeping() → [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")**

[[source]  ](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L1228)

检查引擎是否处于休眠状态。

**async_ reset\_prefix\_cache(_device: Device | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L1192)

重置前缀缓存。

**async static_ run\_engine\_loop(_engine\_ref: ReferenceType_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L781)

我们使用引擎的弱引用，这样正在运行的循环不会阻止引擎被垃圾回收。

**shutdown\_background\_loop() → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L716)

关闭后台循环。

在清理过程中需要调用此方法，以移除对 `self` 的引用，并正确释放异步 LLM 引擎持有的资源（例如执行器及其资源）。

**async_ sleep(_level: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") \= 1_) → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L1195)

让引擎休眠。

**start\_background\_loop() → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L700)

启动后台循环。

**async_ start\_profile() → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L1186)

开始分析引擎。

**async_ stop\_profile() → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L1189)

停止分析引擎。

**async_ wake\_up(_tags: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py#L1198)

唤醒引擎。
