---
title: AsyncLLMEngine
---


>class vllm.AsyncLLMEngine(*args, log_requests: [bool](https://docs.python.org/3/library/functions.html#bool) = True, start_engine_loop: [bool](https://docs.python.org/3/library/functions.html#bool) = True, **kwargs) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/engine/async_llm_engine.html#AsyncLLMEngine)


这是一个用于 [LLMEngine](https://docs.vllm.ai/en/latest/dev/engine/llm_engine.html#vllm.LLMEngine) 的异步包装器。


该类用于包装 [LLMEngine](https://docs.vllm.ai/en/latest/dev/engine/llm_engine.html#vllm.LLMEngine) 类以使其支持异步。它使用 asyncio 创建一个后台循环，不断处理传入的请求。当等待队列中有请求时，通过 generate 方式启用 [LLMEngine](https://docs.vllm.ai/en/latest/dev/engine/llm_engine.html#vllm.LLMEngine) 。generate 方法将 [LLMEngine](https://docs.vllm.ai/en/latest/dev/engine/llm_engine.html#vllm.LLMEngine) 的输出生成给调用者。


**参数：**

* **log_requests** – 是否记录请求。

* **start_engine_loop** – 如果为 True，则运行引擎的后台任务将在生成调用中自动启动。

* ***args** – [LLMEngine](https://docs.vllm.ai/en/latest/dev/engine/llm_engine.html#vllm.LLMEngine) 的参数。

* ****kwargs** – [LLMEngine](https://docs.vllm.ai/en/latest/dev/engine/llm_engine.html#vllm.LLMEngine) 的参数。


>async abort(request_id: [str](https://docs.python.org/3/library/stdtypes.html#str)) → [None](https://docs.python.org/3/library/constants.html#None) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/engine/async_llm_engine.html#AsyncLLMEngine.abort)


中止请求。


中止已提交的请求。如果请求已完成或未找到，则此方法不进行操作。


**参数：**

* **request_id** – 请求的唯一 ID。

>async check_health() → [None](https://docs.python.org/3/library/constants.html#None) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/engine/async_llm_engine.html#AsyncLLMEngine.check_health)

如果引擎不正常，则会引发错误。


>async encode(inputs: [str](https://docs.python.org/3/library/stdtypes.html#str) | [TextPrompt](https://docs.vllm.ai/en/latest/dev/offline_inference/llm_inputs.html#vllm.inputs.TextPrompt) | [TokensPrompt](https://docs.vllm.ai/en/latest/dev/offline_inference/llm_inputs.html#vllm.inputs.TokensPrompt) | ExplicitEncoderDecoderPrompt, pooling_params: PoolingParams, request_id: [str](https://docs.python.org/3/library/stdtypes.html#str), lora_request: LoRARequest | [None](https://docs.python.org/3/library/constants.html#None) = None, trace_headers: [Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, priority: [int](https://docs.python.org/3/library/functions.html#int) = 0) ) → [AsyncGenerator](https://docs.python.org/3/library/typing.html#typing.AsyncGenerator)[EmbeddingRequestOutput, [None](https://docs.python.org/3/library/constants.html#None)] 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/engine/async_llm_engine.html#AsyncLLMEngine.encode)


为嵌入模型的请求生成输出。


生成请求的输出。该方法是一个协程。它将请求添加到 LLMEngine 的等待队列中，并将 LLMEngine 的输出流式传输给调用者。


**参数：**

* **inputs** – 提供给 LLM 的输入。有关每个输入格式的更多详细信息，请参阅 [PromptInputs](https://docs.vllm.ai/en/latest/dev/offline_inference/llm_inputs.html#vllm.inputs.PromptInputs)。

* **pooling_params** – 请求的池化参数。

* **request_id** – 请求的唯一 ID。

* **lora_request** – 用于生成的 LoRA 请求（如果有）。

* **trace_headers** – OpenTelemetry 跟踪头。

* **priority** – 请求的优先级。仅在优先级调度时适用。


**生成结果：**

LLMEngine 为请求输出的 *EmbeddingRequestOutput* 对象。


**详细****信息：**

* 如果引擎未运行，则启动后台循环，该循环会迭代调用 `engine_step()` 来处理等待的请求。

* 将请求添加到引擎的 *RequestTracker*。在下一个后台循环中，该请求将被发送到基础引擎。此外，还将创建相应的 *AsyncStream*。

* 等待来自 *AsyncStream* 的请求输出并生成它们。


**示例**

```python
# Please refer to entrypoints/api_server.py for
# the complete example.
# 请参阅 entrypoints/api_server.py 获得完整案例


# initialize the engine and the example input
# 初始化引擎和案例输入
engine = AsyncLLMEngine.from_engine_args(engine_args)
example_input = {
    "input": "What is LLM?",
    "request_id": 0,
}


# start the generation
# 开始生成
results_generator = engine.encode(
   example_input["input"],
   PoolingParams(),
   example_input["request_id"])


# get the results
# 获取结果
final_output = None
async for request_output in results_generator:
    if await request.is_disconnected():
        # Abort the request if the client disconnects.
        # 如果客户端失联，中止请求
        await engine.abort(request_id)
        # Return or raise an error
        # 返回或发出一个 error
        ...
    final_output = request_output


# Process and return the final output
# 处理并返回追踪结果
...
```


>async engine_step(virtual_engine: [int](https://docs.python.org/3/library/functions.html#int)) → [bool](https://docs.python.org/3/library/functions.html#bool) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/engine/async_llm_engine.html#AsyncLLMEngine.engine_step)

启动引擎处理等待的请求。

如果有正在进行的请求，则返回 True。


>classmethod from_engine_args(engine_args: AsyncEngineArgs, engine_config: EngineConfig | [None](https://docs.python.org/3/library/constants.html#None) = None, start_engine_loop: [bool](https://docs.python.org/3/library/functions.html#bool) = True, usage_context: UsageContext = UsageContext.ENGINE_CONTEXT, stat_loggers: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), StatLoggerBase] | [None](https://docs.python.org/3/library/constants.html#None) = None) → [AsyncLLMEngine](https://docs.vllm.ai/en/latest/dev/engine/async_llm_engine.html#vllm.AsyncLLMEngine) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/engine/async_llm_engine.html#AsyncLLMEngine.from_engine_args)

使用引擎参数创建异步 LLM 引擎。


>async generate(inputs: [str](https://docs.python.org/3/library/stdtypes.html#str) | [TextPrompt](https://docs.vllm.ai/en/latest/dev/offline_inference/llm_inputs.html#vllm.inputs.TextPrompt) | [TokensPrompt](https://docs.vllm.ai/en/latest/dev/offline_inference/llm_inputs.html#vllm.inputs.TokensPrompt) | ExplicitEncoderDecoderPrompt, sampling_params: [SamplingParams](https://docs.vllm.ai/en/latest/dev/sampling_params.html#vllm.SamplingParams), request_id: [str](https://docs.python.org/3/library/stdtypes.html#str), lora_request: LoRARequest | [None](https://docs.python.org/3/library/constants.html#None) = None, trace_headers: [Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)] | [None](https://docs.python.org/3/library/constants.html#None) = None, prompt_adapter_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None) = None, priority: [int](https://docs.python.org/3/library/functions.html#int) = 0) → [AsyncGenerator](https://docs.python.org/3/library/typing.html#typing.AsyncGenerator)[RequestOutput, [None](https://docs.python.org/3/library/constants.html#None)] 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/engine/async_llm_engine.html#AsyncLLMEngine.generate)

为一个请求生成输出。


为一个请求生成输出。该方法是一个协程。它将请求添加到 LLMEngine 的等待队列中，并将输出从 LLMEngine 流式传输给调用者。


**参数：**

* **prompt –** 提供给 LLM 的输入。有关每个输入格式的更多详细信息，请参阅 [PromptType](https://docs.vllm.ai/en/latest/dev/offline_inference/llm_inputs.html#vllm.inputs.PromptType)。

* **in****p****u****t****s** – LLM 的输入。有关每个输入格式的更多详细信息，请参阅 PromptInputs。

* **sampling_params** – 请求的采样参数。

* **request_id** – 请求的唯一 ID。

* **lora_request** – 用于生成的 LoRA 请求（如果有）。

* **trace_headers** – OpenTelemetry 跟踪头。

* **priority** – 请求的优先级。仅在优先级调度时适用。


**生成结果：**

LLMEngine 为请求输出的 *RequestOutput* 对象。


**详细信息：**

* 如果引擎未运行，则启动后台循环，该循环会迭代调用`engine_step()` 来处理等待的请求。

* 将请求添加到引擎的 *RequestTracker*。在下一个后台循环中，该请求将被发送到基础引擎。此外，还将创建相应的 *AsyncStream*。

* 等待来自 *AsyncStream* 的请求输出并生成它们。


**示例：**

```plain
# Please refer to entrypoints/api_server.py for
# the complete example.
# 请参阅 entrypoints/api_server.py 获取完整案例


# initialize the engine and the example input
# 初始化引擎和案例输入
engine = AsyncLLMEngine.from_engine_args(engine_args)
example_input = {
    "prompt": "What is LLM?",
    "stream": False, # assume the non-streaming case
    "temperature": 0.0,
    "request_id": 0,
}


# start the generation
# 开始生成
results_generator = engine.generate(
   example_input["prompt"],
   SamplingParams(temperature=example_input["temperature"]),
   example_input["request_id"])


# get the results
# 获取结果
final_output = None
async for request_output in results_generator:
    if await request.is_disconnected():
        # Abort the request if the client disconnects.
        # 如果客户端失联，中止请求
        await engine.abort(request_id)
        # Return or raise an error
        # 返回或发出一个 error
        ...
    final_output = request_output


# Process and return the final output
# 处理并返回追踪结果
...
```


>async get_decoding_config() → DecodingConfig  
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/engine/async_llm_engine.html#AsyncLLMEngine.get_decoding_config)

获取 vLLM 引擎的解码配置。


>async get_lora_config() → LoRAConfig 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/engine/async_llm_engine.html#AsyncLLMEngine.get_lora_config)

获取 vLLM 引擎的 lora 配置。


>async get_model_config() → ModelConfig 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/engine/async_llm_engine.html#AsyncLLMEngine.get_model_config)

获取 vLLM 引擎的模型配置。


>async get_parallel_config() → ParallelConfig 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/engine/async_llm_engine.html#AsyncLLMEngine.get_parallel_config)

获取 vLLM 引擎的并行配置。


>async get_scheduler_config() → SchedulerConfig 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/engine/async_llm_engine.html#AsyncLLMEngine.get_scheduler_config)

获取 vLLM 引擎的调度配置。


>async static run_engine_loop(engine_ref: weakref) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/engine/async_llm_engine.html#AsyncLLMEngine.run_engine_loop)

我们对引擎使用弱引用，这样运行中的循环就不会阻止引擎被垃圾回收。


>shutdown_background_loop() → [None](https://docs.python.org/3/library/constants.html#None)  
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/engine/async_llm_engine.html#AsyncLLMEngine.shutdown_background_loop)

关闭后台循环。

在清理过程中需要调用此方法，以移除对 self 的引用，并正确地释放 async LLM 引擎所持有的资源（例如，执行程序及其资源）。


>start_background_loop() → [None](https://docs.python.org/3/library/constants.html#None) 
[[源代码]](https://docs.vllm.ai/en/latest/_modules/vllm/engine/async_llm_engine.html#AsyncLLMEngine.start_background_loop)

启动后台循环。

