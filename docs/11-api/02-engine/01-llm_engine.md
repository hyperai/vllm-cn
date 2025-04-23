---
title: LLMEngine
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

# LLMEngine

**\*class\*\*\***vllm.\***\*LLMEngine\*\***(**\***vllm*config: VllmConfig**\***,**\***executor_class:\*\**[Type](https://docs.python.org/3/library/typing.html#typing.Type)_**[ExecutorBase]**\***,**\***log_stats:**\*[bool](https://docs.python.org/3/library/functions.html#bool)**,**\***usage_context: UsageContext**\*\***=**\* \***UsageContext.ENGINE_CONTEXT**\***,**\***stat_loggers:**_[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)_**[**_[str](https://docs.python.org/3/library/stdtypes.html#str)_**, StatLoggerBase] |**_[None](https://docs.python.org/3/library/constants.html#None) ***=**\* \***None**\***,**\***input_registry: InputRegistry**\*\***=**\* \***INPUT_REGISTRY**\***,**\***mm_registry:***[MultiModalRegistry](https://docs.vllm.ai/en/latest/api/multimodal/registry.html#vllm.multimodal.registry.MultiModalRegistry) ***=**\* \***MULTIMODAL_REGISTRY**\***,**\***use_cached_outputs:***[bool](https://docs.python.org/3/library/functions.html#bool) \_**=**\* \***False**\*\*\*)\*\*

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L123)

一个接收请求并生成文本的 LLM 引擎。

这是 vLLM 引擎的主要类。它接收来自客户端的请求，并从 LLM 生成文本。它包括一个分词器、一个语言模型（可能分布在多个 GPU 上），以及为中间状态（即 KV 缓存）分配的 GPU 内存空间。该类利用迭代级调度和高效的内存管理来最大化服务吞吐量。

`LLM` 类包装了该类以进行离线批量推理，而 `AsyncLLMEngine` 类包装了该类以进行在线服务。

配置参数源自 `EngineArgs`。（参见[引擎参数](https://docs.vllm.ai/en/latest/serving/engine_args.html#engine-args)）

**参数:**

- **model_config** – 与 LLM 模型相关的配置。

- **cache_config** – 与 KV 缓存内存管理相关的配置。

- **parallel_config** – 与分布式执行相关的配置。

- **scheduler_config** – 与请求调度器相关的配置。

- **device_config** – 与设备相关的配置。

- **lora_config** (_可选_) – 与多 LoRA 服务相关的配置。

- **speculative_config** (_可选_) – 与推测解码相关的配置。

- **executor_class** – 用于管理分布式执行的模型执行器类。

- **prompt_adapter_config** (_可选_) – 与提示适配器服务相关的配置。

- **log_stats** – 是否记录统计信息。

- **usage_context** – 指定的入口点，用于使用信息收集。

**DO_VALIDATE_OUTPUT\*\*\***:**_[ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)_**[**_[bool](https://docs.python.org/3/library/functions.html#bool)_**] = False\*\*\*

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L123)

标志，用于切换是否验证请求输出的类型。

**abort_request\*\***(**\***request*id:\*\**[str](https://docs.python.org/3/library/stdtypes.html#str)_**|**_[Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)_**[**_[str](https://docs.python.org/3/library/stdtypes.html#str)\_**]**\***)\***\*→\*\*[None](https://docs.python.org/3/library/constants.html#None)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L872)

中止具有给定 ID 的请求。

**参数:**

**request_id** – 要中止的请求的 ID(s)。

**详细信息:**

- 请参阅 `Scheduler` 类中的 `abort_seq_group()`。

**示例**

```plain
>>> # initialize engine and add a request with request_id
>>> # 初始化引擎并添加一个带有 request_id 的请求
>>> request_id = str(0)
>>> # abort the request
>>> # 中止请求
>>> engine.abort_request(request_id)
```

**add_request\*\***(**\***request*id:**\*[str](https://docs.python.org/3/library/stdtypes.html#str)**,**\***prompt: PromptType**\***,**\***params:\*\**[SamplingParams](https://docs.vllm.ai/en/latest/api/inference_params.html#vllm.SamplingParams)_**|**\*[PoolingParams](https://docs.vllm.ai/en/latest/api/inference_params.html#vllm.PoolingParams)**,**\***arrival_time:**_[float](https://docs.python.org/3/library/functions.html#float)_**|**_[None](https://docs.python.org/3/library/constants.html#None) ***=**\* \***None**\***,**\***lora_request: LoRARequest |***[None](https://docs.python.org/3/library/constants.html#None) ***=**\* \***None**\***,**\***trace_headers: Mapping[***[str](https://docs.python.org/3/library/stdtypes.html#str)_**,**_[str](https://docs.python.org/3/library/stdtypes.html#str)_**] |**_[None](https://docs.python.org/3/library/constants.html#None) ***=**\* \***None**\***,**\***prompt_adapter_request: PromptAdapterRequest |***[None](https://docs.python.org/3/library/constants.html#None) ***=**\* \***None**\***,**\***priority:***[int](https://docs.python.org/3/library/functions.html#int) \_**=**\* \***0**\***)\***\*→\*\*[None](https://docs.python.org/3/library/constants.html#None)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/utils.py#L672)

**add_request\*\***(**\***request*id:**\*[str](https://docs.python.org/3/library/stdtypes.html#str)**,\***\*\*\*\*\*\***,**\***inputs: PromptType**\***,**\***params:\*\**[SamplingParams](https://docs.vllm.ai/en/latest/api/inference_params.html#vllm.SamplingParams)_**|**\*[PoolingParams](https://docs.vllm.ai/en/latest/api/inference_params.html#vllm.PoolingParams)**,**\***arrival_time:**_[float](https://docs.python.org/3/library/functions.html#float)_**|**_[None](https://docs.python.org/3/library/constants.html#None) ***=**\* \***None**\***,**\***lora_request: LoRARequest |***[None](https://docs.python.org/3/library/constants.html#None) ***=**\* \***None**\***,**\***trace_headers: Mapping[***[str](https://docs.python.org/3/library/stdtypes.html#str)_**,**_[str](https://docs.python.org/3/library/stdtypes.html#str)_**] |**_[None](https://docs.python.org/3/library/constants.html#None) ***=**\* \***None**\***,**\***prompt_adapter_request: PromptAdapterRequest |***[None](https://docs.python.org/3/library/constants.html#None) ***=**\* \***None**\***,**\***priority:***[int](https://docs.python.org/3/library/functions.html#int) \_**=**\* \***0**\***)\***\*→\*\*[None](https://docs.python.org/3/library/constants.html#None)

将请求添加到引擎的请求池中。

请求被添加到请求池中，并将在调用 `engine.step()` 时由调度器处理。具体的调度策略由调度器决定。

**参数:**

- **request_id** – 请求的唯一 ID。

- **prompt** – 提供给 LLM 的提示。有关每个输入格式的更多详细信息，请参阅 `PromptType`。

- **params** – 采样或池化的参数。`SamplingParams` 用于文本生成。`PoolingParams` 用于池化。

- **arrival_time** – 请求的到达时间。如果为 None，则使用当前的单调时间。

- **lora_request** – 要添加的 LoRA 请求。

- **trace_headers** – OpenTelemetry 跟踪头。

- **prompt_adapter_request** – 要添加的提示适配器请求。

- **priority** – 请求的优先级。仅适用于优先级调度。

**详细信息:**

- 如果 arrival_time 为 None，则将其设置为当前时间。

- 如果 prompt_token_ids 为 None，则将其设置为编码后的提示。

- 创建 `n` 个 `Sequence` 对象。

- 从 `Sequence` 列表中创建一个 `SequenceGroup` 对象。

- 将 `SequenceGroup` 对象添加到调度器中。

**示例**

```plain
>>> # initialize engine
>>> # 初始化引擎
>>> engine = LLMEngine.from_engine_args(engine_args)
>>> # set request arguments
>>> # 设置请求参数
>>> example_prompt = "Who is the president of the United States?"
>>> sampling_params = SamplingParams(temperature=0.0)
>>> request_id = 0
>>>
>>> # add the request to the engine
>>> # 将请求添加到引擎
>>> engine.add_request(
>>>    str(request_id),
>>>    example_prompt,
>>>    SamplingParams(temperature=0.0))
>>> # continue the request processing
>>> # 继续请求处理
>>> ...
```

**do_log_stats\*\***(**\***scheduler*outputs: SchedulerOutputs |\*\**[None](https://docs.python.org/3/library/constants.html#None) ***=**\* \***None**\***,**\***model_output:***[List](https://docs.python.org/3/library/typing.html#typing.List)_**[SamplerOutput] |**_[None](https://docs.python.org/3/library/constants.html#None) ***=**\* \***None**\***,**\***finished_before:***[List](https://docs.python.org/3/library/typing.html#typing.List)_**[**_[int](https://docs.python.org/3/library/functions.html#int)_**] |**_[None](https://docs.python.org/3/library/constants.html#None) ***=**\* \***None**\***,**\***skip:***[List](https://docs.python.org/3/library/typing.html#typing.List)_**[**_[int](https://docs.python.org/3/library/functions.html#int)_**] |**_[None](https://docs.python.org/3/library/constants.html#None) \_**=**\* \***None**\***)\***\*→\*\*[None](https://docs.python.org/3/library/constants.html#None)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L1603)

在没有活动请求时强制记录日志。

**\*classmethod\*\*\***from*engine_args\***\*(\*\*\***engine_args: EngineArgs**\***,**\***usage_context: UsageContext**\*\***=**\* \***UsageContext.ENGINE_CONTEXT**\***,**\***stat_loggers:\*\**[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)_**[**_[str](https://docs.python.org/3/library/stdtypes.html#str)_**, StatLoggerBase] |**_[None](https://docs.python.org/3/library/constants.html#None) \_**=**\* \***None**\***)\***\*→\*\*[LLMEngine](https://docs.vllm.ai/en/latest/api/engine/llm_engine.html#vllm.LLMEngine)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L482)

从引擎参数创建 LLM 引擎。

**get_decoding_config\*\***()\***\*→ DecodingConfig**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L901)

获取解码配置。

**get_lora_config\*\***()\***\*→ LoRAConfig**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L909)

获取 LoRA 配置。

**get_model_config\*\***()\***\*→ ModelConfig**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L893)

获取模型配置。

**get_num_unfinished_requests\*\***()\***\*→**[int](https://docs.python.org/3/library/functions.html#int)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L913)

获取未完成请求的数量。

**get_parallel_config\*\***()\***\*→ ParallelConfig**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L897)

获取并行配置。

**get_scheduler_config\*\***()\***\*→ SchedulerConfig**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L905)

获取调度器配置。

**has_unfinished_requests\*\***()\***\*→**[bool](https://docs.python.org/3/library/functions.html#bool)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L918)

如果有未完成的请求，则返回 True。

**has_unfinished_requests_for_virtual_engine\*\***(**\***virtual_engine:**\*[int](https://docs.python.org/3/library/functions.html#int)**)\***\*→**[bool](https://docs.python.org/3/library/functions.html#bool)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L923)

如果虚拟引擎有未完成的请求，则返回 True。

**reset_prefix_cache\*\***()\***\*→**[bool](https://docs.python.org/3/library/functions.html#bool)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L930)

重置所有设备的前缀缓存。

**step\*\***()\***\*→**[List](https://docs.python.org/3/library/typing.html#typing.List)**[RequestOutput | PoolingRequestOutput]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L1268)

执行一次解码迭代并返回新生成的结果。

![图片](/img/docs/v1-API/01-llm_engine_1.png)

step 函数的概述。

**详细信息:**

- 步骤 1：调度要在下一次迭代中执行的序列以及要交换/复制出的 token 块。

  - 根据调度策略，序列可能会被抢占/重新排序。

  - 序列组（SG）指的是从同一提示生成的一组序列。

- 步骤 2：调用分布式执行器来执行模型。

- 步骤 3：处理模型输出。主要包括：

  - 解码相关输出。

  - 根据其采样参数（是否使用 beam_search）更新调度的序列组。

  - 释放已完成的序列组。

- 最后，创建并返回新生成的结果。

**示例**

```plain
>>> # Please see the example/ folder for more detailed examples.
>>> # 请参阅 example/ 文件夹以获取更详细的示例。
>>>
>>> # initialize engine and request arguments
>>> # 初始化引擎和请求参数
>>> engine = LLMEngine.from_engine_args(engine_args)
>>> example_inputs = [(0, "What is LLM?",
>>>    SamplingParams(temperature=0.0))]
>>>
>>> # Start the engine with an event loop
>>> # 启动引擎并进入事件循环
>>> while True:
>>>     if example_inputs:
>>>         req_id, prompt, sampling_params = example_inputs.pop(0)
>>>         engine.add_request(str(req_id),prompt,sampling_params)
>>>
>>>     # continue the request processing
>>>     # 继续请求处理
>>>     request_outputs = engine.step()
>>>     for request_output in request_outputs:
>>>         if request_output.finished:
>>>             # return or show the request output
>>>             # 返回或显示请求输出
>>>
>>>     if not (engine.has_unfinished_requests() or example_inputs):
>>>         break
```
