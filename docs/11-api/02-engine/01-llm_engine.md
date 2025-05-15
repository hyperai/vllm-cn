---
title: LLMEngine
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

**class_ vllm.LLMEngine(_vllm\_config: VllmConfig_, _executor\_class: [Type](https://docs.python.org/3/library/typing.html#typing.Type "(in Python v3.13)")\[ExecutorBase\]_, _log\_stats: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")_, _usage\_context: UsageContext \= UsageContext.ENGINE\_CONTEXT_, _stat\_loggers: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), StatLoggerBase\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _input\_registry: InputRegistry \= INPUT\_REGISTRY_, _mm\_registry: [MultiModalRegistry](https://docs.vllm.ai/en/v0.8.4_a/api/multimodal/registry.html#vllm.multimodal.registry.MultiModalRegistry "vllm.multimodal.registry.MultiModalRegistry") \= MULTIMODAL\_REGISTRY_, _use\_cached\_outputs: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= False_)**

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

**DO\_VALIDATE\_OUTPUT_: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar "(in Python v3.13)")\[[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")\]_ _\= False**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L123)

标志，用于切换是否验证请求输出的类型。

**abort\_request(_request\_id: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")\]_) → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L872)

中止具有给定 ID 的请求。

**参数:**

**request_id** – 要中止的请求的 ID(s)。

**详细信息:**

- 请参阅 `Scheduler` 类中的 `abort_seq_group()`。

**示例**

```plain
>>> # 初始化引擎并添加一个带有 request_id 的请求
>>> request_id = str(0)
>>> # 中止请求
>>> engine.abort_request(request_id)
```

**add\_request(_request\_id: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _prompt: PromptType_, _params: [SamplingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.SamplingParams "vllm.SamplingParams") | [PoolingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.PoolingParams "vllm.PoolingParams")_, _arrival\_time: [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _lora\_request: LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _trace\_headers: Mapping\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_adapter\_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _priority: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") \= 0_) → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/utils.py#L672)

**add\_request(_request\_id: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _\*_, _inputs: PromptType_, _params: [SamplingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.SamplingParams "vllm.SamplingParams") | [PoolingParams](https://docs.vllm.ai/en/v0.8.4_a/api/inference_params.html#vllm.PoolingParams "vllm.PoolingParams")_, _arrival\_time: [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _lora\_request: LoRARequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _trace\_headers: Mapping\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _prompt\_adapter\_request: PromptAdapterRequest | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _priority: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") \= 0_) → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")**

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
>>> # 初始化引擎
>>> engine = LLMEngine.from_engine_args(engine_args)
>>> # 设置请求参数
>>> example_prompt = "Who is the president of the United States?"
>>> sampling_params = SamplingParams(temperature=0.0)
>>> request_id = 0
>>>
>>> # 将请求添加到引擎
>>> engine.add_request(
>>>    str(request_id),
>>>    example_prompt,
>>>    SamplingParams(temperature=0.0))
>>> # 继续请求处理
>>> ...
```

**do\_log\_stats(_scheduler\_outputs: SchedulerOutputs | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _model\_output: [List](https://docs.python.org/3/library/typing.html#typing.List "(in Python v3.13)")\[SamplerOutput\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _finished\_before: [List](https://docs.python.org/3/library/typing.html#typing.List "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _skip: [List](https://docs.python.org/3/library/typing.html#typing.List "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L1603)

在没有活动请求时强制记录日志。

**classmethod_ from\_engine\_args(_engine\_args: EngineArgs_, _usage\_context: UsageContext \= UsageContext.ENGINE\_CONTEXT_, _stat\_loggers: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), StatLoggerBase\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [LLMEngine](https://docs.vllm.ai/en/v0.8.4_a/api/engine/llm_engine.html#vllm.LLMEngine "vllm.engine.llm_engine.LLMEngine")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L482)

从引擎参数创建 LLM 引擎。

**get\_decoding\_config() → DecodingConfig**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L901)

获取解码配置。

**get\_lora\_config() → LoRAConfig**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L909)

获取 LoRA 配置。

**get\_model\_config() → ModelConfig**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L893)

获取模型配置。

**get\_num\_unfinished\_requests() → [int](https://docs.python.org/3/library/functions.html#int)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L913)

获取未完成请求的数量。

**get\_parallel\_config() → ParallelConfig**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L897)

获取并行配置。

**get\_scheduler\_config() → SchedulerConfig**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L905)

获取调度器配置。

**has\_unfinished\_requests() → [bool](https://docs.python.org/3/library/functions.html#bool)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L918)

如果有未完成的请求，则返回 True。

**has\_unfinished\_requests\_for\_virtual\_engine(_virtual\_engine: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_) → [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L923)

如果虚拟引擎有未完成的请求，则返回 True。

**reset\_prefix\_cache(_device: Device | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py#L930)

重置所有设备的前缀缓存。

**step() → [List](https://docs.python.org/3/library/typing.html#typing.List "(in Python v3.13)")\[RequestOutput | PoolingRequestOutput\]**

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
>>> # 请参阅 example/ 文件夹以获取更详细的示例。
>>>
>>> # 初始化引擎和请求参数
>>> engine = LLMEngine.from_engine_args(engine_args)
>>> example_inputs = [(0, "What is LLM?",
>>>    SamplingParams(temperature=0.0))]
>>>
>>> # 启动引擎并进入事件循环
>>> while True:
>>>     if example_inputs:
>>>         req_id, prompt, sampling_params = example_inputs.pop(0)
>>>         engine.add_request(str(req_id),prompt,sampling_params)
>>>
>>>     # 继续请求处理
>>>     request_outputs = engine.step()
>>>     for request_output in request_outputs:
>>>         if request_output.finished:
>>>             # 返回或显示请求输出
>>>
>>>     if not (engine.has_unfinished_requests() or example_inputs):
>>>         break
```
