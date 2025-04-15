---
title: 输入处理管道
---


1. 输入数据被传递给 [LLMEngine](https://docs.vllm.ai/en/latest/dev/engine/llm_engine.html#vllm.LLMEngine)（或 [AsyncLLMEngine](https://docs.vllm.ai/en/latest/dev/engine/async_llm_engine.html#vllm.AsyncLLMEngine)）。

2. 如有必要，对数据进行分词处理。

3. 使用 [INPUT_REGISTRY.process_input](https://docs.vllm.ai/en/latest/dev/input_processing/model_inputs_index.html#vllm.inputs.registry.InputRegistry.process_input) 处理输入。

   * 例如，添加占位符 token 以预留多模态嵌入的 KV 缓存。

4. 将处理后的输入发送给 `ExecutorBase`。

5. 通过 `WorkerBase` 将输入分发给 `ModelRunnerBase`。

6. 如果数据包含多模态数据，使用 [MULTIMODAL_REGISTRY.map_input](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#vllm.multimodal.MultiModalRegistry.map_input) 将其转换为关键字参数。

   * 例如，将 [PIL.Image.Image](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image) 输入转换为视觉模型的像素值。

### 

