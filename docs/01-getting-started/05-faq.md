---
title: Frequently Asked Questions
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

# 常见问题

> Q：如何使用 OpenAI API 在单个端口上提供多个模型？
> 
> A：假设您指的是使用与 OpenAI 兼容的服务器同时提供多个模型的服务，目前是不支持的。您可以同时运行多个服务器实例（每个实例为不同的模型提供服务），并通过另一层来相应地将传入的请求路由到正确的服务器。

---

> Q：使用哪种模型进行离线推理嵌入？
> 
> A：你可以尝试 [e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) 和 [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)，更多支持的模型列在 [这里](#supported-models)。

通过提取隐藏状态，vLLM 可以自动将 [Llama-3-8](https://huggingface.co/meta-llama/Meta-Llama-3-8B)[B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) 和 [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) 等文本生成模型转换为嵌入模型，但这些模型的效果预计不如专门针对嵌入任务训练的模型。

---

> Q：在 vLLM 中，提示的输出是否会在不同运行中有所不同？
> 
> A：是的。vLLM 不保证输出 tokens 的稳定对数概率 (logprobs)。由于 Torch 操作中的数值不稳定性或在批处理变化时 Torch 操作的非确定性行为，logprobs 可能会有所变化。有关更多详细信息，请参见[数值准确性部分](https://pytorch.org/docs/stable/notes/numerical_accuracy.html#batched-computations-or-slice-computations)。

在 vLLM 中，由于其他并发请求、批大小的变化或在推测解码中的批扩展等因素，相同的请求可能会以不同的方式进行批处理。这些批处理的变化，加上 Torch 操作的数值不稳定性，可能导致每一步的 logit/logprob 值略有不同。这些差异可能会累积，导致对不同的 tokens 进行采样。一旦对不同的 tokens 进行采样，就可能会有进一步的偏差。

## 缓解策略

- 为了提高稳定性和减少方差，请使用 float32。请注意，这将需要更多内存。
- 如果使用 bfloat16，切换到 float16 也可能有所帮助。
- 使用请求种子有助于在温度大于 0 时实现更稳定的生成，但由于精度差异，可能仍会出现不一致的情况。
