---
title: Neuron
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/neuron.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/neuron.py)

```python
# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

# 样本提示。
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# 创建一个采样参数对象。
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 创建一个 LLM。
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_num_seqs=8,
    # max_model_len 和 block_size 参数必须与
    # 定位神经元设备时的最大序列长度。
    # 目前，这是连续批处理支持的已知限制
    # 在 transformers-Neuronx 中。
    # TODO(liangfu): 在 transformers-Neuronx 中支持分页。
    max_model_len=1024,
    block_size=1024,
    # 安装 AWS 神经元 SDK 时可以自动检测到该设备。
    # 设备参数可以被未指定用于自动检测，
    # 或明确分配。
    device="neuron",
    tensor_parallel_size=2)
# 从提示中生成文本。输出是 RequestOutput 对象的包含提示，生成的文本和其他信息的对象列表。
outputs = llm.generate(prompts, sampling_params)
# 打印输出。
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```
