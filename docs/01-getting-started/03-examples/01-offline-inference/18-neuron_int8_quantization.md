---
title: Neuron Int8 Quantization
---

[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/neuron_int8_quantization.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/neuron_int8_quantization.py)

```python
# SPDX-License-Identifier: Apache-2.0

import os

from vllm import LLM, SamplingParams

# creates XLA hlo graphs for all the context length buckets.
# 为所有上下文长度存储桶创建 XLA HLO 图。
os.environ['NEURON_CONTEXT_LENGTH_BUCKETS'] = "128,512,1024,2048"
# creates XLA hlo graphs for all the token gen buckets.
# 为所有 token gen buckets 创建 XLA HLO 图。
os.environ['NEURON_TOKEN_GEN_BUCKETS'] = "128,512,1024,2048"
# Quantizes neuron model weight to int8 ,
# The default config for quantization is int8 dtype.
# 将神经元模型权重量化为 int8
# 量化的默认配置为 int8 dtype。
os.environ['NEURON_QUANT_DTYPE'] = "s8"

# Sample prompts.
# 样本提示。
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
# 创建一个采样参数对象。
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
# 创建一个 LLM。
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_num_seqs=8,
    # The max_model_len and block_size arguments are required to be same as
    # max sequence length when targeting neuron device.
    # Currently, this is a known limitation in continuous batching support
    # in transformers-neuronx.
    # TODO(liangfu): Support paged-attention in transformers-neuronx.
    # max_model_len 和 block_size 参数必须与
    # 定位神经元设备时的最大序列长度。
    # 目前，这是连续批处理支持的已知限制
    # 在 transformers-Neuronx 中。
    # todo (liangfu) :在 transformers-Neuronx 中支持分页。
    max_model_len=2048,
    block_size=2048,
    # The device can be automatically detected when AWS Neuron SDK is installed.
    # The device argument can be either unspecified for automated detection,
    # or explicitly assigned.
    # 安装 AWS Neuron SDK 时可以自动检测到该设备。
    # 设备参数可以被未指定用于自动检测，
    # 或明确分配。
    device="neuron",
    quantization="neuron_quant",
    override_neuron_config={
        "cast_logits_dtype": "bfloat16",
    },
    tensor_parallel_size=2)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
# 从提示中生成文本。输出是 RequestOutput 对象的列表包含提示，生成的文本和其他信息的对象
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
# 打印输出。
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```
