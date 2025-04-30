---
title: Neuron Int8 Quantization
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/neuron_int8_quantization.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/neuron_int8_quantization.py)

```python
# SPDX-License-Identifier: Apache-2.0

import os

from vllm import LLM, SamplingParams

# 为所有上下文长度存储桶创建 XLA HLO 图。
os.environ['NEURON_CONTEXT_LENGTH_BUCKETS'] = "128,512,1024,2048"
# 为所有 token gen buckets 创建 XLA HLO 图。
os.environ['NEURON_TOKEN_GEN_BUCKETS'] = "128,512,1024,2048"
# 将神经元模型权重量化为 int8
# 量化的默认配置为 int8 dtype。
os.environ['NEURON_QUANT_DTYPE'] = "s8"

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
    # todo (liangfu) :在 transformers-Neuronx 中支持分页。
    max_model_len=2048,
    block_size=2048,
    # 安装 AWS Neuron SDK 时可以自动检测到该设备。
    # 设备参数可以被未指定用于自动检测，
    # 或明确分配。
    device="neuron",
    quantization="neuron_quant",
    override_neuron_config={
        "cast_logits_dtype": "bfloat16",
    },
    tensor_parallel_size=2)

# 从提示中生成文本。输出是 RequestOutput 对象的列表包含提示，生成的文本和其他信息的对象
outputs = llm.generate(prompts, sampling_params)
# 打印输出。
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```
