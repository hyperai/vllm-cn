---
title: 离线推理 Neuron
---


源代码: [vllm-project/vllm](https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference_neuron.py)

```python
import os

from vllm import LLM, SamplingParams

# creates XLA hlo graphs for all the context length buckets.
# 为所有上下文长度桶创建 XLA HLO 图。
os.environ['NEURON_CONTEXT_LENGTH_BUCKETS'] = "128,512,1024,2048"
# creates XLA hlo graphs for all the token gen buckets.
# creates XLA hlo graphs for all the token gen buckets.

os.environ['NEURON_TOKEN_GEN_BUCKETS'] = "128,512,1024,2048"

# Sample prompts.
# 提示示例

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
# 创建 sampling params 对象
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
# Create an LLM.

llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_num_seqs=8,
    # The max_model_len and block_size arguments are required to be same as
    # max sequence length when targeting neuron device.
    # Currently, this is a known limitation in continuous batching support
    # in transformers-neuronx.
    # TODO(liangfu): Support paged-attention in transformers-neuronx.
    # `max_model_len` 和 `block_size` 参数需要与目标神经元设备的最大序列长度相同。
    # 目前，这是 transformers-neuronx 中连续批处理支持的已知限制。
    # TODO（liangfu）：在 transformers-neuronx 中支持分页注意力。

    max_model_len=2048,
    block_size=2048,
    # The device can be automatically detected when AWS Neuron SDK is installed.
    # The device argument can be either unspecified for automated detection,
    # or explicitly assigned.
    # 当安装了 AWS Neuron SDK 时，设备可以被自动检测。
    # 设备参数可以不指定以便自动检测，或者显式指定。

    device="neuron",
    tensor_parallel_size=2)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
# 从提示中生成文本。输出是一个 RequestOutput 列表，包含提示、生成文本和其他信息

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```


