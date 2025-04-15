---
title: 离线推理嵌入
---


源代码: [vllm-project/vllm](https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference_embedding.py)

```python
from vllm import LLM

# Sample prompts.
# 提示示例

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create an LLM.
# 创建一个 LLM
model = LLM(model="intfloat/e5-mistral-7b-instruct", enforce_eager=True)
# Generate embedding. The output is a list of EmbeddingRequestOutputs.
# 生成 embedding。输出是一个 EmbeddingRequestOutputs 列表

outputs = model.encode(prompts)
# Print the outputs.
# 打印输出

for output in outputs:
    print(output.outputs.embedding)  # list of 4096 floats
```


