---
title: API Client
---


源代码: [vllm-project/vllm](https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference.py)

```python
from vllm import LLM, SamplingParams

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
llm = LLM(model="facebook/opt-125m")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
# 从提示中生成文本。输出是一个 RequestOutput 列表，包含提示、生成文本和其他信息

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
# 打印输出
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```


