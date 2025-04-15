---
title: FP8 E5M2 KV 缓存
---


int8/int4 量化方案需要更多的 GPU 内存进行存储，这会降低预期的 GPU 内存优势。而 FP8 数据格式保留了 2~3 个尾数位，能够实现 float、fp16、bfloat16 与 fp8 之间的相互转换。


以下是如何启用此功能的示例：

```python
from vllm import LLM, SamplingParams


# Sample prompts.
# 提示示例。
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


# Create a sampling params object.
# 创建一个 Sampling Params 对象。
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


# Create an LLM.
# 创建一个 LLM。
llm = LLM(model="facebook/opt-125m", kv_cache_dtype="fp8")


# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
# 从提示中生成文本。输出是一个 RequestOutput 列表，包含提示、生成文本和其他信息
outputs = llm.generate(prompts, sampling_params)


# Print the outputs.
# 打印输出。
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```


