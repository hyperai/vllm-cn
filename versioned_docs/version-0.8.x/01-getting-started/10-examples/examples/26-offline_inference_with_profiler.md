---
title: 启用分析器的离线推理
---


[源代码](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_with_profiler.py)

```python
import os


from vllm import LLM, SamplingParams


# enable torch profiler, can also be set on cmd line
# 启用 torch 分析器，也可以在命令行中设置
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"


# Sample prompts.
# 示例提示
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
# 创建一个 LLM
llm = LLM(model="facebook/opt-125m", tensor_parallel_size=1)


llm.start_profile()


# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
# 从提示中生成文本。输出是一个 RequestOutput 列表，包含提示、生成文本和其他信息
outputs = llm.generate(prompts, sampling_params)


llm.stop_profile()


# Print the outputs.
# 打印输出
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```


