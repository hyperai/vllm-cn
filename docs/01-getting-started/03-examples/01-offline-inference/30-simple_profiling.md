---
title: Simple Profiling
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/simple_profiling.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/simple_profiling.py)

```python
# SPDX-License-Identifier: Apache-2.0

import os
import time

from vllm import LLM, SamplingParams

# enable torch profiler, can also be set on cmd line
# 启用 torch 分析器，也可以在命令行设置
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

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

if __name__ == "__main__":

    # Create an LLM.
    # 创建一个 LLM。
    llm = LLM(model="facebook/opt-125m", tensor_parallel_size=1)

    llm.start_profile()

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    # 从提示中生成文本。输出是 RequestOutput 的包含提示，生成文本和其他信息的对象列表。
    outputs = llm.generate(prompts, sampling_params)

    llm.stop_profile()

    # Print the outputs.
    # 打印输出。
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.
    # 添加一个缓冲区，在后台过程中等待 profiling(如果 MP 为 ON) 完成分析输出。
    time.sleep(10)

```
