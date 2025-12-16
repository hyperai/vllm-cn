---
title: Prefix Caching
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/prefix_caching.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/prefix_caching.py)

```python
# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

# 注意:这只是一个正在运行的示例。用于基准测试，
# 请参阅基准 benchmarks/benchmark_prefix_caching.py

# 常见前缀。
prefix = (
    "You are an expert school principal, skilled in effectively managing "
    "faculty and staff. Draft 10-15 questions for a potential first grade "
    "Head Teacher for my K-12, all-girls', independent school that emphasizes "
    "community, joyful discovery, and life-long learning. The candidate is "
    "coming in for a first-round panel interview for a 8th grade Math "
    "teaching role. They have 5 years of previous teaching experience "
    "as an assistant teacher at a co-ed, public school with experience "
    "in middle school math teaching. Based on these information, fulfill "
    "the following paragraph: ")

# 样本提示。
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

generating_prompts = [prefix + prompt for prompt in prompts]

# 创建一个采样参数对象。
sampling_params = SamplingParams(temperature=0.0)

# 创建一个没有前缀缓存的 LLM 作为基线。
regular_llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.4)

print("Results without `enable_prefix_caching`")

# 从提示中生成文本。输出是 RequestOutput 对象的包含提示，生成的文本和其他信息的对象列表。
outputs = regular_llm.generate(generating_prompts, sampling_params)

regular_generated_texts = []
# 打印输出。
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    regular_generated_texts.append(generated_text)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("-" * 80)

# 破坏 LLM 对象并释放 GPU 内存。
del regular_llm
cleanup_dist_env_and_memory()

# 使用启用前缀缓存创建一个 LLM。
prefix_cached_llm = LLM(model="facebook/opt-125m",
                        enable_prefix_caching=True,
                        gpu_memory_utilization=0.4)

# 预热，以便计算共享的提示 KV 缓存。
prefix_cached_llm.generate(generating_prompts[0], sampling_params)

# 使用前缀缓存生成。
outputs = prefix_cached_llm.generate(generating_prompts, sampling_params)

print("Results with `enable_prefix_caching`")

cached_generated_texts = []
# 打印输出。您应该看到与以前相同的输出。
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    cached_generated_texts.append(generated_text)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("-" * 80)

# 比较结果并显示加速
generated_same = all([
    regular_generated_texts[i] == cached_generated_texts[i]
    for i in range(len(prompts))
])
print(f"Generated answers are the same: {generated_same}")

```
