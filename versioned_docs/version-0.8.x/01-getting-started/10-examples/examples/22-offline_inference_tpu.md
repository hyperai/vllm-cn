---
title: 离线推理 TPU
---


源代码: [vllm-project/vllm](https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference_tpu.py)

```python
from vllm import LLM, SamplingParams

prompts = [
    "A robot may not injure a human being",
    "It is only with the heart that one can see rightly;",
    "The greatest glory in living lies not in never falling,",
]
answers = [
    " or, through inaction, allow a human being to come to harm.",
    " what is essential is invisible to the eye.",
    " but in rising every time we fall.",
]
N = 1
# Currently, top-p sampling is disabled. `top_p` should be 1.0.
# 当前，top-p 采样被禁用。`top_p` 应设置为 1.0。

sampling_params = SamplingParams(temperature=0.7,
                                 top_p=1.0,
                                 n=N,
                                 max_tokens=16)

# Set `enforce_eager=True` to avoid ahead-of-time compilation.
# In real workloads, `enforace_eager` should be `False`.
# 将 `enforce_eager` 设置为 `True` 以避免提前编译。  
# 在实际工作负载中，`enforce_eager` 应设置为 `False`。

llm = LLM(model="google/gemma-2b", enforce_eager=True)
outputs = llm.generate(prompts, sampling_params)
for output, answer in zip(outputs, answers):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    assert generated_text.startswith(answer)
```


