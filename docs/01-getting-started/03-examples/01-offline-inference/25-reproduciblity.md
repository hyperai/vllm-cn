---
title: Reproduciblity
---

[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/reproduciblity.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/reproduciblity.py)

```python
# SPDX-License-Identifier: Apache-2.0
import os

from vllm import LLM, SamplingParams

# 为了性能考虑，vllm 不能保证结果的默认情况下可重复性，
# 您需要做以下事情才能实现
# 可复现结果:
# 1.关闭多处理以使计划确定性。
# Note (Woosuk) :这是不需要的，对于 V0而言，这将被忽略。
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# 2.修复五十年据种子以获得可重复性。默认种子为 None，不可复现。
SEED = 42


# Note (Woosuk) :即使使用上述两个设置，vLLM 也仅提供
# 当它在相同的硬件和相同的 vLLM 版本上运行时，它的可重复性。
# 此外，在线服务 API ( "vLLM 服务") 不支持可重复性
# 因为几乎不可能在在线服务设置。

llm = LLM(model="facebook/opt-125m", seed=SEED)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```
