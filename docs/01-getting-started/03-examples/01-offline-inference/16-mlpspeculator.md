---
title: Mlpspeculator
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/mlpspeculator.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/mlpspeculator.py)

```python
# SPDX-License-Identifier: Apache-2.0

import gc
import time

from vllm import LLM, SamplingParams


def time_generation(llm: LLM, prompts: list[str],
                    sampling_params: SamplingParams):
    # 从提示中生成文本。输出是 RequestOutput 的包含提示，生成文本和其他信息的对象列表。
    # 首先预热
    llm.generate(prompts, sampling_params)
    llm.generate(prompts, sampling_params)
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()
    print((end - start) / sum([len(o.outputs[0].token_ids) for o in outputs]))
    # 打印输出。
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"text: {generated_text!r}")


if __name__ == "__main__":

    template = (
        "Below is an instruction that describes a task. Write a response "
        "that appropriately completes the request.\n\n### Instruction:\n{}"
        "\n\n### Response:\n")

    # 样本提示。
    prompts = [
        "Write about the president of the United States.",
    ]
    prompts = [template.format(prompt) for prompt in prompts]
    # 创建一个采样参数对象。
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)

    # 创建一个不使用规格解码的 LLM
    llm = LLM(model="meta-llama/Llama-2-13b-chat-hf")

    print("Without speculation")
    time_generation(llm, prompts, sampling_params)

    del llm
    gc.collect()

    # 与规格解码创建一个 LLM
    llm = LLM(
        model="meta-llama/Llama-2-13b-chat-hf",
        speculative_config={
            "model": "ibm-ai-platform/llama-13b-accelerator",
        },
    )

    print("With speculation")
    time_generation(llm, prompts, sampling_params)

```
