---
title: 离线推理 MlpSpeculator
---


源代码: [vllm-project/vllm](https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference_mlpspeculator.py)

```python
import gc
import time
from typing import List

from vllm import LLM, SamplingParams


def time_generation(llm: LLM, prompts: List[str],
                    sampling_params: SamplingParams):
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    # 从提示中生成文本。输出是一个 RequestOutput 列表，包含提示、生成文本和其他信息
    # Warmup first
    # 第一步暖机
    llm.generate(prompts, sampling_params)
    llm.generate(prompts, sampling_params)
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()
    print((end - start) / sum([len(o.outputs[0].token_ids) for o in outputs]))
    # Print the outputs.
    # 打印输出
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"text: {generated_text!r}")


if __name__ == "__main__":

    template = (
        "Below is an instruction that describes a task. Write a response "
        "that appropriately completes the request.\n\n### Instruction:\n{}"
        "\n\n### Response:\n")

    # Sample prompts.
    # 提示示例

    prompts = [
        "Write about the president of the United States.",
    ]
    prompts = [template.format(prompt) for prompt in prompts]
    # Create a sampling params object.
    # 创建 sampling params 对象
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)

    # Create an LLM without spec decoding
    # 创建一个 LLM，不带 spec decoding

    llm = LLM(model="meta-llama/Llama-2-13b-chat-hf")

    print("Without speculation")
    time_generation(llm, prompts, sampling_params)

    del llm
    gc.collect()

    # Create an LLM with spec decoding
    # 创建一个 LLM，不带 spec decoding

    llm = LLM(
        model="meta-llama/Llama-2-13b-chat-hf",
        speculative_model="ibm-fms/llama-13b-accelerator",
        # These are currently required for MLPSpeculator decoding
        # 这些当前是 MLPSpeculator 解码所必需的。
        use_v2_block_manager=True,
    )

    print("With speculation")
    time_generation(llm, prompts, sampling_params)
```


