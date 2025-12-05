---
title: Distributed
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/distributed.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/distributed.py)

```python
# SPDX-License-Identifier: Apache-2.0

"""
此示例显示了如何使用 ray 数据进行离线批处理推断
在多节点群集上分布。
在 https://docs.ray.io/en/latest/data/data.html 中了解有关 ray 数据的更多信息
"""

from typing import Any

import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM, SamplingParams

assert Version(ray.__version__) >= Version(
    "2.22.0"), "Ray version must be at least 2.22.0"

# 创建一个采样参数对象。
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 每个实例设置张量并行性。
tensor_parallel_size = 1

# 设置数量的实例。每个实例都将使用 Tensor_parallel_size GPU。
num_instances = 1


# 创建一个以进行批处理推理的类。
class LLMPredictor:

    def __init__(self):
        # Create an LLM.
        # 创建一个 LLM。
        self.llm = LLM(model="meta-llama/Llama-2-7b-chat-hf",
                       tensor_parallel_size=tensor_parallel_size)

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, list]:
        # 从提示中生成文本。
        # 输出是包含提示的 RequestOutput 对象的列表，
        # 生成的文本和其他信息。
        outputs = self.llm.generate(batch["text"], sampling_params)
        prompt: list[str] = []
        generated_text: list[str] = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(' '.join([o.text for o in output.outputs]))
        return {
            "prompt": prompt,
            "generated_text": generated_text,
        }


# 从 S3读取一个文本文件。 ray 数据支持读取多个文件
# 来自云存储 (例如 JSONL，Parquet，CSV，二进制格式)。
ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")



# 对于 tensor_parallel_size> 1，我们需要为 vLLM 创建放置组
# 要使用。每个演员都必须拥有自己的安置小组。
def scheduling_strategy_fn():
    # One bundle per tensor parallel worker
    # 每张张量的平行工人一捆
    pg = ray.util.placement_group(
        [{
            "GPU": 1,
            "CPU": 1
        }] * tensor_parallel_size,
        strategy="STRICT_PACK",
    )
    return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
        pg, placement_group_capture_child_tasks=True))


resources_kwarg: dict[str, Any] = {}
if tensor_parallel_size == 1:
    # 对于 tensor_parallel_size == 1，我们只是设置 num_gpus = 1。
    resources_kwarg["num_gpus"] = 1
else:
    # 否则，我们必须设置 num_gpus = 0并提供
    # 一个将创建一个安置组的函数
    # 每个实例。
    resources_kwarg["num_gpus"] = 0
    resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

# 应用所有输入数据的批处理推断。
ds = ds.map_batches(
    LLMPredictor,
    # Set the concurrency to the number of LLM instances.
    # 将并发设置为 LLM 实例的数量。
    concurrency=num_instances,
    # Specify the batch size for inference.
    # 指定推理的批次大小。
    batch_size=32,
    **resources_kwarg,
)

# 窥视前10个结果。
# 注意:这是用于本地测试和调试。对于生产用例，
# 应该写出完整的结果，如下所示。
outputs = ds.take(limit=10)
for output in outputs:
    prompt = output["prompt"]
    generated_text = output["generated_text"]
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# ds.write_parquet("s3://<your-output-bucket>")
# 将推理输出数据作为 parquet 文件输出到 S3。
# 多个文件将写入输出目标，
# 每个任务将分别编写一个或多个文件。
#
# ds.write_parquet ( "s3:// <your-out-output-bucket>")

```
