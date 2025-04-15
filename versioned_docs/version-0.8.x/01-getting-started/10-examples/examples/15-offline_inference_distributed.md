---
title: 离线推理分布式
---


源代码: [vllm-project/vllm](https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference_distributed.py)

```python
"""
This example shows how to use Ray Data for running offline batch inference
distributively on a multi-nodes cluster.

Learn more about Ray Data in https://docs.ray.io/en/latest/data/data.html
此示例演示了如何使用 Ray Data 在多节点集群上分布式地运行离线批量推断。
了解更多关于 Ray Data 的信息，请访问 https://docs.ray.io/en/latest/data/data.html
"""

from typing import Any, Dict, List

import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM, SamplingParams

assert Version(ray.__version__) >= Version(
    "2.22.0"), "Ray version must be at least 2.22.0"

# Create a sampling params object.
# 创建 sampling params 对象

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Set tensor parallelism per instance.
# 每个实例都设置 tensor parallelism

tensor_parallel_size = 1

# Set number of instances. Each instance will use tensor_parallel_size GPUs.
num_instances = 1
# 设置实例数量。每个实例将使用 tensor_parallel_size 个 GPU



# Create a class to do batch inference.
# 创建一个用于批量推断的类

class LLMPredictor:

    def __init__(self):
        # Create an LLM.
        # Create an LLM.

        self.llm = LLM(model="meta-llama/Llama-2-7b-chat-hf",
                       tensor_parallel_size=tensor_parallel_size)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts.
        # 从提示中生成文本
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        # 输出是一个 RequestOutput 列表，包含提示、生成文本和其他信息

        outputs = self.llm.generate(batch["text"], sampling_params)
        prompt: List[str] = []
        generated_text: List[str] = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(' '.join([o.text for o in output.outputs]))
        return {
            "prompt": prompt,
            "generated_text": generated_text,
        }


# Read one text file from S3. Ray Data supports reading multiple files
# from cloud storage (such as JSONL, Parquet, CSV, binary format).
# 从 S3 读取一个文本文件。Ray Data 支持从云存储中读取多个文件
#（如 JSONL、Parquet、CSV、二进制格式）。

ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")


# For tensor_parallel_size > 1, we need to create placement groups for vLLM
# to use. Every actor has to have its own placement group.
# 对于 tensor_parallel_size > 1，我们需要为 vLLM 创建放置组
# 每个 actor 都必须有自己的放置组。

def scheduling_strategy_fn():
    # One bundle per tensor parallel worker
    # 每个 tensor parallel worker 对应一个 bundle
    pg = ray.util.placement_group(
        [{
            "GPU": 1,
            "CPU": 1
        }] * tensor_parallel_size,
        strategy="STRICT_PACK",
    )
    return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
        pg, placement_group_capture_child_tasks=True))


resources_kwarg: Dict[str, Any] = {}
if tensor_parallel_size == 1:
    # For tensor_parallel_size == 1, we simply set num_gpus=1.
    # 对于 tensor_parallel_size == 1，我们只需将 num_gpus 设置为 1
    resources_kwarg["num_gpus"] = 1
else:
    # Otherwise, we have to set num_gpus=0 and provide
    # a function that will create a placement group for
    # each instance.
    # 否则，我们必须将 num_gpus 设置为 0，并提供一个函数，该函数将为每个实例
    # 创建一个 placement group。
    resources_kwarg["num_gpus"] = 0
    resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

# Apply batch inference for all input data.
# 对所有输入数据应用批接口

ds = ds.map_batches(
    LLMPredictor,
    # Set the concurrency to the number of LLM instances.
    # 设置 concurrency 为 LLM 的实例数量
    concurrency=num_instances,
    # Specify the batch size for inference.
    # 指定推理的批处理大小
    batch_size=32,
    **resources_kwarg,
)

# Peek first 10 results.
# 查看前10个结果

# NOTE: This is for local testing and debugging. For production use case,
# one should write full result out as shown below.
# 注意：这仅用于本地测试和调试。对于生产使用场景，应该按照下面所示的方式将完整结果写出。

outputs = ds.take(limit=10)
for output in outputs:
    prompt = output["prompt"]
    generated_text = output["generated_text"]
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# Write inference output data out as Parquet files to S3.
# Multiple files would be written to the output destination,
# and each task would write one or more files separately.
# 将推断输出数据写出为 Parquet 文件到 S3。多个文件将被写入到输出目标，
# 每个任务将单独写入一个或多个文件。

#
# ds.write_parquet("s3://<your-output-bucket>")

```


