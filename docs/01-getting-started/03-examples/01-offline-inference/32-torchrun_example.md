---
title: Torchrun Example
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/torchrun_example.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/torchrun_example.py)

````python
# SPDX-License-Identifier: Apache-2.0

"""
实验性支持使用 torchrun 进行张量并行推理，
动机和使用场景请参考 https://github.com/vllm-project/vllm/issues/11400
运行脚本命令：`torchrun --nproc-per-node=2 torchrun_example.py`，
参数 2 需与下方 `tensor_parallel_size` 保持一致。
单元测试见 `tests/distributed/test_torchrun_example.py`
"""

from vllm import LLM, SamplingParams

# 创建提示，在所有 rank 中相同
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# 创建采样参数，所有 rank 都相同
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 使用 `distributed_executor_backend="external_launcher"` 配置，
# 使当前 LLM 引擎/实例仅创建一个工作进程
llm = LLM(
    model="facebook/opt-125m",
    tensor_parallel_size=2,
    distributed_executor_backend="external_launcher",
)

outputs = llm.generate(prompts, sampling_params)

# 所有 rank 都将具有相同的输出
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, "
          f"Generated text: {generated_text!r}")

"""
更多使用技巧：

1. 跨所有进程(rank)传递控制消息时，使用 CPU 组：
   基于 GLOO 后端的 PyTorch 进程组(ProcessGroup)

```python
from vllm.distributed.parallel_state import get_world_group
cpu_group = get_world_group().cpu_group
torch_rank = dist.get_rank(group=cpu_group)
if torch_rank == 0:
    # 为 rank 0 执行特定操作，例如将结果保存到磁盘
```

2. 跨所有进程传输数据时，使用模型的设备组：
基于 NCCL 后端的 PyTorch 进程组

```python
from vllm.distributed.parallel_state import get_world_group
device_group = get_world_group().device_group
```

3. 在每个进程中直接访问模型，使用以下代码：

```python
llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
```

"""
