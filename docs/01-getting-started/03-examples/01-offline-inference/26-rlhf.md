---
title: Rlhf
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/rlhf.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf.py)

```python
# SPDX-License-Identifier: Apache-2.0

"""
一个基于 vLLM 的 RLHF 简单实现演示，灵感来源于
OpenRLHF 框架 https://github.com/OpenRLHF/OpenRLHF 。
该设计采用训练进程 (training processes) 与推理进程 (inference processes)
分离的方案，它们运行在不同的 GPU 上。
训练进程向推理进程发送提示 (prompts) 以生成数据，
同时通过将模型权重从训练进程广播 (broadcast) 到推理进程
来实现模型权重的同步。
注意：本演示仅展示单个训练实例 (training instance) 和单个
推理实例 (inference instance) 的简单场景。
实际应用中可能存在多个训练实例和多个推理实例。
完整实现请参考 OpenRLHF 框架。
"""
import os

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rlhf_utils import stateless_init_process_group
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port


class MyLLM(LLM):

    def __init__(self, *args, **kwargs):
        # a hack to make the script work.
        # stop ray from manipulating CUDA_VISIBLE_DEVICES
        # at the top-level
        # 临时解决方案：确保脚本正常运行
        # 禁止 Ray 在顶层修改 CUDA_VISIBLE_DEVICES 环境变量
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)


"""
开始训练过程，在这里我们使用 HuggingFace Transformer
作为在 GPU 0 上保存模型的示例。
"""

train_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
train_model.to("cuda:0")

"""
启动推理过程，我们使用 vLLM 在 GPU 1和 GPU 2。有关如何使用 ray 的详细信息，
请参考 ray 文档 https://docs.ray.io/en/latest/。
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
ray.init()

pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())
scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

"""
启动 vLLM 推理引擎。
在这里，我们使用 `enforce_eager` 减少开始时间。
"""
llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(MyLLM).remote(
    model="facebook/opt-125m",
    enforce_eager=True,
    worker_extension_cls="rlhf_utils.WorkerExtension",
    tensor_parallel_size=2,
    distributed_executor_backend="ray",
)

# 从提示中生成文本。
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0)

outputs = ray.get(llm.generate.remote(prompts, sampling_params))

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, "
          f"Generated text: {generated_text!r}")

# 设置训练进程与推理引擎之间的通信
master_address = get_ip()
master_port = get_open_port()

handle = llm.collective_rpc.remote("init_weight_update_group",
                                   args=(master_address, master_port, 1, 3))

model_update_group = stateless_init_process_group(master_address, master_port,
                                                  0, 3, torch.device("cuda:0"))
ray.get(handle)

# 模拟训练，修改模型的权重。
for name, p in train_model.named_parameters():
    p.data.zero_()

# 同步从训练过程到推理引擎的权重。
for name, p in train_model.named_parameters():
    handle = llm.collective_rpc.remote("update_weight",
                                       args=(name, p.dtype, p.shape))
    model_update_group.broadcast(p, src=0, stream=torch.cuda.current_stream())
    ray.get(handle)

# 检查权重是否更新。
assert all(ray.get(llm.collective_rpc.remote("check_weights_changed")))

# 使用更新的模型生成文本，它们会胡说八道
# 因为权重都是零。
outputs_updated = ray.get(llm.generate.remote(prompts, sampling_params))
for output in outputs_updated:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, "
          f"Generated text: {generated_text!r}")

```
