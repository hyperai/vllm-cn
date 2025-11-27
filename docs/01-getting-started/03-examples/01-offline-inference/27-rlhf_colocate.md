---
title: Rlhf Colocate
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/rlhf_colocate.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_colocate.py)

```python
# SPDX-License-Identifier: Apache-2.0

"""
一个简单的演示，展示如何将 vLLM 工作进程与训练执行器（training actors）
协同部署在同一 GPU上，适用于类 RLHF 应用。

关键要点：
- 通过正确设置 VLLM_RAY_PER_WORKER_GPUS 和 VLLM_RAY_BUNDLE_INDICES，
  使用 Ray 控制 vLLM 工作进程的部署位置
- 使用 CUDA-IPC 传递张量，因为在同一 GPU 上存在多个进程时 NCCL 无法正常工作
"""
import os

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM


class MyLLM(LLM):

    def __init__(self, *args, bundle_indices: list, **kwargs):

        # 临时方案使脚本能运行
        # 阻止Ray在顶层操作CUDA_VISIBLE_DEVICES
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # 每个工作进程将使用 0.4 个 GPU，这样我们可以在同一 GPU 上调度 2 个实例
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "0.4"
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(
            map(str, bundle_indices))
        print(f"creating LLM with bundle_indices={bundle_indices}")
        super().__init__(*args, **kwargs)


class RayTrainingActor:

    def __init__(self):

        # ray 将 CUDA_VISIBLE_DEVICES 设置为分配的 GPU
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        self.model.to("cuda:0")
        for name, p in self.model.named_parameters():
            p.data.zero_()
        torch.cuda.synchronize()
        # get_device_uuid 的参数是
        # 可见设备中 GPU 的索引
        from vllm.platforms import current_platform
        self.device_uuid = current_platform.get_device_uuid(0)

    def report_device_id(self) -> str:
        return self.device_uuid

    def get_weight_ipc_handles(self):
        from torch.multiprocessing.reductions import reduce_tensor
        data = {}
        for name, p in self.model.named_parameters():

            # 训练执行器（training actor）可能只拥有部分权重，
            # 需要从所有执行器进行 all-gather 操作获取完整权重。
            # 出于演示目的，此处我们假设所有训练执行器都拥有完整权重。
            data[name] = reduce_tensor(p.detach())
        return {self.device_uuid: data}


# ray 管理4 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
ray.init()

# 我们需要将 vLLM 实例和训练执行器（training actor）协同部署在同一组 GPU 上
# 具体部署方案如下：
# GPU 0 和 1：训练执行器 0、1 和 vLLM 实例 0（TP=2）
# GPU 2 和 3：训练执行器 2、3 和 vLLM 实例 1（TP=2）

pg = placement_group([{"GPU": 1, "CPU": 0}] * 4)
ray.get(pg.ready())
print(f"placement group has bundles {pg.bundle_specs=}")

training_actors = []
training_actor_device_ids = []
inference_engines = []
inference_engine_device_ids = []

for bundle_index in [0, 1, 2, 3]:
    training_actor = ray.remote(
        num_cpus=0,
        num_gpus=0.4,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_index,
        ),
    )(RayTrainingActor).remote()
    training_actors.append(training_actor)

for bundle_index, training_actor in enumerate(training_actors):
    device_id = ray.get(training_actor.report_device_id.remote())
    print(f"training actor {bundle_index} is on {device_id}")
    training_actor_device_ids.append(device_id)

for (i, bundle_indices) in enumerate([[0, 1], [2, 3]]):

    # and cause unexpected behaviors.
    # 重要:创建 vLLM 实例时，我们需要
    # 确保目标 GPU 上没有 GPU 活动，
    # 否则，它们将干扰 vLLM 内存分析，
    # 并引起意外的行为。
    llm = ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
        ),
    )(MyLLM).remote(
        model="facebook/opt-125m",
        enforce_eager=True,
        worker_extension_cls="rlhf_utils.ColocateWorkerExtension",
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.4,
        bundle_indices=bundle_indices,
    )
    inference_engines.append(llm)
    # don't call any method on the inference engine here,
    # otherwise it will block until the vLLM instance is created.
    # 在此处的推理引擎上不要调用任何方法，
    # 否则，它将锁定直到创建 vLLM 实例。

for i, llm in enumerate(inference_engines):
    inference_engine_device_ids.append(
        ray.get(llm.collective_rpc.remote("report_device_id", args=tuple())))
    print(f"inference engine {i} is on {inference_engine_device_ids[-1]}")

# 检查部署情况
# 前两个训练执行器(training actors)应当
# 与第一个推理引擎(inference engine)部署在同一GPU上
assert training_actor_device_ids[:2] == inference_engine_device_ids[0]

# 最后两个训练执行器(training actors)应当
# 与第二个推理引擎(inference engine)部署在同一GPU上
assert training_actor_device_ids[2:] == inference_engine_device_ids[1]

print("gather all the IPC handles from the training actors")
ipc_handles = {}
for actor in training_actors:
    ipc_handles.update(ray.get(actor.get_weight_ipc_handles.remote()))

print("update the weights of the inference engines")
for llm in inference_engines:
    ray.get(
        llm.collective_rpc.remote("update_weights_from_ipc_handles",
                                  args=(ipc_handles, )))
print("check if the weights are updated")
for llm in inference_engines:
    assert ray.get(
        llm.collective_rpc.remote("check_weights_changed", args=tuple()))

```
