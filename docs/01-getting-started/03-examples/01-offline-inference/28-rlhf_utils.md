---
title: Rlhf Utils
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/rlhf_utils.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_utils.py)

```python
# SPDX-License-Identifier: Apache-2.0
import torch


def stateless_init_process_group(master_address, master_port, rank, world_size,
                                 device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    """
    vLLM 提供 `StatelessProcessGroup` 来创建进程组，
    无需考虑 torch.distributed 中的全局进程组。
    建议先创建 `StatelessProcessGroup`，然后初始化
    外部（训练进程）与 vLLM 工作进程之间的数据平面通信（NCCL）。
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(host=master_address,
                                      port=master_port,
                                      rank=rank,
                                      world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class WorkerExtension:
    """
    The class for vLLM's worker to inherit from.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """
    """
    vLLM 工作进程的基类。
    通过定义扩展类，无论底层工作进程类是什么，代码都能正常工作。
    这种方式使代码能同时兼容 vLLM V0 和 V1。
    注意：我们在单独模块中定义此类，主模块应将完整限定名
    作为 `worker_extension_cls` 参数传递。
    """

    def init_weight_update_group(self, master_address, master_port,
                                 rank_offset, world_size):
        from vllm.distributed.parallel_state import get_world_group
        rank = get_world_group().rank + rank_offset
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(weight,
                                          src=0,
                                          stream=torch.cuda.current_stream())

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        """
        检查权重是否已更新为 0。
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(
                p, torch.zeros_like(p))
        return weights_updated


class ColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """
    """
    vLLM 工作进程在协同部署场景下的基类。
    通过定义扩展类，无论底层工作进程类是什么，代码都能正常工作。
    这种方式使代码能同时兼容 vLLM V0 和 V1。
    注意：我们在单独模块中定义此类，主模块应将完整限定名
    作为 `worker_extension_cls` 参数传递。
    """

    def report_device_id(self) -> str:
        from vllm.platforms import current_platform
        self.device_uuid = current_platform.get_device_uuid(self.device.index)
        return self.device_uuid

    def update_weights_from_ipc_handles(self, ipc_handles):
        handles = ipc_handles[self.device_uuid]
        device_id = self.device.index
        weights = []
        for name, handle in handles.items():
            func, args = handle
            list_args = list(args)
            # the key is to change device id to the current device id
            # in case two processes have different CUDA_VISIBLE_DEVICES
            # 关键是将设备 ID 改为当前设备 ID，
            # 以防两个进程有不同的 CUDA_VISIBLE_DEVICES
            list_args[6] = device_id
            tensor = func(*list_args)
            weights.append((name, tensor))
        self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        """
        检查权重是否已更新为0。
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(
                p, torch.zeros_like(p))
        return weights_updated

```
