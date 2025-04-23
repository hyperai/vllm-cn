---
title: 故障排除
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

#

本文档概述了一些可考虑的故障排除策略。如果您认为发现了一个错误，请先[搜索现有问题](https://github.com/vllm-project/vllm/issues?q=is%253Aissue)查看是否已报告。如果没有，请[提交新问题](https://github.com/vllm-project/vllm/issues/new/choose)，提供尽可能多的相关信息。

**注意**

调试问题后，请记得关闭任何定义的调试环境变量，或启动新的 shell 以避免残留的调试设置影响系统。否则，系统可能会因启用了调试功能而变慢。

## 模型下载卡顿

如果模型尚未下载到磁盘，vLLM 将从互联网下载，这可能需要时间并取决于您的网络连接。建议先使用 [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli) 下载模型，并将本地模型路径传递给 vLLM。这样可以隔离问题。

## 从磁盘加载模型卡顿

如果模型较大，从磁盘加载可能需要较长时间。请注意模型存储的位置。某些集群在节点间使用共享文件系统（例如分布式文件系统或网络文件系统），这可能较慢。最好将模型存储在本地磁盘中。此外，查看 CPU 内存使用情况，当模型过大时可能占用大量 CPU 内存，导致操作系统因频繁在磁盘和内存间交换而变慢。

> **注意**
> 要隔离模型下载和加载问题，可使用 `--load-format dummy` 参数跳过加载模型权重。这样可以检查模型下载和加载是否为瓶颈。

## 内存不足

如果模型过大无法放入单个 GPU，将出现内存不足（OOM）错误。考虑 [使用张量并行](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#distributed-serving) 将模型拆分到多个 GPU 上。此时，每个进程将读取整个模型并拆分为块，这会延长磁盘读取时间（与张量并行的规模成正比）。您可以使用 [examples/offline_inference/save_sharded_state.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/save_sharded_state.py) 将模型检查点转换为分片检查点。转换过程可能需要一些时间，但之后可以更快加载分片检查点。模型加载时间应保持恒定，不受张量并行规模影响。

## 生成质量已更改

在 v0.8.0 中，[Pull Request #12622](https://github.com/vllm-project/vllm/pull/12622#) 中更改了默认采样参数的来源。在 v0.8.0 之前，默认采样参数来自 vLLM 的中性默认值集。从 v0.8.0 开始，默认采样参数来自模型创建者提供的 `generation_config.json`。

在大多数情况下，这应该会带来更高质量的响应，因为模型创建者可能知道哪些采样参数最适合他们的模型。但是，在某些情况下，模型创建者提供的默认值可能会导致性能下降。

您可以通过尝试使用旧的默认值来检查是否发生这种情况，使用 `--generation-config vllm` 表示在线，使用 `generation_config="vllm"` 进行离线。如果在尝试此作后，您的生成质量有所提高，我们建议您继续使用 vLLM 默认值，并在 [https://huggingface.co](https://huggingface.co/) 上请求模型创建者更新其默认 `generation_config.json`，以便生成质量更好的生成。

## 启用更多日志

如果其他策略无法解决问题，vLLM 实例可能卡在某个地方。可使用以下环境变量帮助调试：

- `export VLLM_LOGGING_LEVEL=DEBUG` 启用更多日志。
- `export CUDA_LAUNCH_BLOCKING=1` 识别导致问题的 CUDA 内核。
- `export NCCL_DEBUG=TRACE` 启用 NCCL 的更多日志。
- `export VLLM_TRACE_FUNCTION=1` 记录所有函数调用以便在日志文件中检查崩溃或卡住的函数。

## 网络配置错误

如果网络配置复杂，vLLM 实例可能无法获取正确的 IP 地址。您可以在日志中找到类似 `DEBUG 06-10 21:32:17 parallel_state.py:88] world_size=8 rank=0 local_rank=0 distributed_init_method=tcp://xxx.xxx.xxx.xxx:54641 backend=nccl` 的条目，IP 地址应为正确的。如果错误，可使用环境变量 `export VLLM_HOST_IP=<您的 IP 地址>` 覆盖 IP 地址。

您可能还需要设置 `export NCCL_SOCKET_IFNAME=<您的网络接口>` 和 `export GLOO_SOCKET_IFNAME=<您的网络接口>` 以指定 IP 地址的网络接口。

## `self.graph.replay()` 附近的错误

如果 vLLM 崩溃且错误跟踪显示在 `vllm/worker/model_runner.py` 的 `self.graph.replay()` 附近，则为 CUDAGraph 内部的 CUDA 错误。要识别导致错误的特定 CUDA 操作，可在命令行添加 `--enforce-eager` 或向 `LLM` 类添加 `enforce_eager=True` 以禁用 CUDAGraph 优化并隔离导致错误的 CUDA 操作。

## 硬件/驱动错误

如果 GPU/CPU 通信无法建立，可使用以下 Python 脚本并按照说明确认 GPU/CPU 通信是否正常工作：

```plain
# Test PyTorch NCCL
# 测试 PyTorch NCCL
import torch
import torch.distributed as dist
dist.init_process_group(backend="nccl")
local_rank = dist.get_rank() % torch.cuda.device_count()
torch.cuda.set_device(local_rank)
data = torch.FloatTensor([1,] * 128).to("cuda")
dist.all_reduce(data, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
value = data.mean().item()
world_size = dist.get_world_size()
assert value == world_size, f"Expected {world_size}, got {value}"


print("PyTorch NCCL is successful!")


# Test PyTorch GLOO
# 测试 PyTorch GLOO
gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
cpu_data = torch.FloatTensor([1,] * 128)
dist.all_reduce(cpu_data, op=dist.ReduceOp.SUM, group=gloo_group)
value = cpu_data.mean().item()
assert value == world_size, f"Expected {world_size}, got {value}"


print("PyTorch GLOO is successful!")


if world_size <= 1:
    exit()


# Test vLLM NCCL, with cuda graph
# 测试 vLLM NCCL（使用 CUDA 图）
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


pynccl = PyNcclCommunicator(group=gloo_group, device=local_rank)
# pynccl is enabled by default for 0.6.5+,
# but for 0.6.4 and below, we need to enable it manually.
# keep the code for backward compatibility when because people
# prefer to read the latest documentation.
# v0.6.5+ 默认启用 pynccl，
# 但 v0.6.4 及以下版本需手动启用。
# 保留代码以向后兼容。
pynccl.disabled = False


s = torch.cuda.Stream()
with torch.cuda.stream(s):
    data.fill_(1)
    out = pynccl.all_reduce(data, stream=s)
    value = out.mean().item()
    assert value == world_size, f"Expected {world_size}, got {value}"


print("vLLM NCCL is successful!")


g = torch.cuda.CUDAGraph()
with torch.cuda.graph(cuda_graph=g, stream=s):
    out = pynccl.all_reduce(data, stream=torch.cuda.current_stream())


data.fill_(1)
g.replay()
torch.cuda.current_stream().synchronize()
value = out.mean().item()
assert value == world_size, f"Expected {world_size}, got {value}"


print("vLLM NCCL with cuda graph is successful!")


dist.destroy_process_group(gloo_group)
dist.destroy_process_group()
```

如果使用单节点测试，调整 `--nproc-per-node` 为要使用的 GPU 数量：

```go
NCCL_DEBUG=TRACE torchrun --nproc-per-node=<number-of-GPUs> test.py
```

如果使用多节点测试，调整 `--nproc-per-node` 和 `--nnodes` 并根据您的设置设置 `MASTER_ADDR` 为主节点的正确 IP 地址（所有节点均可访问）。然后运行：

```go
NCCL_DEBUG=TRACE torchrun --nnodes 2 --nproc-per-node=2 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR test.py
```

如果脚本成功运行，您将看到消息 `sanity check is successful!`。

如果测试脚本卡住或崩溃，通常表示硬件/驱动存在某些问题。您应联系系统管理员或硬件供应商寻求进一步帮助。作为常见解决方法，可尝试调整一些 NCCL 环境变量，例如 `export NCCL_P2P_DISABLE=1` 查看是否有效。请参阅 [NCCL 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) 获取更多信息。请仅将这些环境变量作为临时解决方法，因为它们可能影响系统性能。最佳解决方案仍是修复硬件/驱动以使测试脚本成功运行。

> **注意**
> 多节点环境比单节点更复杂。如果看到类似 `torch.distributed.DistNetworkError` 的错误，可能是网络/DNS 配置错误。此时可手动分配节点排名并通过命令行参数指定 IP：
> 在第一个节点运行： `NCCL_DEBUG=TRACE torchrun --nnodes 2 --nproc-per-node=2 --node-rank 0 --master_addr $MASTER_ADDR test.py`.
> 在第二个节点运行： `NCCL_DEBUG=TRACE torchrun --nnodes 2 --nproc-per-node=2 --node-rank 1 --master_addr $MASTER_ADDR test.py`.
> 根据您的设置调整 `--nproc-per-node`、`--nnodes` 和 `--node-rank`，确保在不同节点执行不同命令（使用不同的 `--node-rank`）。

## Python 多进程

### `RuntimeError` 异常

如果您在日志中看到如下警告：

```go
WARNING 12-11 14:50:37 multiproc_worker_utils.py:281] CUDA was previously
    initialized. We must use the `spawn` multiprocessing start method. Setting
    VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See
    https://docs.vllm.ai/en/latest/getting_started/troubleshooting.html#python-multiprocessing
    for more information.
```

或来自 Python 的如下错误：

```go
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.


        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:


            if __name__ == '__main__':
                freeze_support()
                ...


        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.


        To fix this issue, refer to the "Safe importing of main module"
        section in https://docs.python.org/3/library/multiprocessing.html
```

则必须更新 Python 代码，将 `vllm` 的使用包裹在 `if __name__ == '__main__':` 块中。例如，将以下代码：

```plain
import vllm


llm = vllm.LLM(...)
```

改为：

```plain
if __name__ == '__main__':
    import vllm


    llm = vllm.LLM(...)
```

## `torch.compile` 错误

vLLM 高度依赖 `torch.compile` 优化模型以获得更好性能，这引入了对 `torch.compile` 功能和 `triton` 库的依赖。默认情况下，我们使用 `torch.compile` [优化部分函数](https://github.com/vllm-project/vllm/pull/10406)。在运行 vLLM 前，可通过运行以下脚本检查 `torch.compile` 是否正常工作：

```plain
import torch


@torch.compile
def f(x):
    # a simple function to test torch.compile
    # 测试 torch.compile 的简单函数
    x = x + 1
    x = x * 2
    x = x.sin()
    return x


x = torch.randn(4, 4).cuda()
print(f(x))
```

如果从 `torch/_inductor` 目录引发错误，通常表示您使用的自定义 `triton` 库与 PyTorch 版本不兼容。示例请参阅[此问题](https://github.com/vllm-project/vllm/issues/12219)。

## 模型检查失败

如果看到如下错误：

```plain
  File "vllm/model_executor/models/registry.py", line xxx, in _raise_for_unsupported
    raise ValueError(
ValueError: Model architectures ['<arch>'] failed to be inspected. Please check the logs for more details.
```

表示 vLLM 未能导入模型文件。通常与 vLLM 构建中缺少依赖项或二进制文件过时有关。请仔细查看日志以确定根本原因。

## 模型不支持

如果看到如下错误：

```plain
Traceback (most recent call last):
...
  File "vllm/model_executor/models/registry.py", line xxx, in inspect_model_cls
    for arch in architectures:
TypeError: 'NoneType' object is not iterable
```

或：

```plain
  File "vllm/model_executor/models/registry.py", line xxx, in _raise_for_unsupported
    raise ValueError(
ValueError: Model architectures ['<arch>'] are not supported for now. Supported architectures: [...]
```

但您确认模型在[支持模型列表](https://docs.vllm.ai/en/latest/models/supported_models.html#supported-models)中，可能是 vLLM 的模型解析存在问题。此时，请按照[这些步骤](https://docs.vllm.ai/en/latest/serving/offline_inference.html#model-resolution)显式指定模型的 vLLM 实现。

## 无法推断设备类型

如果看到错误 `RuntimeError: Failed to infer device type`，表示 vLLM 未能推断运行时环境的设备类型。可查看[代码](https://github.com/vllm-project/vllm/blob/main/vllm/platforms/__init__.py)了解 vLLM 如何推断设备类型及为何未按预期工作。[此 PR](https://github.com/vllm-project/vllm/pull/14195#) 后，您还可设置环境变量 `VLLM_LOGGING_LEVEL=DEBUG` 查看更详细的日志以帮助调试。

## 已知问题

- 在 `v0.5.2`、`v0.5.3` 和 `v0.5.3.post1` 中，存在由 [zmq](https://github.com/zeromq/pyzmq/issues/2000) 引起的错误，可能因机器配置不同偶尔导致 vLLM 卡住。解决方案是升级到最新版 `vllm` 以包含 [修复](https://github.com/vllm-project/vllm/pull/6759#)。
- 为避免 NCCL [bug](https://github.com/NVIDIA/nccl/issues/1234)，所有 vLLM 进程将设置环境变量 `NCCL_CUMEM_ENABLE=0` 以禁用 NCCL 的 `cuMem` 分配器。这不影响性能，仅提供内存优势。当外部进程希望与 vLLM 进程建立 NCCL 连接时，也应设置此环境变量，否则环境不一致将导致 NCCL 卡住或崩溃，如 [RLHF 集成](https://github.com/OpenRLHF/OpenRLHF/pull/604)和[讨论](https://github.com/vllm-project/vllm/issues/5723#issuecomment-2554389656)中所述。
