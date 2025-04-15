---
title: 调试技巧
---


## 调试挂起与崩溃问题

当一个 vLLM 实例挂起或崩溃时，调试问题会非常困难。但请稍等，也有可能 vLLM 正在执行某些确实需要较长时间的任务：


* **下载模型**: 您的磁盘中是否已经下载了模型？如果没有，vLLM 将从互联网上下载模型，这可能需要很长时间。请务必检查互联网连接。最好先使用 [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli) 下载模型，然后使用模型的本地路径。这样可以就没有问题了。

* **从磁盘加载模型**: 如果模型很大，从磁盘加载模型可能需要很长时间。请注意存储模型的位置。一些集群在节点之间共享文件系统的速度可能很慢，例如分布式文件系统或网络文件系统。最好将模型存储在本地磁盘中。另外，还请注意 CPU 内存使用情况。当模型太大时，可能会占用大量 CPU 内存，这可能会降低操作系统的速度，因为它需要频繁地在磁盘和内存之间交换内存。

* **张量并行推理**: 如果模型太大而无法容纳在单个 GPU 中，您可能需要使用张量并行将模型拆分到多个 GPU 上。在这种情况下，每个进程都会读取整个模型并将其拆分成块，这会使磁盘读取时间更长（与张量并行度的大小成正比）。您可以使用[该脚本](https://docs.vllm.ai/en/latest/getting_started/examples/save_sharded_state.html)将模型检查点转换为分片检查点。转换过程可能需要一些时间，但之后您可以更快地加载分片检查点。无论张量并行度的大小如何，模型加载时间都应维持稳定。


如果您已经解决了上述问题，但 vLLM 实例仍然挂起，CPU 和 GPU 的利用率接近于零，那么 vLLM 实例可能卡在了某个地方。以下是一些有助于调试问题的提示：


* 设置环境变量 `export VLLM_LOGGING_LEVEL=DEBUG` 以打开更多日志记录。  

* 设置环境变量 `export CUDA_LAUNCH_BLOCKING=1` 以准确定位哪个 CUDA 内核引发了问题。  

* 设置环境变量 `export NCCL_DEBUG=TRACE` 以开启 NCCL 的更多日志记录。  

* 设置环境变量 `export VLLM_TRACE_FUNCTION=1`。vLLM 中的所有函数调用将被记录下来。检查这些日志文件，找出哪个函数崩溃或挂起。


通过更多日志记录，希望您能够找到问题的根本原因。


如果程序崩溃，并且错误追踪显示在 `vllm/worker/model_runner.py` 中的 `self.graph.replay()` 附近，那么这是一个发生在 cudagraph 内部的 CUDA 错误。要知道引发错误的具体 CUDA 操作，您可以在命令行中添加 `--enforce-eager`，或者在 `LLM` 类中设置 `enforce_eager=True`，以禁用 cudagraph 优化。通过这种方式，您可以准确定位导致错误的 CUDA 操作。


以下是一些可能导致挂起的常见问题: 


* 错误的网络设置：如果你的网络配置很复杂，vLLM 实例可能无法获取正确的 IP 地址。您可以找到类似 `DEBUG 06-10 21:32:17 parallel_state.py:88] world_size=8rank=0 local_rank=0Distributed_init_method=tcp://xxx.xxx.xxx.xxx:54641 backend=nccl` 这样的日志。IP 地址应该是正确的。如果不正确，请通过设置环境变量 `export VLLM_HOST_IP=your_ip_address` 来覆盖 IP 地址。您可能还需要设置 `export NCCL_SOCKET_IFNAME=your_network_interface` 和 `export GLOO_SOCKET_IFNAME=your_network_interface` 来指定 IP 地址的网络接口。

* 错误的硬件 / 驱动：无法建立 GPU/CPU 通信。您可以运行以下完整性检查脚本来查看 GPU/CPU 通信是否正常工作。

```python
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


# Test vLLM NCCL, with cuda graph
# 使用 cuda 图测试 vLLM NCCL


from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


pynccl = PyNcclCommunicator(group=gloo_group, device=local_rank)
pynccl.disabled = False


s = torch.cuda.Stream()
with torch.cuda.stream(s):
    data.fill_(1)
    pynccl.all_reduce(data, stream=s)
    value = data.mean().item()
    assert value == world_size, f"Expected {world_size}, got {value}"


print("vLLM NCCL is successful!")


g = torch.cuda.CUDAGraph()
with torch.cuda.graph(cuda_graph=g, stream=s):
    pynccl.all_reduce(data, stream=torch.cuda.current_stream())


data.fill_(1)
g.replay()
torch.cuda.current_stream().synchronize()
value = data.mean().item()
assert value == world_size, f"Expected {world_size}, got {value}"


print("vLLM NCCL with cuda graph is successful!")


dist.destroy_process_group(gloo_group)
dist.destroy_process_group()
```


**提示**


将脚本保存为`test.py` 。


如果您在单节点中进行测试，请使用 `NCCL_DEBUG=TRACE torchrun --nproc-per-node=8 test.py` 运行它，将 `--nproc-per-node` 调整为您想使用的 GPU 数量。


如果您正在使用多节点进行测试，请使用 `NCCL_DEBUG=TRACE torchrun --nnodes 2 --nproc-per-node=2 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR test.py` 运行它。根据您的设置调整 `--nproc-per-node` 和 `--nnodes`。确保 `MASTER_ADDR` 满足以下条件: 


* 是主节点的正确 IP 地址

* 所有节点均可访问

* 在运行脚本之前设置


如果脚本成功运行，您会看到消息 `sanity check is successful!` 。


如果问题仍然存在，请随时在 GitHub 上新建一个 [issue](https://github.com/vllm-project/vllm/issues/new/choose)，并详细描述问题、您的环境以及日志。


一些已知问题：


* 在`v0.5.2`、`v0.5.3`和 `v0.5.3.post1`中，存在由 [zmq](https://github.com/zeromq/pyzmq/issues/2000) 引起的错误，这可能会导致低概率挂起（大约 20 次挂起一次，具体取决于机器配置）。解决方案是升级到最新版本的 `vllm`以包含[修复](https://github.com/vllm-project/vllm/pull/6759)程序。


**警告**


在找到根本原因并解决问题后，请记得关闭上述定义的所有调试环境变量，或者简单地启动一个新 shell，以避免受到调试设置的影响。如果不这样做，系统可能会因为许多调试功能被打开而变慢。


