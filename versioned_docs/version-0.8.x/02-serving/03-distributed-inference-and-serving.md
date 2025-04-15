---
title: 分布式推理和服务
---

## 如何决定分布式推理策略？

在详细介绍分布式推理和服务之前，我们首先明确何时使用分布式推理以及有哪些可用的策略。以下是常见的做法： 


* **单 GPU（无分布式推理）**: 如果您的模型可以在单个 GPU 中运行，那么您可能不需要使用分布式推理。只需使用单个 GPU 即可运行推理。

* **单节点多 GPU（张量并行推理）**: 如果您的模型太大而无法在单个 GPU 中运行，但可以在具有多个 GPU 的单个节点中运行，则可以使用张量并行。张量并行大小是您要使用的 GPU 数量。例如，如果单个节点中有 4 个 GPU，则可以将张量并行大小设置为 4。

* **多节点多 GPU（张量并行加管道并行推理）**: 如果您的模型太大而无法在单个节点中运行，您可以将张量并行与管道并行结合使用。张量并行大小是每个节点要使用的 GPU 数量，管道并行大小是要使用的节点数量。例如，如果 2 个节点中有 16 个 GPU（每个节点 8 个 GPU），则可以将张量并行大小设置为 8，将管道并行大小设置为 2。


简而言之，您应该增加 GPU 数量和节点数量，直到有足够的 GPU 内存来容纳模型。张量并行大小应为每个节点中的 GPU 数量，管道并行大小应为节点数量。


在添加了足够的 GPU 和节点来容纳模型后，您可以先运行 vLLM，它会打印一些日志，例如 `# GPU blocks: 790` 。将数字乘以 `16` （块大小），您可以大致得到当前配置上能够处理的最大 tokens 数量。如果这个数字不令人满意，例如你想要更高的吞吐量，可以进一步增加 GPU 或节点的数量，直到块的数量足够为止。


**注意：**

有一种特殊情况：如果模型适合在具有多个 GPU 的单个节点中运行，但 GPU 的数量无法均匀划分模型大小，则可以使用管道并行性，它将模型沿层分割并支持不均匀分割。在这种情况下，张量并行大小应为 1，管道并行大小应为 GPU 数量。


## 分布式推理和服务的详细信息

vLLM 支持分布式张量并行推理和服务。目前，我们支持 [Megatron-LM 的张量并行算法](https://arxiv.org/pdf/1909.08053.pdf)。我们还支持将管道并行作为在线服务的测试版功能。我们使用 [Ray](https://github.com/ray-project/ray) 或 python 的原生多进程来管理分布式运行时。在单节点部署时可以使用多进程，多节点推理目前需要 Ray。


当未在 Ray 放置组中运行时，并且同一节点上有足够的 GPU 可用于配置的 `tensor_parallel_size` ，则默认情况下将使用多进程，否则将使用 Ray。这个默认设置可以通过 `LLM` 类的 `distributed-executor-backend` 参数或 API 服务器的 `--distributed-executor-backend` 参数来覆盖。将其设置为 `mp` （用于多进程） 或 `ray` （用于 Ray）。对于多进程情况，不需要安装 Ray。


要使用 `LLM` 类运行多 GPU 推理，请将 `tensor_parallel_size` 参数设置为要使用的 GPU 数量。例如，要在 4 个 GPU 上运行推理: 

```python
from vllm import LLM
llm = LLM(`facebook/opt-13b`, tensor_parallel_size=4)
output = llm.generate(`San Franciso is a`)
```


要运行多 GPU 服务，请在启动服务器时传入 `--tensor-parallel-size` 参数。例如，要在 4 个 GPU 上运行 API 服务器: 

```plain
vllm serve facebook/opt-13b \
    --tensor-parallel-size 4
```


您还可以另外指定 `--pipeline-parallel-size` 以启用管道并行性。例如，要在 8 个 GPU 上运行具有管道并行性和张量并行性的 API 服务器: 

```plain
vllm serve gpt2 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2
```


**注意：**

管道并行是一项测试功能，仅支持在线服务以及 LLaMa、GPT2、Mixtral、Qwen、Qwen2 和 Nemotron 风格模型。


## 多节点推理和服务

如果单个节点没有足够的 GPU 来容纳模型，您可以使用多个节点运行模型。确保所有节点上的执行环境相同是非常重要的，包括模型路径、Python 环境等。推荐的方法是使用 docker 镜像来确保相同的环境，并通过将它们映射到相同的 docker 配置来隐藏主机的异构性。


第一步是启动容器并将它们组织成一个集群。我们提供了一个辅助[脚本](https://github.com/vllm-project/vllm/tree/main/examples/run_cluster.sh)来启动集群。


选择一个节点作为头节点，然后运行以下命令：

```plain
bash run_cluster.sh \
                  vllm/vllm-openai \
                  ip_of_head_node \
                  --head \
                  /path/to/the/huggingface/home/in/this/node
```


在其余工作节点上，运行以下命令：

```plain
bash run_cluster.sh \
                  vllm/vllm-openai \
                  ip_of_head_node \
                  --worker \
                  /path/to/the/huggingface/home/in/this/node
```


然后你会得到一个由容器组成的 Ray 集群。请注意，您需要使运行这些命令的 shell 保持活动状态以维持集群。任何 shell 的断开连接都会终止集群。另外，请注意参数 `ip_of_head_node` 应该是头节点的 IP 地址，所有工作节点都可以访问该 IP 地址。一个常见的误解是使用工作节点的 IP 地址，这是不正确的。


然后，在任意节点上，使用 `docker exec -it node /bin/bash` 进入容器，执行 `ray status` 查看 Ray 集群的状态。您应该看到正确数量的节点和 GPU。


之后，在任何节点上，您都可以照常使用 vLLM，就像所有的 GPU 都在一个节点上一样。常见的做法是将张量并行大小设置为每个节点中的 GPU 数量，将管道并行大小设置为节点数量。例如，如果 2 个节点中有 16 个 GPU（每个节点 8 个 GPU），则可以将张量并行大小设置为 8，将管道并行大小设置为 2：

```plain
vllm serve /path/to/the/model/in/the/container \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2
```


您也可以使用张量并行而不使用管道并行，只需将张量并行大小设置为集群中的 GPU 数量即可。例如，如果 2 个节点中有 16 个 GPU（每个节点 8 个 GPU），则可以将张量并行大小设置为 16：

```plain
vllm serve /path/to/the/model/in/the/container \
    --tensor-parallel-size 16
```


为了使张量并行具有良好的性能，您应该确保节点之间的通信高效，例如使用 Infiniband 等高速网卡。要正确设置集群以使用 Infiniband，请将 `--privileged -e NCCL_IB_HCA=mlx5` 等附加参数附加到 `run_cluster.sh` 脚本中。请联系您的系统管理员以获取有关如何设置标志的更多信息。确认 Infiniband 是否正常工作的一种方法是使用 `NCCL_DEBUG=TRACE` 环境变量集运行 vLLM，例如`NCCL_DEBUG=TRACE vllmserve ...` ，并检查日志以了解 NCCL 版本和使用的网络。如果你在日志中发现 `[send] via NET/Socket` ，则意味着 NCCL 使用原始 TCP Socket，这对于跨节点张量并行来说效率不高。如果您在日志中找到 `[send] via NET/IB/GDRDMA` ，则意味着 NCCL 使用 Infiniband 和 GPU-Direct RDMA，效率很高。


**警告**

在启动 Ray 集群后，最好检查一下节点之间的 GPU-GPU 通信，设置这个并不简单。请参阅 [健全性检查脚本](https://docs.vllm.ai/en/latest/getting_started/debugging.html) 了解更多信息。如果需要为通信配置设置一些环境变量，可以将它们附加到`run_cluster.sh`脚本中，例如`-e NCCL_SOCKET_IFNAME=eth0`。请注意，在 shell 中设置环境变量（例如 `NCCL_SOCKET_IFNAME=eth0 vllmserve ...`）仅适用于同一节点中的进程，不适用于其他节点中的进程，推荐在创建集群时设置环境变量。有关更多信息，请参阅 [讨论](https://github.com/vllm-project/vllm/issues/6803)。


**警告**

请确保你已将模型下载到所有节点（具有相同的路径），或者将模型下载到所有节点均可访问的某个分布式文件系统中。


当您使用 Huggingface repo id 来引用模型时，您应该将您的 Huggingface token 附加到 `run_cluster.sh` 脚本中，如`-e HF_TOKEN=`。推荐的方式是先下载模型，然后使用路径引用模型。


