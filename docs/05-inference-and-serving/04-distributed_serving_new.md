---
title: 分布式推理与服务
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

## 如何决定分布式推理策略？

在深入探讨分布式推理和服务的细节之前，我们首先需要明确何时使用分布式推理以及有哪些可用的策略。常见的做法是：

- **单** **GPU（无分布式推理）**：如果你的模型可以放入单个 GPU 中，你可能不需要使用分布式推理。只需使用单个 GPU 运行推理即可。
- **单节点多GPU（张量并行推理）**：如果你的模型太大，无法放入单个 GPU，但可以放入单个节点中的多个 GPU 中，你可以使用张量并行。张量并行的大小是你想要使用的 GPU 数量。例如，如果你的单个节点中有4个 GPU，你可以将张量并行大小设置为 4。
- **多节点多** **GPU（张量并行加流水线并行推理）**：如果你的模型太大，无法放入单个节点中，你可以同时使用张量并行和流水线并行。张量并行大小是每个节点中你想要使用的 GPU 数量，流水线并行大小是你想要使用的节点数量。例如，如果你有 2 个节点中的 16 个 GPU（每个节点 8 个GPU），你可以将张量并行大小设置为 8，流水线并行大小设置为 2。

简而言之，您应该增加 GPU 和节点的数量，直到有足够的 GPU 内存来容纳模型。张量并行大小应该是每个节点中的 GPU 数量，流水线并行大小应该是节点数量。

在添加足够的GPU和节点以容纳模型后，你可以先运行 vLLM，它会打印一些日志，如 `# GPU blocks: 790`。将该数字乘以 `16` （块大小），你可以大致得到当前配置下可以服务的最大 token 数量。如果这个数字不令人满意，例如你想要更高的吞吐量，你可以进一步增加 GPU 或节点的数量，直到块数量足够。

> **注意**
> 有一种边缘情况：如果模型可以放入单个节点中的多个 GPU 中，但 GPU 数量不能均匀地分割模型大小，你可以使用流水线并行，它沿着层分割模型并支持不均匀的分割。在这种情况下，张量并行大小应为 1，流水线并行大小应为 GPU 数量。

## 在单个节点上运行 vLLM

vLLM 支持分布式张量并行和流水线并行推理与服务。目前，我们支持 [Megatron-LM 的张量并行算法](https://arxiv.org/pdf/1909.08053.pdf)。我们使用 [Ray](https://github.com/ray-project/ray) 或 Python 原生多进程管理分布式运行时。当在单个节点上部署时，可以使用多进程，多节点推理目前需要 Ray。

当不在 Ray 放置组中运行并且在同一节点上有足够的 GPU 用于配置的 `tensor_parallel_size` 时，默认情况下将使用多进程，否则将使用 Ray。可以通过 `LLM` 类的 `distributed_executor_backend` 参数或 `--distributed-executor-backend` API 服务器参数覆盖此默认值。将其设置为 `mp` 以使用多进程，或设置为 `ray` 以使用 Ray。对于多进程情况，不需要安装 Ray。

要使用 `LLM` 类运行多 GPU 推理，请将 `tensor_parallel_size` 参数设置为你想要使用的 GPU 数量。例如，要在 4 个 GPU 上运行推理：

```plain
from vllm import LLM
llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
output = llm.generate("San Francisco is a")
```

要运行多 GPU 服务，请在启动服务器时传入 `--tensor-parallel-size` 参数。例如，要在 4 个 GPU 上运行 API 服务器：

```go
 vllm serve facebook/opt-13b \
     --tensor-parallel-size 4
```

你还可以额外指定 `--pipeline-parallel-size` 以启用流水线并行。例如，要在 8 个 GPU 上运行 API 服务器，同时使用流水线并行和张量并行：

```go
 vllm serve gpt2 \
     --tensor-parallel-size 4 \
     --pipeline-parallel-size 2
```

## 在多个节点上运行 vLLM

如果单个节点没有足够的 GPU 来容纳模型，你可以使用多个节点运行模型。重要的是确保所有节点上的执行环境相同，包括模型路径、Python 环境。推荐的方法是使用 docker 镜像来确保相同的环境，并通过将它们映射到相同的 docker 配置中来隐藏主机机器的异构性。

第一步是启动容器并将它们组织成一个集群。我们提供了帮助脚本 [examples/online_serving/run_cluster.sh](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/run_cluster.sh) 来启动集群。请注意，此脚本启动的 docker 没有访问 GPU 性能计数器所需的特权，这些计数器在运行分析和跟踪工具时需要。为此，脚本可以通过在 docker run 命令中使用 `--cap-add` 选项将 `CAP_SYS_ADMIN` 添加到 docker 容器中。

选择一个节点作为头节点，并运行以下命令：

```go
bash run_cluster.sh \
                vllm/vllm-openai \
                ip_of_head_node \
                --head \
                /path/to/the/huggingface/home/in/this/node \
                -e VLLM_HOST_IP=ip_of_this_node
```

在其他工作节点上，运行以下命令：

```go
bash run_cluster.sh \
                vllm/vllm-openai \
                ip_of_head_node \
                --worker \
                /path/to/the/huggingface/home/in/this/node \
                -e VLLM_HOST_IP=ip_of_this_node
```

然后您将获得一个**容器**的 ray 集群。请注意，您需要保持运行这些命令的 shell 存活以维持集群。任何 shell 断开连接都会终止集群。此外，请注意参数 `ip_of_head_node` 应该是头节点的 IP 地址，所有工作节点都可以访问。每个工作节点的 IP 地址应在 `VLLM_HOST_IP` 环境变量中指定，并且每个工作节点应不同。请检查集群的网络配置，确保节点可以通过指定的 IP 地址相互通信。

> **警告** >**由于这是一个\*\***容器的 ray 集群，所有以下命令都应在容器中执行，\*\*否则你是在主机机器上执行命令，而主机机器并未连接到 ray 集群。要进入容器，你可以使用 `docker exec -it node /bin/bash`。

然后，在任意节点上，使用 `docker exec -it node /bin/bash` 进入容器，执行 `ray status` 和 `ray list nodes` 以检查 Ray 集群的状态。你应该看到正确的节点和 GPU 数量。

之后，在任何节点上，再次使用 `docker exec -it node /bin/bash` 进入容器。**在容器中\*\***，\*\*你可以像往常一样使用 vLLM，就像你在一台节点上拥有所有 GPU 一样。常见的做法是将张量并行大小设置为每个节点中的 GPU 数量，流水线并行大小设置为节点数量。例如，如果你有 2 个节点中的 16 个 GPU（每个节点 8 个 GPU），你可以将张量并行大小设置为 8，流水线并行大小设置为 2：

```go
 vllm serve /path/to/the/model/in/the/container \
     --tensor-parallel-size 8 \
     --pipeline-parallel-size 2
```

你也可以使用张量并行而不使用流水线并行，只需将张量并行大小设置为集群中的 GPU 数量。例如，如果你有 2 个节点中的 16 个 GPU（每个节点 8 个 GPU），你可以将张量并行大小设置为 16：

```go
vllm serve /path/to/the/model/in/the/container \
     --tensor-parallel-size 16
```

为了使张量并行高效运行，你应该确保节点之间的通信是高效的，例如使用高速网卡如 Infiniband。使用 Infiniband 需要正确设置集群，请将额外的参数附加到 `run_cluster.sh` 脚本中，如 `--privileged -e NCCL_IB_HCA=mlx5`。请联系你的系统管理员以获取有关如何设置这些标志的更多信息。确认 Infiniband 是否正常工作的一种方法是运行 vLLM 时设置 `NCCL_DEBUG=TRACE` 环境变量，例如 `NCCL_DEBUG=TRACE vllm serve ...` 并检查日志中的 NCCL 版本和使用的网络。如果你在日志中看到 `[send] via NET/Socket`，这意味着 NCCL 使用原始 TCP Socket，这对于跨节点张量并行来说并不高效。如果你在日志中看到 `[send] via NET/IB/GDRDMA`，这意味着 NCCL 使用 Infiniband 与 GPU-Direct RDMA，可以高效运行。

> **警告**
> 启动 Ray 集群后，您最好也检查节点之间的 GPU-GPU 通信。设置它可能并不简单。请参考 [健康检查脚本](https://docs.vllm.ai/en/latest/getting_started/troubleshooting.html#troubleshooting-incorrect-hardware-driver) 以获取更多信息。如果你需要为通信配置设置一些环境变量，你可以将它们附加到 `run_cluster.sh` 脚本中，例如 `-e NCCL_SOCKET_IFNAME=eth0`。请注意，在 shell 中设置环境变量（例如 `NCCL_SOCKET_IFNAME=eth0 vllm serve ...`）仅对同一节点中的进程有效，而不对其他节点中的进程有效。创建集群时设置环境变量是推荐的方式。有关更多信息，请参见[问题 #6803](https://github.com/vllm-project/vllm/issues/6803#)。

> **警告**
> 请确保你将模型下载到所有节点（使用相同的路径），或者模型下载到所有节点都可以访问的分布式文件系统中。
> 当你使用 huggingface 仓库 ID 来引用模型时，你应该将你的 huggingface 令牌附加到 `run_cluster.sh` 脚本中，例如 `-e HF_TOKEN=`。推荐的方法是先下载模型，然后使用路径来引用模型。

> **警告**
> 如果你不断收到错误消息 `Error: No available node types can fulfill resource request`，但你在集群中有足够的 GPU，可能是你的节点有多个 IP 地址，vLLM 无法找到正确的 IP 地址，尤其是在使用多节点推理时。请确保 vLLM 和 ray 使用相同的 IP 地址。你可以在 `run_cluster.sh` 脚本中设置 `VLLM_HOST_IP` 环境变量为正确的 IP 地址（每个节点不同！），并检查 `ray status` 和 `ray list nodes` 以查看 Ray 使用的 IP 地址。有关更多信息，请参见[问题 #7815](https://github.com/vllm-project/vllm/issues/7815#)。
