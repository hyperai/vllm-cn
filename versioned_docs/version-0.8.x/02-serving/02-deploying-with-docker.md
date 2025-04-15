---
title: 使用 Docker 进行部署
---


vLLM 提供了一个官方 Docker 镜像用于部署。该镜像可用于运行与 OpenAI 兼容服务器，并且可在 Docker Hub 上以 [vllm/vllm-openai](https://hub.docker.com/r/vllm/vllm-openai/tags) 的形式获取。

```plain
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-v0.1
```


**注意：**

您可以使用 `ipc=host` 标志或 `--shm-size` 标志来允许容器访问主机的共享内存。 vLLM 使用 PyTorch，而 PyTorch 在底层使用共享内存在进程之间共享数据，特别是在张量并行推理中。


您可以通过提供的 [Dockerfile](https://github.com/vllm-project/vllm/blob/main/Dockerfile) 从源代码构建并运行 vLLM。构建 vLLM：

```plain
DOCKER_BUILDKIT=1 docker build . --target vllm-openai --tag vllm/vllm-openai # optionally specifies: --build-arg max_jobs=8 --build-arg nvcc_threads=2


DOCKER_BUILDKIT=1 docker 构建 . --target vllm-openai --tag vllm/vllm-openai # 可选指定： --build-arg max_jobs=8 --build-arg nvcc_threads=2
```


**注意：** 

默认情况下，为实现最广泛分发，vLLM 将为所有 GPU 类型进行构建。如果您只是针对机器运行的当前 GPU 类型进行构建，则可以为 vLLM 添加参数 `--build-arg torch_cuda_arch_list= ""` 来查找当前 GPU 类型并为其构建。


运行 vLLM：

```plain
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    vllm/vllm-openai <args...>
```


**注意：**

**仅适用于*****v0.4.1*****和*****v0.4.2*** **版本** - 这些版本下的 vLLM docker 镜像应该在 root 用户下运行，因为在运行时需要加载位于 root 用户主目录下的一个库，即 `/ root/.config/vllm/nccl/cu12/libnccl.so.2.18.1` 。如果您在不同用户下运行容器，则可能需要先更改库（以及所有父目录）的权限以允许用户访问它，然后使用环境变量 `VLLM_NCCL_SO_PATH=/root/.config/vllm/nccl/cu12/libnccl.so.2.18.1`。


