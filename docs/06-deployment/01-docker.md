---

title: 使用 Docker 进行部署

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


## 使用 vLLM 官方 Docker 镜像

vLLM 提供官方 Docker 镜像进行部署。该镜像可用于运行 OpenAI 兼容服务器，并可在 Docker Hub 上以 [vllm/vllm-openai](https://hub.docker.com/r/vllm/vllm-openai/tags) 形式获取。

```plain
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-v0.1
```
你可以在 image 标签(`vllm/vllm-openai:latest`)后添加其他你需要的[引擎参数](https://docs.vllm.ai/en/latest/serving/engine_args.html#engine-args)。

**注意** 

您可以使用 `ipc=host` 标志或 `--shm-size` 标志来允许容器访问主机的共享内存。 vLLM 使用 PyTorch，它使用共享内存在后台进程之间共享数据，特别是对于张量并行推理。



**注意**

为了避免许可问题（例如 [Issue #8030](https://github.com/vllm-project/vllm/issues/8030)），可选依赖项未包含在基础镜像中。


如果您需要使用这些依赖项（并已接受许可条款），可以在基础镜像之上创建一个自定义 Dockerfile，添加一个额外的层来安装这些依赖项：

```plain
FROM vllm/vllm-openai:v0.7.3


# e.g. install the `audio` and `video` optional dependencies
# NOTE: Make sure the version of vLLM matches the base image!
RUN uv pip install --system vllm[audio,video]==0.7.3
```



**提示**

一些新型号可能仅在 [HF Transformers](https://github.com/huggingface/transformers) 的主分支上可用。


要使用 `transformer` 的开发版本，请在基础映像上创建自定义 Dockerfile，其中包含一个额外的层，用于从源代码安装其代码：

```plain
FROM vllm/vllm-openai:latest


RUN uv pip install --system git+https://github.com/huggingface/transformers.git
```


## 从源代码构建 vLLM Docker 镜像

您可以通过提供的 [Dockerfile](https://github.com/vllm-project/vllm/blob/main/Dockerfile) 从源代码构建并运行 vLLM。构建 vLLM：

```plain
# optionally specifies: --build-arg max_jobs=8 --build-arg nvcc_threads=2
# 可选配置：  --build-arg max_jobs=8 --build-arg nvcc_threads=2
DOCKER_BUILDKIT=1 docker build . --target vllm-openai --tag vllm/vllm-openai
```


>**注意**
>默认情况下，为实现最广泛分发，vLLM 将为所有 GPU 类型进行构建。如果您只是针对机器运行的当前 GPU 类型进行构建，则可以为 vLLM 添加参数 `--build-arg torch_cuda_arch_list= ""` 来查找当前 GPU 类型并为其构建。
>
>如果你正使用 Podman 代替 Docker，在运行 `podman build` 命令时，你可能需要通过添加 `--security-opt label=disable` 禁用 SELinux 标签，避免某些[已知问题](https://github.com/containers/buildah/discussions/4184)。

## 

## 为 Arm64/aarch64 构建

可以为 aarch64 系统（例如 Nvidia Grace-Hopper）构建 Docker 容器。截至目前，这需要使用 PyTorch Nightly 版本，并应被视为**实验性**功能。使用 `--platform "linux/arm64"` 标志将尝试为 arm64 构建。


>**注意**
>多个模块需要编译，因此该过程可能需要较长时间。建议使用 `--build-arg max_jobs=` 和 `--build-arg nvcc_threads=` 标志来加速构建过程。但请确保 `max_jobs` 远大于 `nvcc_threads`，以获得最大收益。同时，在进行并行作业时请注意内存使用情况，因为其占用可能较大（参阅下例）。

```plain
# Example of building on Nvidia GH200 server. (Memory usage: ~15GB, Build time: ~1475s / ~25 min, Image size: 6.93GB)
# 在 Nvidia GH200 服务器上的构建示例。(内存使用: ~15GB, 构建时间: ~1475s / ~25 min, 镜像大小: 6.93GB)
python3 use_existing_torch.py
DOCKER_BUILDKIT=1 docker build . \
  --target vllm-openai \
  --platform "linux/arm64" \
  -t vllm/vllm-gh200-openai:latest \
  --build-arg max_jobs=66 \
  --build-arg nvcc_threads=2 \
  --build-arg torch_cuda_arch_list="9.0+PTX" \
  --build-arg vllm_fa_cmake_gpu_arches="90-real"
```


## 使用自定义构建的 Docker 镜像

使用自定义构建的 Docker 镜像运行 vLLM：

```plain
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    vllm/vllm-openai <args...>
```
参数 `vllm/vllm-openai` 指定要运行的映像，应替换为自定义构建映像的名称（build 命令中的 `-t` 标记）。

>**注意**
>**仅适用于*****v0.4.1*****和*****v0.4.2*** - 这些版本下的 vLLM docker 镜像应该在 root 用户下运行，因为库位于 root 用户的主目录下，即 `/ root/.config/vllm/nccl/cu12/libnccl.so.2.18.1` 需要在运行时加载。如果您在不同用户下运行容器，则可能需要首先更改库 （以及所有父目录） 的权限以允许用户访问它，然后运行带有环境变量的 vLLM。`VLLM_NCCL_SO_PATH=/root/.config/vllm/nccl/cu12/libnccl.so.2.18.1`。

