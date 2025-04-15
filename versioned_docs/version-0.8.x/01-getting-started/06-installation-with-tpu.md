---
title: 使用 TPU 安装
---


vLLM 使用 PyTorch XLA 支持 Google Cloud TPU。


## 依赖环境

* Google Cloud TPU VM（单主机和多主机）

* TPU 版本: v5e、v5p、v4

* Python: 3.10


安装选项：


1. [使用`Dockerfile.tpu`构建 Docker 镜像](#使用-dockerfiletpu-构建-docker-镜像)

2. [从源代码构建](#从源代码构建)


## 使用`Dockerfile.tpu` 构建 Docker 镜像

[Dockerfile.tpu](https://github.com/vllm-project/vllm/blob/main/Dockerfile.tpu) 用于构建具有 TPU 支持的 docker 镜像。

```plain
docker build -f Dockerfile.tpu -t vllm-tpu .
```


您可以使用以下命令运行 docker 镜像：

```plain
# Make sure to add `--privileged --net host --shm-size=16G`.

# 确保添加 `--privileged --net host --shm-size=16G`。

docker run --privileged --net host --shm-size=16G -it vllm-tpu
```


## 从源代码构建

您还可以从源代码构建并安装 TPU 后端。


首先，安装依赖：

```plain
# (Recommended) Create a new conda environment.
#（推荐）创建一个新的 conda 环境。

conda create -n myenv python=3.10 -y
conda activate myenv

# Clean up the existing torch and torch-xla packages.
# 清理现有的 torch 和 torch-xla 包。

pip uninstall torch torch-xla -y

# Install PyTorch and PyTorch XLA.
# 安装 PyTorch 和 PyTorch XLA。

export DATE="20240828"
export TORCH_VERSION="2.5.0"
pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-${TORCH_VERSION}.dev${DATE}-cp310-cp310-linux_x86_64.whl
pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-${TORCH_VERSION}.dev${DATE}-cp310-cp310-linux_x86_64.whl

# Install JAX and Pallas.
# 安装 JAX 和 Pallas。

pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

# Install other build dependencies.
# 安装其他构建依赖项。

pip install -r requirements-tpu.txt
```


下一步，从源代码构建 vLLM。这只需要几秒钟：

```plain
VLLM_TARGET_DEVICE="tpu" python setup.py develop
```


**注意**


由于 TPU 依赖于需要静态形状的 XLA，因此 vLLM 会将可能的输入形状进行分桶处理，并为每个不同的形状编译 XLA 图。第一次运行的编译时间可能需要 20~30 分钟。不过由于 XLA 图会缓存在磁盘中（默认在`VLLM_XLA_CACHE_PATH` 或 `~/.cache/vllm/xla_cache` 中），之后的编译时间会减少到大约 5 分钟。


**提示**


如果您遇到以下错误：

```plain
from torch._C import *  # noqa: F403

ImportError: libopenblas.so.0: cannot open shared object file: No such file or directory
```


请使用以下命令安装 OpenBLAS：

```plain
sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev
```


