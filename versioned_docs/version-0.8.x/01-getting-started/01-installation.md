---
title: 安装
---


vLLM 是一个 Python 库，包含预编译的 C++ 和 CUDA (12.1) 二进制文件。


## 依赖环境

* 操作系统：Linux

* Python：3.8 - 3.12

* GPU：计算能力 7.0 或更高（例如 V100、T4、RTX20xx、A100、L4、H100 等）


## 使用 pip 安装

您可以使用 pip 安装 vLLM:

```plain
# (Recommended) Create a new conda environment.
#（推荐）创建一个新的 conda 环境。

conda create -n myenv python=3.10 -y
conda activate myenv

# Install vLLM with CUDA 12.1.
# 安装带有 CUDA 12.1 的 vLLM。

pip install vllm
```


**注意**


截至目前，vLLM 的二进制文件默认使用 CUDA 12.1 和公共 PyTorch 发行版本进行编译。我们还提供使用 CUDA 11.8 和公共 PyTorch 发行版本编译的 vLLM 二进制文件：

```plain
# Install vLLM with CUDA 11.8.
# 安装带有 CUDA 11.8 的 vLLM。

export VLLM_VERSION=0.4.0
export PYTHON_VERSION=310
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```


为了提高性能，vLLM 必须编译多个 cuda 内核。遗憾的是，这种编译会引入其他 CUDA 版本和 PyTorch 版本的二进制不兼容性，即使对于具有不同构建配置的相同 PyTorch 版本也是如此。


因此，建议使用**全新的** conda 环境安装 vLLM。如果您有不同的 CUDA 版本或者想要使用现有的 PyTorch 安装，则需要从源代码构建 vLLM。请参阅以下的说明。


**注意**


自 v0.5.3 版本以来，vLLM 还为每次提交发布一个 wheel 子集（Python 3.10、3.11 和 CUDA 12）。您可以使用以下命令下载它们：

```plain
export VLLM_VERSION=0.5.4 # vLLM's main branch version is currently set to latest released tag

export VLLM_VERSION=0.5.4 # vLLM 的主分支版本当前设置为最新发布的标签

pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-${VLLM_VERSION}-cp38-abi3-manylinux1_x86_64.whl
# You can also access a specific commit

# 你还可以访问特定的提交

# export VLLM_COMMIT=...

# 导出 VLLM_COMMIT=...

# pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/${VLLM_COMMIT}/vllm-${VLLM_VERSION}-cp38-abi3-manylinux1_x86_64.whl
```


## 从源代码构建

您还可以从源代码构建并安装 vLLM：

```plain
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .  # This may take 5-10 minutes.

pip install -e 。  # 这可能需要 5-10 分钟。
```


**注意**


vLLM 只能在 Linux 上完全运行，但您仍然可以在其他系统（例如 macOS）上构建它。此构建仅用于开发目的，允许导入并提供更方便的开发环境。这些二进制文件不会被编译，也无法在非 Linux 系统上运行。您可以使用以下命令创建这样的构建：

```plain
export VLLM_TARGET_DEVICE=empty
pip install -e .
```


**提示**


从源代码进行构建需要大量的编译工作。如果您多次从源代码构建，那么缓存编译结果是很有益处的。例如，您可以通过 *conda install ccache* 或 *apt install ccache* 安装 [ccache](https://github.com/ccache/ccache) 。只要 *which ccache* 命令可以找到 *ccache* 二进制文件，构建系统就会自动使用它。在第一次构建之后，后续的构建将会快很多。


**提示**


为了避免系统过载，您可以通过环境变量 *MAX_JOBS* 限制同时运行的编译任务数量。例如：

```plain
export MAX_JOBS=6
pip install -e .
```


**提示**


如果您在构建 vLLM 时遇到问题，我们建议使用 NVIDIA PyTorch Docker 镜像。

```plain
# Use `--ipc=host` to make sure the shared memory is large enough.

# 使用 `--ipc=host` 确保共享内存足够大。

docker run --gpus all -it --rm --ipc=host nvcr.io/nvidia/pytorch:23.10-py3
```


如果您不想使用 docker，建议完整安装 CUDA 工具包。您可以从[官方网站](https://developer.nvidia.com/cuda-toolkit-archive)下载并安装它。安装完成后，将环境变量 *CUDA_HOME* 设置为 CUDA 工具包的安装路径，并确保 *nvcc* 编译器在您的 *PATH* 中，例如：

```plain
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
```


以下是验证 CUDA 工具包是否已正确安装的完整检查：

```plain
nvcc --version # verify that nvcc is in your PATH

nvcc --version # 验证 nvcc 是否在您的 PATH 中

${CUDA_HOME}/bin/nvcc --version # verify that nvcc is in your CUDA_HOME

${CUDA_HOME}/bin/nvcc --version # 验证 nvcc 是否在您的 CUDA_HOME 中
```
