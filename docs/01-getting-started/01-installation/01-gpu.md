---
title: GPU
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM 是一个支持如下 GPU 类型的 Python 库，根据您的 GPU 型号查看相应的说明。

## NVIDIA CUDA

vLLM 包含预编译的 C++ 和 CUDA (12.8/12.6/11.8) 二进制库。

## AMD ROCm

vLLM 支持拥有 ROCm 6.3 的 AMD GPU。

> **注意：**
> 该设备没有预构建的轮子，所以您必须使用预构建的 Docker 镜像或者从源代码构建 vLLM。

## Inter XPU

vLLM 最初在 Intel GPU 平台上支持基础的模型推理和服务。

> **注意：**
> 该设备没有预构建的轮子或映像，所以您必须从源代码构建 vLLM。

# 需求

- 系统：Linux
- Python: 3.9 – 3.12

## NVIDIA CUDA

- GPU：算力 7.0 及以上（如 V100、T4、RTX20xx、A100、L4、H100 等）

## AMD ROCm

- GPU: MI200s (gfx90a)、MI300 (gfx942)、Radeon RX 7900 系列 (gfx1100)
- ROCm 6.3

## Inter XPU

- 支持硬件：Intel Data Center GPU、Intel ARC GPU
- OneAPI 要求: oneAPI 2024.2

# 使用 Python 安装

## 创建 1 个新 Python 环境

您可以使用 conda 创建 1 个新的 Python 环境：

```bash
# (推荐) 创建一个新的 conda 环境
conda create -n vllm python=3.12 -y
conda activate vllm
```

> **注意：**
> [PyTorch 已](https://github.com/pytorch/pytorch/issues/138506)[弃用](https://github.com/pytorch/pytorch/issues/138506)[该 conda ](https://github.com/pytorch/pytorch/issues/138506)[发布](https://github.com/pytorch/pytorch/issues/138506)[频道](https://github.com/pytorch/pytorch/issues/138506)。如果您使用 conda，请仅使用它创建 Python 环境，而不要用它安装包。

或者您可以使用 [uv](https://docs.astral.sh/uv/) 创建 Python 环境，uv 是一个非常快速的 Python 环境管理器。请依照[该文档](https://docs.astral.sh/uv/#getting-started)安装 uv。安装 uv 以后，你可以使用以下命令创建新的 Python 环境：

```bash
# (推荐) 创建一个新的 uv 环境。使用 `--seed` 在环境中安装 `pip` 和 `setuptools`。
uv venv vllm --python 3.12 --seed
source vllm/bin/activate
```

### NVIDIA CUDA

> **注意：**
> 通过 conda 安装的 PyTorch 会静态链接 NCCL 库，这会导致当 vLLM 尝试使用 NCCL 时出错。详情可见 [Issue #8420](https://github.com/vllm-project/vllm/issues/8420#)。

为了实现高性能，vLLM 需要编译多个 cuda 内核。然而，这一编译过程会导致与其他 CUDA 版本和 PyTorch 版本的二进制不兼容问题。即便是在相同版本的 PyTorch 中，不同的构建配置也可能引发此类不兼容性。

因此，建议使用**全新的** conda 环境安装 vLLM。如果您有不同的 CUDA 版本，或者想要使用现有的 PyTorch 安装，则需要从源代码构建 vLLM。更多说明请参阅下文。

### AMD ROCm

对于该设备没有关于创建新 Python 环境的额外信息。

### Inter XPU

对于该设备没有关于创建新 Python 环境的额外信息。

## 预构建安装包

### NVIDIA CUDA

您可以使用 pip 或 uv pip 安装 vLLM。

```plain
# 安装使用 CUDA 12.1 的 vLLM
pip install vllm  # 如果使用 pip
uv pip install vllm # 如果使用 uv
```

目前，vLLM 的二进制文件默认使用 CUDA 12.1 和公开发布的 PyTorch 版本进行编译。我们还提供使用 CUDA 12.1、11.8 和公开发布的 PyTorch 版本编译的 vLLM 二进制文件：

```plain
# 安装使用 CUDA 11.8 的vLLM
export VLLM_VERSION=0.6.1.post1
export PYTHON_VERSION=310
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

#### 安装最新代码

LLM 推理是一个快速发展的领域，最新代码可能包含错误修复、性能改进和尚未发布的新功能。为了让用户在下一个版本发布前就能体验到最新的代码，vLLM 为运行在 x86 架构上的 Linux 系统提供了 CUDA 12 的预编译包（wheel），这些预编译包覆盖了从 v0.5.3 版本开始的每一次代码提交。

#### 使用 pip 安装最新代码

```plain
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

在使用 pip 安装时，必须加上 --pre 参数，这样 pip 才会考虑安装预发布版本。

如果您想获取先前提交的安装包（例如，用于分析行为变化或性能回退），由于 pip 的限制，您需要通过在 URL 中嵌入提交哈希值来指定轮子文件的完整 URL：

```plain
export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd # use full commit hash from the main branch
pip install https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```

请注意，这些安装包是使用 Python 3.8 ABI 构建的（有关 ABI 的更多详情，请参阅 [PEP 425](https://peps.python.org/pep-0425/)），因此它们兼容 Python 3.8 及更高版本。安装包文件名中的版本号 (1.0.0.dev) 只是一个占位符，用于提供统一的 URL，实际的安装包版本信息包含在安装包的元数据中（在额外索引 URL 中列出的安装包具有正确的版本号）。尽管我们不再支持 Python 3.8（因为 PyTorch 2.5 已经停止对 Python 3.8 的支持），但这些安装包仍然使用 Python 3.8 ABI 构建，以保持与之前相同的安装包名称。

#### 使用 uv 安装最新代码

另一种安装最新代码的方法是使用 uv：

```plain
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
```

如果您想获取先前提交的 wheels 安装包（例如，用于分析行为变化或性能回退），可以在 URL 中指定提交哈希值：

```plain
export VLLM_COMMIT=72d9c316d3f6ede485146fe5aabd4e61dbc59069 # use full commit hash from the main branch
uv pip install vllm --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}
```

uv 方法适用于 vLLM v0.6.6 及更高版本，并提供了一条易于记忆的命令。uv 的一个独特特性是，来自 --extra-index-url 的软件包[优先级高于默认索引](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)中的软件包。
例如，如果最新的公开发布版本是 v0.6.6.post1，uv 允许通过指定 --extra-index-url 安装 v0.6.6.post1 之前的某个提交版本。

相比之下，pip 会合并 --extra-index-url 和默认索引中的软件包，并仅选择最新版本，这使得安装早于已发布版本的开发版本变得困难。

### AMD ROCm

目前没有预构建的 ROCm wheels 安装包。

### Inter XPU

目前没有预构建的 XPU wheels 安装包。

### 从源码构建安装包 (wheel)

### NVIDIA CUDA

#### 使用仅限 Python 的构建方式（无需编译）

如果您只需要修改 Python 代码，则可以在不进行编译的情况下构建并安装 vLLM。使用 `pip` 的 `--editable`[ 标志](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs)，您对代码的更改将在运行 vLLM 时生效：

```plain
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

该命令将执行以下操作：

1. 在您的 vLLM 克隆中查找当前分支。

2. 确定主分支中对应的基础提交。

3. 下载该基础提交的预构建 wheel 文件。

4. 在安装过程中使用其已编译的库文件。

> **注意**
> 
> 如果您修改了 C++ 或内核代码，则无法使用仅限 Python 的构建方式，否则会出现「找不到库 (library not found)」或「未定义符号 (undefined symbol)」的导入错误。
> 如果您对开发分支进行了 rebase，建议卸载 vLLM 并重新运行上述命令，以确保您的库文件是最新的。

如果运行上述命令时出现「找不到 wheel (the wheel not found)」错误，可能是因为您基于的主分支提交刚刚合并，而 wheel 文件仍在构建中。在这种情况下，您可以等待大约一小时后再尝试，或者手动指定先前的提交哈希值，并使用 `VLLM_PRECOMPILED_WHEEL_LOCATION` 环境变量进行安装。

```plain
export VLLM_COMMIT=72d9c316d3f6ede485146fe5aabd4e61dbc59069 # 使用主分支中的完整提交哈希值
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
pip install --editable .
```

您可以在[安装最新代码](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html?device=cuda#install-the-latest-code)中找到更多关于 vLLM wheel 文件的信息。

> **注意**
> 您的源代码提交 ID 可能与最新的 vLLM wheel 文件不同，这可能会导致未知错误。建议您使用与安装的 vLLM wheel 相同的提交 ID 进行构建。请参考「安装最新代码」部分了解如何安装指定的 wheel 文件。

---

#### 完整构建（包含编译）

如果您需要修改 C++ 或 CUDA 代码，则需要从源代码构建 vLLM。此过程可能需要几分钟时间：

```plain
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

> **提示**
>
> 从源代码构建需要大量编译过程。如果您需要多次从源代码构建，建议缓存编译结果，以提高效率。
>
> 例如，您可以使用 `conda install ccache` 或 `apt install ccache` 安装 [ccache](https://github.com/ccache/ccache)。只要 `which ccache` 命令可以找到 `ccache` 可执行文件，构建系统就会自动使用它。首次构建完成后，后续构建将会快得多。
>
> [sccache](https://github.com/mozilla/sccache) 的工作方式与 `ccache` 类似，但可以在远程存储环境中进行缓存。您可以设置以下环境变量来配置 vLLM 的 `sccache` 远程缓存:`SCCACHE_BUCKET=vllm-build-sccache SCCACHE_REGION=us-west-2 SCCACHE_S3_NO_CREDENTIALS=1`。我们还建议设置 `SCCACHE_IDLE_TIMEOUT=0`。

#### 使用现有 PyTorch 安装

某些情况下，PyTorch 依赖无法通过 pip 安装，例如：

- 使用 PyTorch nightly 版本或自定义 PyTorch 构建版本来编译 vLLM
- 在 aarch64 架构且支持 CUDA（GH200）的环境下编译 vLLM（PyPI 未提供对应 PyTorch 预编译包）。目前仅 PyTorch nightly 版本提供 aarch64 架构的 CUDA 预编译包。可通过运行  `pip3 install --pre torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/nightly/cu124](https://download.pytorch.org/whl/nightly/cu124)`  安装 PyTorch nightly 版本，然后在其基础上编译 vLLM

使用现有 PyTorch 安装编译 vLLM 的步骤：

```plain
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
pip install -r requirements-build.txt
pip install -e . --no-build-isolation
```

#### 使用本地 cutlass 编译

当前 vLLM 默认从 GitHub 获取 cutlass 代码进行编译。若需使用本地 cutlass 版本，可通过设置环境变量  `VLLM_CUTLASS_SRC_DIR`  指定本地 cutlass 目录：

```plain
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_CUTLASS_SRC_DIR=/path/to/cutlass pip install -e .
```

#### 故障排除

为避免系统负载过高，可通过  `MAX_JOBS`  环境变量限制并行编译任务数。例如：

```plain
export MAX_JOBS=6
pip install -e .
```

该设置对低性能机器尤为重要。例如在 WSL 环境下（[默认仅分配 50% 总内存](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#main-wsl-settings)），使用  `export MAX_JOBS=1`  可避免因并行编译导致内存不足。副作用是编译时间显著延长。

此外，如果您在构建 vLLM 时遇到问题，建议使用 NVIDIA PyTorch Docker 镜像解决编译问题：

```plain
# 使用 --ipc=host 确保共享内存容量充足
docker run --gpus all -it --rm --ipc=host nvcr.io/nvidia/pytorch:23.10-py3
```

如果您不想使用 Docker，建议完整安装 CUDA Toolkit。您可以从[官方网站](https://developer.nvidia.com/cuda-toolkit-archive)下载并安装。安装后，需设置 `CUDA_HOME` 环境变量为 CUDA Toolkit 的安装路径，并确保 `nvcc` 编译器在 `PATH` 中，例如：

```plain
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
```

可以使用以下命令检查 CUDA Toolkit 是否正确安装：

```plain
nvcc --version # verify that nvcc is in your PATH # 检查 nvcc 是否在 PATH 中
${CUDA_HOME}/bin/nvcc --version # verify that nvcc is in your CUDA_HOME
```

#### 不支持的操作系统构建

vLLM 仅能在 Linux 系统上完整运行，但对于开发目的，您仍然可以在其他操作系统（例如 macOS）上构建 vLLM，使其能够被导入，从而提供更便捷的开发环境。但请注意，vLLM 在非 Linux 系统上不会编译二进制文件，因此无法运行推理任务。

在安装前，可以禁用 `VLLM_TARGET_DEVICE` 变量，如下所示：

```plain
export VLLM_TARGET_DEVICE=empty
pip install -e .
```

### AMD ROCm

1. 安装依赖（如果您已经处于一个已安装以下内容的环境中或 Docker 容器中，则可以跳过此步骤）:

- [ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html)
- [PyTorch](https://pytorch.org/)

对于安装 PyTorch，您可以从 1 个新的 docker 镜像开始，例如 `rocm/pytorch:rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0`、`rocm/pytorch-nightly`。如果您正在使用 docker 镜像，跳到步骤 3。

或者，您可以使用 PyTorch wheels 安装 PyTorch，在 PyTorch [入门指南](https://pytorch.org/get-started/locally/) 中的 PyTorch 安装指南可以查看相关信息。

```plain
pip uninstall torch -y
pip install --no-cache-dir --pre torch --index-url https://download.pytorch.org/whl/rocm6.3
```

2. 安装 [Triton flash attention for ROCm](https://github.com/ROCm/triton)

按照 [ROCm/triton](https://github.com/ROCm/triton/blob/triton-mlir/README.md) 的说明安装 ROCm 的 Triton flash attention（默认 triton-mlir 分支）

```plain
python3 -m pip install ninja cmake wheel pybind11
pip uninstall -y triton
git clone https://github.com/OpenAI/triton.git
cd triton
git checkout e5be006
cd python
pip3 install .
cd ../..
```

> **注意：**
> 如果在构建 triton 期间您遇见了有关下载包的 HTTP 问题，请再次尝试，因为 HTTP 问题是间歇性的。

3. 或者，如果您选择使用 CK flash Attention，您可以安装 [flash Attention for ROCm](https://github.com/ROCm/flash-attention/tree/ck_tile)。

按照 [ROCm/flash-attention](https://github.com/ROCm/flash-attention/tree/ck_tile#amd-gpurocm-support) 的说明安装 ROCm's Flash Attention (v2.7.2)。或者也可以在发布页面中找到专为 vLLM 使用准备的轮子 (wheel)。

例如，对于 ROCm 6.3，假定您的 gfx 架构是 `gfx90a`，想获取您的 gfx 架构，请运行 `rocminfo |grep gfx`。

```plain
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention
git checkout b7d29fb
git submodule update --init
GPU_ARCHS="gfx90a" python3 setup.py install
cd ..
```

> **注意：**
> 您可能需要将 "ninja" 版本降级到 1.10，编译 flash-attention-2 时不会使用它（例如 `pip install ninja==1.10.2.4`）

4. 构建 vLLM。例如，基于 ROCm 6.3 的 vLLM 可以通过以下步骤构建：

```plain
pip install --upgrade pip


# 构建并安装 AMD SMI
pip install /opt/rocm/share/amd_smi


# 安装依赖
pip install --upgrade numba scipy huggingface-hub[cli,hf_transfer] setuptools_scm
pip install "numpy<2"
pip install -r requirements-rocm.txt


# 为 MI210/MI250/MI300 构建 vLLM
export PYTORCH_ROCM_ARCH="gfx90a;gfx942"
python3 setup.py develop
```

该步可能花费 5-10 分钟。目前 `pip install` 不适用于 ROCm 的安装。

> **提示**
>
> - 默认使用 Triton flash attention。为了进行基准测试，建议在收集性能数据之前先运行一个预热步骤。
> - Triton flash attention 目前不支持滑动窗口注意力 (sliding window attention)。如果使用半精度 (half precision)，请使用 CK flash-attention 以支持滑动窗口。
>
> 如果要使用 CK flash-attention 或 PyTorch naive attention，请使用以下命令关闭 Triton flash attention: `export VLLM_USE_TRITON_FLASH_ATTN=0`。
>
> 理想情况下，PyTorch 的 ROCm 版本最好与 ROCm 驱动程序版本匹配。

> **提示**
> 对于 MI300x (gfx942) 用户，为了获得最佳性能，请参考 [MI300x 调优指南](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html)，以获取系统和工作流级别的性能优化和调优建议。对于 vLLM，请参考 [vLLM 性能优化指南](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html#vllm-performance-optimization)。

### Inter XPU

- 首先，安装所需的驱动程序和 intel OneAPI 2024.2 (或更高版本)。
- 其次，安装用于 vLLM XPU 后端构建的 Python 包：

```plain
source /opt/intel/oneapi/setvars.sh
pip install --upgrade pip
pip install -v -r requirements-xpu.txt
```

- 最后，构建并安装 vLLM XPU 后端：

```plain
VLLM_TARGET_DEVICE=xpu python setup.py install
```

> **注意：**
> FP16 是当前 XPU 后端的默认数据类型。BF16 数据类型在英特尔数据中心 GPU 上受支持，但在英特尔 Arc GPU 上尚不支持。

# 使用 Docker 进行设置

## 预构建镜像

### NVIDIA CUDA

查阅[使用 ](https://docs.vllm.ai/en/latest/deployment/docker.html#deployment-docker-pre-built-image)[v](https://docs.vllm.ai/en/latest/deployment/docker.html#deployment-docker-pre-built-image)[LLM 官方 Docker 镜像](https://docs.vllm.ai/en/latest/deployment/docker.html#deployment-docker-pre-built-image)获得使用官方 Docker 镜像的教程。

另一种获取最新代码的方法是使用 Docker 镜像：

```plain
export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd # use full commit hash from the main branch 使用主分支上的完整提交哈希哈值。
docker pull public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:${VLLM_COMMIT}
```

这些 docker 镜像仅用于 CI 和测试，不作为生产使用，将会在数日后过期。
最新代码可能包含 bug 且不稳定，请谨慎使用。

### AMD ROCm

[AMD Infinity hub for vLLM](https://hub.docker.com/r/rocm/vllm/tags) 提供了预构建、优化过的 docker 镜像，旨在验证 AMD Instinct™ MI300X 加速器上的推理性能。

> **提示：**
> 请检查在 [AMD Instinct MI300X 上](https://rocm.docs.amd.com/en/latest/how-to/performance-validation/mi300x/vllm-benchmark.html)[对](https://rocm.docs.amd.com/en/latest/how-to/performance-validation/mi300x/vllm-benchmark.html)[ LLM 推理性能验证](https://rocm.docs.amd.com/en/latest/how-to/performance-validation/mi300x/vllm-benchmark.html)，并查看如何使用该预构建 docker 镜像的说明。

### Inter XPU

目前没有预构建的 XPU 镜像。

## 从源代码构建镜像

### NVIDIA CUDA

有关构建 Docker 映像的说明，请参阅[从源代码构建 vLLM Docker 镜像](https://docs.vllm.ai/en/latest/deployment/docker.html#deployment-docker-build-image-from-source)。

### AMD ROCm

将 vLLM 与 ROCm 结合使用的推荐方式是从源码构建 Docker 镜像。

#### （可选）构建包含 ROCm 软件栈的镜像

vLLM 需要从 [Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/Dockerfile.rocm_base) 构建一个包含 ROCm 软件栈的 Docker 镜像。**此步骤是可选的，因为\*\***此\***\*rocm_base 镜像为了加快用户体验，通常已预构建并存储在**[Docker Hub](https://hub.docker.com/r/rocm/vllm-dev)**上，标签为**`rocm/vllm-dev:base`**。**如果您选择自己构建此 rocm_base 镜像，步骤如下。

注意，用户需要使用 BuildKit 启动 Docker 构建。用户可以通过在调用 `docker build` 命令时设置环境变量 `DOCKER_BUILDKIT=1`，或者在 Docker 守护进程配置文件 /etc/docker/daemon.jso` 中按照以下方式启用 BuildKit 并重启守护进程：

```json
{
  "features": {
    "buildkit": true
  }
}
```

如需为 MI200 和 MI300 系列构建基于 ROCm 6.3 的 vLLM 镜像，可以使用默认命令：

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.rocm_base -t rocm/vllm-dev:base .
```

#### 构建包含 vLLM 的镜像

首先，从 [Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/Dockerfile.rocm) 构建 Docker 镜像，并从该镜像启动容器。注意，用户需要使用 BuildKit 启动 Docker 构建。可以通过在调用 `docker build` 命令时设置环境变量 `DOCKER_BUILDKIT=1`，或者在 Docker 守护进程配置文件 /etc/docker/daemon.jso` 中按照以下方式启用 BuildKit 并重启守护进程：

```json
{
  "features": {
    "buildkit": true
  }
}
```

[Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/Dockerfile.rocm) 默认使用 ROCm 6.3，但也支持 ROCm 5.7、6.0、6.1 和 6.2（在较旧的 vLLM 分支中）。它提供了以下构建参数以灵活构建自定义 Docker 镜像：

- `BASE_IMAGE`：指定构建 Docker 镜像时使用的基础镜像。默认值 `rocm/vllm-dev:base` 是由 AMD 发布和维护的镜像，它是使用 `Dockerfile.rocm_base` 构建的。
- `USE_CYTHON`：在 Docker 构建时对部分 Python 文件子集运行 Cython 编译的选项。
- `BUILD_RPD`：在镜像中包含 RocmProfileData 性能分析工具。
- `ARG_PYTORCH_ROCM_ARCH`：允许覆盖基础镜像中的 gfx 架构值。

这些参数可以通过 `--build-arg` 选项传递给 `docker build` 命令。

如需为 MI200 和 MI300 系列构建基于 ROCm 6.3 的 vllm 镜像，可以使用默认命令：

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.rocm -t vllm-rocm .
```

如需为 Radeon RX7900 系列 (gfx1100) 构建基于 ROCm 6.3 的 vllm 镜像，应选择替代的基础镜像：

```bash
DOCKER_BUILDKIT=1 docker build --build-arg BASE_IMAGE="rocm/vllm-dev:navi_base" -f Dockerfile.rocm -t vllm-rocm .
```

请使用以下命令运行上述 Docker 镜像 `vllm-rocm`：

```bash
docker run -it \
   --network=host \
   --group-add=video \
   --ipc=host \
   --device /dev/kfd \
   --device /dev/dri \
   -v <path/to/model>:/app/model \
   vllm-rocm \
   bash
```

其中 `<path/to/model>` 是模型存储的路径，例如 llama2 或 llama3 模型的权重文件路径。

### Inter XPU

```plain
docker build -f Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .
docker run -it \
             --rm \
             --network=host \
             --device /dev/dri \
             -v /dev/dri/by-path:/dev/dri/by-path \
             vllm-xpu-env
```

# 支持的功能

## NVIDIA CUDA

在 [Feature x Hardware](https://docs.vllm.ai/en/latest/features/compatibility_matrix.html#feature-x-hardware) 兼容矩阵中可查看功能支持信息。

## AMD ROCm

在 [Feature x Hardware](https://docs.vllm.ai/en/latest/features/compatibility_matrix.html#feature-x-hardware) 兼容矩阵中可查看功能支持信息。

## Inter XPU

XPU 平台支持张量并行推理/服务，并且还支持在线服务中的流水线并行（作为测试版功能）。我们要求使用 Ray 作为分布式运行时后端。例如，参考的执行命令如下：

```plain
python -m vllm.entrypoints.openai.api_server \
     --model=facebook/opt-13b \
     --dtype=bfloat16 \
     --device=xpu \
     --max_model_len=1024 \
     --distributed-executor-backend=ray \
     --pipeline-parallel-size=2 \
     -tp=8
```

默认情况下，如果系统中未检测到现有的 Ray 实例，则会自动启动一个 Ray 实例，其中 `num-gpus` 等于 `parallel_config.world_size`。我们建议在执行前正确启动一个 Ray 集群，可以参考 [examples/online_serving/run_cluster.sh](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/run_cluster.sh) 辅助脚本。
