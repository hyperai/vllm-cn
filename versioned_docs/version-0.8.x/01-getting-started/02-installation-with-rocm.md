---
title: 使用 ROCm 安装
---


vLLM 支持采用 ROCm 6.1 的 AMD GPU。


## 依赖环境

* 操作系统：Linux

* Python：3.8 -- 3.11

* GPU：MI200s (gfx90a)、MI300 (gfx942)、Radeon RX 7900 系列 (gfx1100)

* ROCm 6.1


安装选项：


1. [使用 docker 从源代码构建](#选项-1-使用-docker-从源代码构建--推荐-)

2. [从源代码构建](#选项-2-从源代码构建)


## 选项 1：使用 docker 从源代码构建 （推荐）

您可以从源代码构建并安装 vLLM。


首先，从 [Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/Dockerfile.rocm) 构建一个 docker 镜像，并从该镜像启动一个 docker 容器。


[Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/Dockerfile.rocm) 默认使用 ROCm 6.1，但在较旧的 vLLM 分支中也支持 ROCm 5.7 和 6.0。方法非常灵活，可以使用以下参数自定义 Docker 镜像的构建：


* *BASE_IMAGE*：指定运行 `docker build` 时使用的基础镜像，特别是 ROCm 基础镜像上的 PyTorch。

* *BUILD_FA*：指定是否构建 CK flash-attention。默认值为 1。对于 [Radeon RX 7900 系列 (gfx1100)](https://rocm.docs.amd.com/projects/radeon/en/latest/index.html)，在 flash-attention 支持该目标前应将其设置为 0。

* *FX_GFX_ARCHS*：指定用于构建 CK flash-attention 的 GFX 架构，例如 MI200 和 MI300 的 *gfx90a;gfx942*。默认为 *gfx90a;gfx942**。*

* *FA_BRANCH*：指定用于在 [ROCm's flash-attention repo](https://github.com/ROCmSoftwarePlatform/flash-attention) 中构建 CK flash-attention 的分支。默认为 *ae7928c**。*

* *BUILD_TRITON*: 指定是否构建 triton flash-attention。默认值为 1。


这些值可以在使用 `--build-arg` 选项运行 `docker build` 时传入。


要在 ROCm 6.1 上为 MI200 和 MI300 系列构建 vllm，您可以使用默认值：

```plain
DOCKER_BUILDKIT=1 docker build -f Dockerfile.rocm -t vllm-rocm .
```


要在 ROCm 6.1 上为 Radeon RX7900 系列 (gfx1100) 构建 vllm，您应该指定 `BUILD_FA` ，如下所示：

```plain
DOCKER_BUILDKIT=1 docker build --build-arg BUILD_FA="0" -f Dockerfile.rocm -t vllm-rocm .
```


要运行上面的 docker 镜像 `vllm-rocm`，请使用以下命令：

```plain
docker run -it \
   --network=host \
   --group-add=video \
   --ipc=host \
   --cap-add=SYS_PTRACE \
   --security-opt seccomp=unconfined \
   --device /dev/kfd \
   --device /dev/dri \
   -v <path/to/model>:/app/model \
   vllm-rocm \
   bash
```


其中 `<path/to/model>` 是存储模型的位置，例如 llama2 或 llama3 模型的权重。


## 选项 2：从源代码构建

1. 安装依赖（如果您已经在安装了以下内容的环境或者 docker 中，则可以跳过）：


* [ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html)

* [PyTorch](https://pytorch.org/)

* [hipBLAS](https://rocm.docs.amd.com/projects/hipBLAS/en/latest/install.html)


对于安装 PyTorch，您可以从一个新的 docker 镜像开始，例如 *rocm/pytorch:rocm6.1.2_ubuntu20.04_py3.9_pytorch_staging*、*rocm/pytorch-nightly*。


或者，你可以使用 PyTorch wheels 安装 PyTorch。你可以查看 PyTorch [入门指南](https://pytorch.org/get-started/locally/)中的 PyTorch 安装指南。


1. 安装 [Triton flash attention for ROCm](https://github.com/ROCm/triton)


按照 [ROCm/triton](https://github.com/ROCm/triton/blob/triton-mlir/README.md) 的说明安装 ROCm's Triton flash attention（默认 triton-mlir 分支）


1. 或者，如果您选择使用 CK flash Attention，您可以安装 [flash Attention for ROCm](https://github.com/ROCm/flash-attention/tree/ck_tile)


按照 [ROCm/flash-attention](https://github.com/ROCm/flash-attention/tree/ck_tile#amd-gpurocm-support) 的说明安装 ROCm's Flash Attention (v2.5.9.post1)。用于 vLLM 的 wheels 也可以在发布版本中获取。


**注意**

* 您可能需要将「ninja」版本降级到 1.10，编译 flash-attention-2 时不会使用它（例如，通过 *pip install ninja==1.10.2.4**安装*）


1. 构建 vLLM。

```plain
cd vllm
pip install -U -r requirements-rocm.txt
python setup.py develop # This may take 5-10 minutes. Currently, `pip install .`` does not work for ROCm installation

python setup.pydevelop # 这可能需要 5-10 分钟。目前，`pip install .`不适用于 ROCm 安装
```


**提示**

例如，ROCM 6.1 上的 vLLM v0.5.3 可以通过以下步骤构建:

```plain
pip install --upgrade pip

# Install PyTorch
# 安装 PyTorch

pip uninstall torch -y
pip install --no-cache-dir --pre torch==2.5.0.dev20240726 --index-url https://download.pytorch.org/whl/nightly/rocm6.1

# Build & install AMD SMI
# 构建并安装 AMD SMI

pip install /opt/rocm/share/amd_smi

# Install dependencies
# 安装依赖项

pip install --upgrade numba scipy huggingface-hub[cli]
pip install "numpy<2"
pip install -r requirements-rocm.txt

# Apply the patch to ROCM 6.1 (requires root permission)
# 将补丁应用到 ROCM 6.1（需要 root 权限）

wget -N https://github.com/ROCm/vllm/raw/fa78403/rocm_patch/libamdhip64.so.6 -P /opt/rocm/lib
rm -f "$(python3 -c 'import torch; print(torch.__path__[0])')"/lib/libamdhip64.so*

# Build vLLM for MI210/MI250/MI300.
# 为 MI210/MI250/MI300 构建 vLLM。

export PYTORCH_ROCM_ARCH="gfx90a;gfx942"
python3 setup.py develop
```


**提示**

* 默认情况下使用 Triton flash attention。进行基准测试时，建议在收集性能数据之前运行预热步骤。

* Triton flash attention 目前不支持滑动窗口 attention。如果使用半精度，请使用 CK flash-attention 来支持滑动窗口。

* 若要使用 CK flash-attention 或 PyTorch naive Attention，请使用此标志 `export VLLM_USE_TRITON_FLASH_ATTN=0` 来关闭 triton flash attention。

* 理想情况下，PyTorch 的 ROCm 版本应与 ROCm 驱动程序版本匹配。


**提示**

* 对于 MI300x(gfx942) 用户，为了实现最佳性能，请参考 [MI300x 调优指南](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html) 以获取系统和工作流级别的性能优化和调优建议。对于 vLLM，请参考 [vLLM 性能优化](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html#vllm-performance-optimization)。


