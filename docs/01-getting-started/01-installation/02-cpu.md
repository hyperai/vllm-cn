---
title: CPU
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM 是一个支持以下 CPU 变体的 Python 库。根据您的 CPU 类型查看厂商特定的说明：

#### Intel/AMD x86

vLLM 初步支持在 x86 CPU 平台进行基础模型推理和服务，支持 FP32、FP16 和 BF16 数据类型。

> **注意**
> 
> 此设备没有预编译的 wheel 包或镜像，您必须从源码构建 vLLM。

#### ARM AArch64

vLLM 已适配支持具备 NEON 指令集的 ARM64 CPU，基于最初为 x86 平台开发的 CPU 后端实现。

ARM CPU 后端当前支持 Float32、FP16 和 BFloat16 数据类型。

> **注意**
> 
> 此设备没有预编译的 wheel 包或镜像，您必须从源码构建 vLLM。

#### Apple silicon

vLLM 对 macOS 上的 Apple 芯片提供实验性支持。目前用户需从源码构建 vLLM 以在 macOS 上原生运行。

macOS 的 CPU 实现当前支持 FP32 和 FP16 数据类型。

> **注意**
> 
> 此设备没有预编译的 wheel 包或镜像，您必须从源码构建 vLLM。

#### IBM Z (S390X)

vLLM 对 IBM Z 平台上的 s390x 架构提供实验性支持。目前用户需从源码构建 vLLM 以在 IBM Z 平台上原生运行。

s390x 架构的 CPU 实现当前仅支持 FP32 数据类型。

> **注意**
> 
> 此设备没有预编译的 wheel 包或镜像，您必须从源码构建 vLLM。

## 系统要求

- Python: 3.9 – 3.12

#### Intel/AMD x86

- 操作系统：Linux
- 编译器：`gcc/g++ >= 12.3.0`（可选，推荐）
- 指令集架构 (ISA)：AVX512（可选，推荐）

> **提示** >[Intel Extension for PyTorch (IPEX)](https://github.com/intel/intel-extension-for-pytorch)  通过最新特性优化扩展 PyTorch，可在 Intel 硬件上获得额外性能提升。

#### ARM AArch64

- 操作系统：Linux
- 编译器：`gcc/g++ >= 12.3.0`（可选，推荐）
- 指令集架构 (ISA)：需要 NEON 支持

#### Apple silicon

- 操作系统：`macOS Sonoma`  或更高版本
- SDK：`XCode 15.4`  或更高版本（含命令行工具）
- 编译器：`Apple Clang >= 15.0.0`

#### IBM Z (S390X)

- 操作系统: `Linux`
- SDK：`gcc/g++ >= 12.3.0`  或更高版本（含命令行工具）
- 指令集架构 (ISA)：需要 VXE 支持（适用于 Z14 及以上机型）
- 需手动构建的 Python 包：`pyarrow`、`torch`  和  `torchvision`

## 使用 Python 安装

### 创建一个新的 Python 虚拟环境

您可以使用  `conda`  创建新环境：

```plain
# （推荐）创建新的 conda 环境
conda create -n vllm python=3.12 -y
conda activate vllm
```

> **注意** >[PyTorch 已弃用 conda 发布渠道](https://github.com/pytorch/pytorch/issues/138506)。若使用  `conda`，建议仅用于创建环境而非安装软件包。
> 或者可以使用超快的 Python 环境管理工具  [uv](https://docs.astral.sh/uv/)  创建环境。安装  `uv`  后执行以下命令创建新的 Python 环境：

```plain
# （推荐）创建新的 uv 环境（使用 `--seed` 安装 `pip` 和 `setuptools`）
uv venv vllm --python 3.12 --seed
source vllm/bin/activate
```

### 预编译包

当前无预编译的 CPU 版本 wheel 包。

### 从源码构建 wheel

#### Intel/AMD x86

第一步，安装推荐编译器，我们建议使用 `gcc/g++ >= 12.3.0` 作为默认编译器，避免潜在问题。例如在 Ubuntu 22.4 上你可以运行：

```go
sudo apt-get update  -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

第二步，克隆 vLLM 仓库：

```go
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source
```

第三步，安装 vLLM CPU 后端构建所需 Python 包：

```go
pip install --upgrade pip
pip install "cmake>=3.26" wheel packaging ninja "setuptools-scm>=8" numpy
pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

最后，构建并安装 vLLM CPU 后端：

```go
VLLM_TARGET_DEVICE=cpu python setup.py install
```

> **注意**
> 
> `AVX512_BF16`  指令集提供原生 BF16 数据类型转换和向量计算指令，性能优于纯 AVX512。构建脚本会自动检测 CPU 是否支持。
> 若需强制启用 AVX512_BF16（如交叉编译），可在构建前设置环境变量  `VLLM_CPU_AVX512BF16=1`。

#### ARM AArch64

第一步，安装推荐编译器，我们建议使用 `gcc/g++ >= 12.3.0` 作为默认编译器，避免潜在问题。例如在 Ubuntu 22.4 上你可以运行：

```go
sudo apt-get update  -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

第二步，克隆 vLLM 仓库：

```go
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source
```

第三步，安装 vLLM CPU 后端构建所需 Python 包：

```go
pip install --upgrade pip
pip install "cmake>=3.26" wheel packaging ninja "setuptools-scm>=8" numpy
pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

最后，构建并安装 vLLM CPU 后端：

```go
VLLM_TARGET_DEVICE=cpu python setup.py install
```

已在 AWS Graviton3 实例验证兼容性。

#### Apple silicon

在安装 XCode 和命令行工具（包括 Apple Clan）后，执行以下命令行，从源代码构建并安装 vLLM：

```go
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/cpu.txt
pip install -e .
```

> **注意**
> 
> macOS 会自动设置  `VLLM_TARGET_DEVICE=cpu`，此为当前唯一支持的设备。

#### 故障排查

若出现 C++ 头文件缺失错误（如  `'map' file not found`），尝试重新安装  [Xcode 命令行工具](https://developer.apple.com/download/all/)。

```plain
[...] fatal error: 'map' file not found
          1 | #include <map>
            |          ^~~~~
      1 error generated.
      [2/8] Building CXX object CMakeFiles/_C.dir/csrc/cpu/pos_encoding.cpp.o


[...] fatal error: 'cstddef' file not found
         10 | #include <cstddef>
            |          ^~~~~~~~~
      1 error generated.
```

#### IBM Z (S390X)

在构建 vLLM 前从包管理器中安装以下包，例如在 RHEL 9.4 中：

```go
dnf install -y \
    which procps findutils tar vim git gcc g++ make patch make cython zlib-devel \
    libjpeg-turbo-devel libtiff-devel libpng-devel libwebp-devel freetype-devel harfbuzz-devel \
    openssl-devel openblas openblas-devel wget autoconf automake libtool cmake numactl-devel
```

安装 Rust ≥1.80 `outlines-core` 和 `uvloop` python 包的安装需要它：

```go
curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    . "$HOME/.cargo/env"
```

执行以下命令，从源代码构建并安装 vLLM。

> **提示**
> 
> 在构建 vLLM 之前，请从源代码构建下列依赖：`torchvision`, `pyarrow`。

```go
    sed -i '/^torch/d' requirements-build.txt    # remove torch from requirements-build.txt since we use nightly builds
    pip install -v \
        --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
        -r requirements-build.txt \
        -r requirements-cpu.txt \
    VLLM_TARGET_DEVICE=cpu python setup.py bdist_wheel && \
    pip install dist/*.whl
```

## 使用 Docker 安装

### 预编译镜像

#### Intel/AMD x86

查看  [https://gallery.ecr.aws/q9t5s3a7/vllm-cpu-release-repo](https://gallery.ecr.aws/q9t5s3a7/vllm-cpu-release-repo)

### 从源码构建镜像

```plain
$ docker build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g .
$ docker run -it \
             --rm \
             --network=host \
             --cpuset-cpus=<cpu-id-list, optional> \
             --cpuset-mems=<memory-node, optional> \
             vllm-cpu-env
```

> **提示**
> 
> ARM 或 Apple 芯片使用  `Dockerfile.arm`

> **提示**
> 
> IBM Z（s390x）使用  `Dockerfile.s390x`，并在  `docker run`  中添加参数  `--dtype float`

## 支持的功能

vLLM CPU 后端支持以下特性：

- 张量并行 (Tensor Parallel)
- 模型量化 (`INT8 W8A8`、`AWQ`、`GPTQ`)
- 分块预填充 (Chunked-prefill
- 前缀缓存 (Prefix-caching)
- FP8-E5M2 KV 缓存

## 相关运行时环境变量

- `VLLM_CPU_KVCACHE_SPACE` : 指定 KV 缓存大小（例如  `VLLM_CPU_KVCACHE_SPACE=40`  表示 40 GiB 的 KV 缓存空间），设置更大的值可以让 vLLM 并行处理更多请求。该参数应根据硬件配置和用户的内存管理模式进行调整。
- `VLLM_CPU_OMP_THREADS_BIND` : 指定专用于 OpenMP 线程的 CPU 核心。例如：

`VLLM_CPU_OMP_THREADS_BIND=0-31`  表示将 32 个 OpenMP 线程绑定到 0-31 号 CPU 核心

`VLLM_CPU_OMP_THREADS_BIND=0-31|32-63`  表示启用 2 个张量并行进程，rank0 的 32 个 OpenMP 线程绑定到 0-31 号核心，rank1 的线程绑定到 32-63 号核心

- `VLLM_CPU_MOE_PREPACK` : 是否为 MoE 层使用预打包功能。该参数会传递给  `ipex.llm.modules.GatedMLPMOE` 。默认值为  `1` （启用）。在不支持的 CPU 上可能需要设置为  `0` （禁用）。

## 性能优化建议

- 我们强烈推荐使用 TCMalloc 实现高性能内存分配和更好的缓存局部性。例如，在 Ubuntu 22.4 上可执行：

```go
sudo apt-get install libtcmalloc-minimal4 # install TCMalloc library
find / -name *libtcmalloc* # find the dynamic link library path
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD # prepend the library to LD_PRELOAD
python examples/offline_inference/basic/basic.py # run vLLM
```

- 使用在线服务时，建议为服务框架预留 1-2 个 CPU 核心以避免 CPU 过载。例如在 32 物理核心的平台上，预留 30 和 31 号核心给框架，0-29 号核心用于 OpenMP：

```go
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_OMP_THREADS_BIND=0-29
vllm serve facebook/opt-125m
```

- 在支持超线程的机器上使用 vLLM CPU 后端时，建议通过  `VLLM_CPU_OMP_THREADS_BIND`  将每个物理 CPU 核心只绑定一个 OpenMP 线程。在 16 逻辑核心 / 8 物理核心的超线程平台上：

```plain
$ lscpu -e # check the mapping between logical CPU cores and physical CPU cores


# "CPU" 列表示逻辑核心 ID，"CORE" 列表示物理核心 ID。该平台上两个逻辑核心共享一个物理核心。
CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ      MHZ
0    0      0    0 0:0:0:0          yes 2401.0000 800.0000  800.000
1    0      0    1 1:1:1:0          yes 2401.0000 800.0000  800.000
2    0      0    2 2:2:2:0          yes 2401.0000 800.0000  800.000
3    0      0    3 3:3:3:0          yes 2401.0000 800.0000  800.000
4    0      0    4 4:4:4:0          yes 2401.0000 800.0000  800.000
5    0      0    5 5:5:5:0          yes 2401.0000 800.0000  800.000
6    0      0    6 6:6:6:0          yes 2401.0000 800.0000  800.000
7    0      0    7 7:7:7:0          yes 2401.0000 800.0000  800.000
8    0      0    0 0:0:0:0          yes 2401.0000 800.0000  800.000
9    0      0    1 1:1:1:0          yes 2401.0000 800.0000  800.000
10   0      0    2 2:2:2:0          yes 2401.0000 800.0000  800.000
11   0      0    3 3:3:3:0          yes 2401.0000 800.0000  800.000
12   0      0    4 4:4:4:0          yes 2401.0000 800.0000  800.000
13   0      0    5 5:5:5:0          yes 2401.0000 800.0000  800.000
14   0      0    6 6:6:6:0          yes 2401.0000 800.0000  800.000
15   0      0    7 7:7:7:0          yes 2401.0000 800.0000  800.000


# 在此平台上，建议仅将 OpenMP 线程绑定到 0-7 或 8-15 号逻辑核心
$ export VLLM_CPU_OMP_THREADS_BIND=0-7
$ python examples/offline_inference/basic/basic.py
```

- 在多插槽 NUMA 机器上使用 vLLM CPU 后端时，应注意通过  `VLLM_CPU_OMP_THREADS_BIND`  设置 CPU 核心，避免跨 NUMA 节点的内存访问。

## 其他注意事项

- CPU 后端与 GPU 后端有显著差异，因为 vLLM 架构最初是为 GPU 优化的。需要多项优化来提升其性能。
- 建议将 HTTP 服务组件与推理组件解耦。在 GPU 后端配置中，HTTP 服务和分词任务运行在 CPU 上，而推理运行在 GPU 上，这通常不会造成问题。但在基于 CPU 的环境中，HTTP 服务和分词可能导致显著的上下文切换和缓存效率降低。因此强烈建议分离这两个组件以获得更好的性能。
- 在启用 NUMA 的 CPU 环境中，内存访问性能可能受  [拓扑结构](https://github.com/intel/intel-extension-for-pytorch/blob/main/docs/tutorials/performance_tuning/tuning_guide.inc.md#non-uniform-memory-access-numa)  影响较大。对于 NUMA 架构，推荐两种优化方案：张量并行或数据并行。

  - 延迟敏感场景使用张量并行：遵循 GPU 后端设计，基于 NUMA 节点数量（例如双 NUMA 节点系统 TP=2）使用 Megatron-LM 的并行算法切分模型。随着  [CPU 上的 TP 功能](https://github.com/vllm-project/vllm/pull/6125#)  合并，张量并行已支持服务和离线推理。通常每个 NUMA 节点被视为一个 GPU 卡。以下是启用张量并行度为 2 的服务示例：

```go
VLLM_CPU_KVCACHE_SPACE=40 VLLM_CPU_OMP_THREADS_BIND="0-31|32-63" vllm serve meta-llama/Llama-2-7b-chat-hf -tp=2 --distributed-executor-backend mp
```

- 最大吞吐量场景使用数据并行：在每个 NUMA 节点上部署一个 LLM 服务端点，并增加一个负载均衡器来分发请求到这些端点。推荐使用  [Nginx](https://docs.vllm.ai/en/latest/deployment/nginx.html#nginxloadbalancer)  或 HAProxy 等通用解决方案。Anyscale Ray 项目提供了 LLM [服务](https://docs.ray.io/en/latest/serve/index.html)功能。这里有使用  [Ray Serve](https://github.com/intel/llm-on-ray/blob/main/docs/setup.inc.md)  设置可扩展 LLM 服务的示例。
