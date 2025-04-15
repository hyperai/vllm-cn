---
title: 使用 CPU 安装
---


vLLM 最初支持在 x86 CPU 平台上的基本模型推理和服务，数据类型为 FP32 和 BF16。


目录：


1. [依赖环境](#依赖环境)

2. [使用 Dockerfile 快速开始](#使用-dockerfile-快速开始)

3. [从源代码构建](#从源代码构建)

4. [相关运行时环境变量](#相关运行时环境变量)

5. [PyTorch 的英特尔扩展](#pytorch-的英特尔扩展)

6. [性能提示](#性能提示)


## 依赖环境

* 操作系统：Linux

* 编译器：gcc/g++>=12.3.0（可选，推荐）

* 指令集架构 (ISA) 依赖：AVX512（可选，推荐）


## 使用 Dockerfile 快速开始

```plain
docker build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g .
docker run -it \
             --rm \
             --network=host \
             --cpuset-cpus=<cpu-id-list, optional> \
             --cpuset-mems=<memory-node, optional> \
             vllm-cpu-env
```


## 从源代码构建

* 首先，安装推荐的编译器。我们建议使用 `gcc/g++ >= 12.3.0` 作为默认编译器，以避免潜在的问题。例如，在 Ubuntu 22.4 上，您可以运行：

```plain
sudo apt-get update  -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```


* 其次，安装用于 vLLM CPU 后端构建的 Python 包：

```plain
pip install --upgrade pip
pip install wheel packaging ninja "setuptools>=49.4.0" numpy
pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
```


* 第三步，从源代码构建并安装 oneDNN 库：

```plain
git clone -b rls-v3.5 https://github.com/oneapi-src/oneDNN.git
cmake -B ./oneDNN/build -S ./oneDNN -G Ninja -DONEDNN_LIBRARY_TYPE=STATIC \
    -DONEDNN_BUILD_DOC=OFF \
    -DONEDNN_BUILD_EXAMPLES=OFF \
    -DONEDNN_BUILD_TESTS=OFF \
    -DONEDNN_BUILD_GRAPH=OFF \
    -DONEDNN_ENABLE_WORKLOAD=INFERENCE \
    -DONEDNN_ENABLE_PRIMITIVE=MATMUL
cmake --build ./oneDNN/build --target install --config Release


```


* 最后，构建并安装 vLLM CPU 后端：

```plain
VLLM_TARGET_DEVICE=cpu python setup.py install
```


**注意**


* BF16 是当前 CPU 后端的默认数据类型 （这意味着后端会将 FP16 转换为 BF16），并且与所有支持 AVX512 ISA 的 CPU 兼容。

* AVX512_BF16 是 ISA 的扩展，提供原生的 BF16 数据类型转换和向量积指令，与纯 AVX512 相比会带来一定的性能提升。CPU 后端构建脚本将检查主机 CPU 标志，以确定是否启用 AVX512_BF16。

* 如果要强制启用 AVX512_BF16 进行交叉编译，请在编译前设置环境变量 VLLM_CPU_AVX512BF16=1。


## 相关运行时环境变量

* `VLLM_CPU_KVCACHE_SPACE`：指定 KV 缓存大小（例如，`VLLM_CPU_KVCACHE_SPACE=40` 表示 KV 缓存空间为 40 GB），设置得越大，允许 vLLM 并行处理的请求就越多。该参数应根据用户的硬件配置和内存管理模式来设置。

* `VLLM_CPU_OMP_THREADS_BIND`: 指定专用于 OpenMP 线程的 CPU 内核。例如， `VLLM_CPU_OMP_THREADS_BIND=0-31`表示将有 32 个 OpenMP 线程绑定在 0-31 个 CPU 内核上。`VLLM_CPU_OMP_THREADS_BIND=0-31|32-63` 表示将有 2 个张量并行进程，rank0 的 32 个 OpenMP 线程绑定在 0-31 个 CPU 内核上，rank1 的 OpenMP 线程绑定在 32-63 个 CPU 内核上。


## PyTorch 的英特尔扩展

* [PyTorch](https://github.com/intel/intel-extension-for-pytorch)[ 的英特尔扩展](https://github.com/intel/intel-extension-for-pytorch)[ (IPEX)](https://github.com/intel/intel-extension-for-pytorch) 对 PyTorch 进行了扩展，增加了最新的特性优化，以便在 Intel 硬件上实现额外的性能提升。


## 性能提示

* 我们强烈建议使用 TCMalloc 来实现高性能内存分配和更好的缓存局部性。例如，在 Ubuntu 22.4 上，您可以运行：

```plain
sudo apt-get install libtcmalloc-minimal4 # install TCMalloc library

sudo apt-get install libtcmalloc-minimal4 # 安装 TCMalloc 库

find / -name *libtcmalloc* # find the dynamic link library path

find / -name *libtcmalloc* #查找动态链接库路径

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD # prepend the library to LD_PRELOAD

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD # 将库添加到 LD_PRELOAD 之前

python examples/offline_inference.py # run vLLM

python examples/offline_inference.py # 运行 vLLM
```


* 使用在线服务时，建议为服务框架预留 1-2 个 CPU 核心，以避免 CPU 超额使用。例如，在一个具有 32 个物理 CPU 核心的平台上，为框架预留 CPU 30 和 31，并将 CPU 0-29 用于 OpenMP：

```plain
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_OMP_THREADS_BIND=0-29 
vllm serve facebook/opt-125m
```


* 如果在具有超线程的计算机上使用 vLLM CPU 后端，建议使用 `VLLM_CPU_OMP_THREADS_BIND`在每个物理 CPU 核心上仅绑定一个 OpenMP 线程。在一个启用超线程且具有 16 个逻辑 CPU 核心 / 8 个物理 CPU 核心的平台上：

```plain
lscpu -e # check the mapping between logical CPU cores and physical CPU cores
lscpu -e # 查看逻辑 CPU 核和物理 CPU 核的映射关系


# The "CPU" column means the logical CPU core IDs, and the "CORE" column means the physical core IDs. On this platform, two logical cores are sharing one physical core. 
# 「CPU」列表示逻辑 CPU 核心 ID，「CORE」列表示物理核心 ID。在此平台上，两个逻辑核心共享一个物理核心。

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

# On this platform, it is recommend to only bind openMP threads on logical CPU cores 0-7 or 8-15

# 在此平台上，建议仅在逻辑 CPU 核心 0-7 或 8-15 上绑定 openMP 线程

export VLLM_CPU_OMP_THREADS_BIND=0-7 
python examples/offline_inference.py
```


* 如果在具有非统一内存访问架构（NUMA）的多插槽机器上使用 vLLM 的 CPU 后端，请注意使用 `VLLM_CPU_OMP_THREADS_BIND` 设置 CPU 核心，以避免跨 NUMA 节点的内存访问。



