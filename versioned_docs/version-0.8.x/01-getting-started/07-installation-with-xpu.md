---
title: 使用 XPU 安装
---


vLLM 最初在 Intel GPU 平台上支持基本模型推理和服务。


1. [依赖环境](#依赖环境)

2. [使用 Dockerfile 快速开始](#使用-dockerfile-快速开始)

3. [从源代码构建](#从源代码构建)


## 依赖环境

* 操作系统：Linux

* 支持的硬件：英特尔数据中心 GPU（英特尔 ARC GPU WIP）

* OneAPI 要求：oneAPI 2024.1


## 使用 Dockerfile 快速开始

```plain
docker build -f Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .
docker run -it \
             --rm \
             --network=host \
             --device /dev/dri \
             -v /dev/dri/by-path:/dev/dri/by-path \
             vllm-xpu-env
```


## 从源代码构建

* 首先，安装所需的驱动程序和 intel OneAPI 2024.1 (或更高版本)。

* 其次，安装用于 vLLM XPU 后端构建的 Python 包:

```plain
source /opt/intel/oneapi/setvars.sh
pip install --upgrade pip
pip install -v -r requirements-xpu.txt 
```


* 最后，构建并安装 vLLM XPU 后端:

```plain
VLLM_TARGET_DEVICE=xpu python setup.py install
```


**注意**

* FP16 是当前 XPU 后端的默认数据类型，未来将支持 BF16 数据类型。


