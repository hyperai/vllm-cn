---
title: 使用 OpenVINO 安装
---


由 OpenVINO 驱动的 vLLM 支持来自 `vLLM 支持的模型列表 <../models/supported_models>` 中的所有 LLM 模型，并且可以在所有 x86-64 CPU 上（至少需要 AVX2 支持）进行最佳的模型服务。OpenVINO 的 vLLM 后端支持以下高级 vLLM 特性：


* 前缀缓存 （`--enable-prefix-caching`）

* 分块预填充 （`--enable-chunked-prefill`）


**目录****：**

* [依赖环境](#依赖环境)

* [使用 dockerfile 快速开始](#使用-dockerfile-快速开始)

* [从源代码安装](#从源代码安装)

* [性能提示](#性能提示)

* [局限性](#局限性)


## 依赖环境

* 操作系统：Linux

* 指令集架构 (ISA) 依赖：至少 AVX2


## 使用 Dockerfile 快速开始

```plain
docker build -f Dockerfile.openvino -t vllm-openvino-env .
docker run -it --rm vllm-openvino-env
```


## 从源代码安装

* 首先，安装 Python。例如，在 Ubuntu 22.04 上，您可以运行：

```plain
    sudo apt-get update  -y
    sudo apt-get install python3
```
* 其次，安装 vLLM OpenVINO 后端的依赖：

```plain
    pip install --upgrade pip
    pip install -r requirements-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
* 最后，安装带有 OpenVINO 后端的 vLLM：

```plain
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" VLLM_TARGET_DEVICE=openvino python -m pip install -v .
```


## 性能提示

vLLM OpenVINO 后端使用以下环境变量来控制行为：


* `VLLM_OPENVINO_KVCACHE_SPACE` 用于指定 KV 缓存大小 （例如，`VLLM_OPENVINO_KVCACHE_SPACE=40` 表示 KV 缓存空间为 40 GB），设置得越大，允许 vLLM 并行处理的请求就越多。该参数应根据用户的硬件配置和内存管理模式来设置。

* `VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8` 用于控制 KV 缓存精度。默认情况下，根据平台使用 FP16 或 BF16。

* 设置 `VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON` 在模型加载阶段启用 U8 权重压缩。默认情况下压缩是关闭的。您还可以使用 *optimum-cli* 以不同的压缩技术导出模型，并将导出的文件夹传递为  `<model_id>` 


为了实现更好的 TPOT / TTFT 延迟，您可以使用 vLLM 的分块预填充功能 (`--enable-chunked-prefill`)。根据实验，建议的批处理大小为 256 (`--max-num-batched-tokens`)


OpenVINO 最著名的配置是：

```plain
VLLM_OPENVINO_KVCACHE_SPACE=100 VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8 VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
    python3 vllm/benchmarks/benchmark_throughput.py --model meta-llama/Llama-2-7b-chat-hf --dataset vllm/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --enable-chunked-prefill --max-num-batched-tokens 256
```


## 局限性

* 不支持 LoRA 服务。

* 目前仅支持 LLM 模型。LLaVa 和编码器-解码器模型在 vLLM OpenVINO 集成中尚未启用。

* 张量和管道并行在 vLLM 集成中也未启用。


