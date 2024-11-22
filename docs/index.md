---
title: 索引
---


# 欢迎来到 vLLM!


vLLM 是一个快速且易于使用的库，专为大型语言模型 (LLM) 的推理和部署而设计。


vLLM 的核心特性包括：

* 最先进的服务吞吐量

* 使用 **PagedAttention** 高效管理注意力键和值的内存

* 连续批处理传入请求

* 使用 CUDA/HIP 图实现快速执行模型

* 量化： [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), INT4, INT8, 和 FP8

* 优化的 CUDA 内核，包括与 FlashAttention 和 FlashInfer 的集成

* 推测性解码

* 分块预填充


vLLM 的灵活性和易用性体现在以下方面：

* 无缝集成流行的 HuggingFace 模型

* 具有高吞吐量服务以及各种解码算法，包括*并行采样*、*束搜索*等

* 支持张量并行和流水线并行的分布式推理

* 流式输出

* 提供与 OpenAI 兼容的 API 服务器

* 支持 NVIDIA GPU、AMD CPU 和 GPU、Intel CPU 和 GPU、PowerPC CPU、TPU 以及 AWS Neuron

* 前缀缓存支持

* 支持多 LoRA


欲了解更多信息，请参阅以下内容：

* [vLLM announcing blog post](https://vllm.ai) (PagedAttention 教程)

* [vLLM paper](https://arxiv.org/abs/2309.06180) (SOSP 2023)

* [How continuous batching enables 23x throughput in LLM inference
](https://www.anyscale.com/blog/continuous-batching-llm-inference) [while reducing p50
](https://www.anyscale.com/blog/continuous-batching-llm-inference)[ ](https://www.anyscale.com/blog/continuous-batching-llm-inference)[latency](https://www.anyscale.com/blog/continuous-batching-llm-inference)
 by Cade Daniel et al.

* [vLLM 聚会](https://vllm.hyper.ai/docs/community/vllm-meetups)


## 文档

### 入门

[安装](https://vllm.hyper.ai/docs/getting-started/installation)

[使用 ROCm 进行安装](https://vllm.hyper.ai/docs/getting-started/installation-with-rocm)

[使用 OpenVINO 进行安装](https://vllm.hyper.ai/docs/getting-started/installation-with-openvino)

[使用 CPU 进行安装](https://vllm.hyper.ai/docs/getting-started/installation-with-cpu)

[使用 Neuron 进行安装](https://vllm.hyper.ai/docs/getting-started/installation-with-neuron)

[使用 TPU 进行安装](https://vllm.hyper.ai/docs/getting-started/installation-with-tpu)

[使用 XPU 进行安装](https://vllm.hyper.ai/docs/getting-started/installation-with-xpu)

[快速入门](https://vllm.hyper.ai/docs/getting-started/quickstart)

[调试提示](https://vllm.hyper.ai/docs/getting-started/debugging-tips)

[示例](https://vllm.hyper.ai/docs/getting-started/examples/)


### 部署

[OpenAI 兼容服务器](https://vllm.hyper.ai/docs/serving/openai-compatible-server)

[使用 Docker 部署](https://vllm.hyper.ai/docs/serving/deploying-with-docker)

[分布式推理和服务](https://vllm.hyper.ai/docs/serving/distributed-inference-and-serving)

[生产指标](https://vllm.hyper.ai/docs/serving/production-metrics)

[环境变量](https://vllm.hyper.ai/docs/serving/environment-variables)

[使用统计数据收集](https://vllm.hyper.ai/docs/serving/usage-stats-collection)

[整合](https://vllm.hyper.ai/docs/serving/integrations/)

[使用 CoreWeave 的 Tensorizer 加载模型](https://vllm.hyper.ai/docs/serving/tensorizer)

[兼容性矩阵](https://vllm.hyper.ai/docs/serving/compatibility%20matrix)

[常见问题解答](https://vllm.hyper.ai/docs/serving/frequently-asked-questions)


### 模型

[支持的模型](https://vllm.hyper.ai/docs/models/supported-models)

[添加新模型](https://vllm.hyper.ai/docs/models/adding-a-new-model)

[启用多模态输入](https://vllm.hyper.ai/docs/models/enabling-multimodal-inputs)

[引擎参数](https://vllm.hyper.ai/docs/models/engine-arguments)

[使用 LoRA 适配器](https://vllm.hyper.ai/docs/models/using-lora-adapters)

[使用 VLMs](https://vllm.hyper.ai/docs/models/using-vlms)

[在 vLLM 中使用推测性解码](https://vllm.hyper.ai/docs/models/speculative-decoding-in-vllm)

[性能和调优](https://vllm.hyper.ai/docs/models/performance-and-tuning)


### 量化

[量化内核支持的硬件](https://vllm.hyper.ai/docs/quantization/supported_hardware)

[AutoAWQ](https://vllm.hyper.ai/docs/quantization/autoawq)

[BitsAndBytes](https://vllm.hyper.ai/docs/quantization/bitsandbytes)

[GGUF](https://vllm.hyper.ai/docs/quantization/gguf)

[INT8 W8A8](https://vllm.hyper.ai/docs/quantization/int8-w8a8)

[FP8 W8A8](https://vllm.hyper.ai/docs/quantization/fp8-w8a8)

[FP8 E5M2 KV 缓存](https://vllm.hyper.ai/docs/quantization/fp8-e5m2-kv-cache)

[FP8 E4M3 KV 缓存](https://vllm.hyper.ai/docs/quantization/fp8-e4m3-kv-cache)


### 自动前缀缓存

[简介](https://vllm.hyper.ai/docs/automatic-prefix-caching/introduction-apc)

[实现](https://vllm.hyper.ai/docs/automatic-prefix-caching/implementation)

[广义缓存策略](https://vllm.hyper.ai/docs/automatic-prefix-caching/implementation)

### 性能基准测试

[vLLM 的基准套件](https://vllm.hyper.ai/docs/performance-benchmarks/benchmark-suites-of-vllm)


### 开发者文档

[采样参数](https://vllm.hyper.ai/docs/developer-documentation/sampling-parameters)

[离线推理](https://vllm.hyper.ai/docs/developer-documentation/offline-inference/)

- [LLM 类](https://vllm.hyper.ai/docs/developer-documentation/offline-inference/llm-class)

- [LLM 输入](https://vllm.hyper.ai/docs/developer-documentation/offline-inference/llm-inputs)

[vLLM 引擎](https://vllm.hyper.ai/docs/developer-documentation/vllm-engine/)

[LLM 引擎](https://vllm.hyper.ai/docs/developer-documentation/vllm-engine/)

- [LLMEngine](https://vllm.hyper.ai/docs/developer-documentation/vllm-engine/llmengine)

- [AsyncLLMEngine](https://vllm.hyper.ai/docs/developer-documentation/vllm-engine/asyncllmengine)

[vLLM 分页注意力](https://vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention)

- [输入处理](https://vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention#%E8%BE%93%E5%85%A5)

- [概念](https://vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention#%E6%A6%82%E5%BF%B5)

- [查询](https://vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention#%E8%AF%A2%E9%97%AE-query)

- [键](https://vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention#%E9%94%AE-key)

- [QK](https://vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention#qk)

- [Softmax](https://vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention#softmax)

- [值](https://vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention#%E5%80%BC)

- [LV](https://vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention#lv)

- [输出](https://vllm.hyper.ai/docs/developer-documentation/vllm-paged-attention#%E8%BE%93%E5%87%BA)

[输入处理](https://vllm.hyper.ai/docs/developer-documentation/input-processing/model_inputs_index)

- [指南](https://vllm.hyper.ai/docs/developer-documentation/input-processing/model_inputs_index#%E6%8C%87%E5%8D%97)

- [模块内容](https://vllm.hyper.ai/docs/developer-documentation/input-processing/model_inputs_index#%E6%A8%A1%E5%9D%97%E5%86%85%E5%AE%B9)

[多模态](https://vllm.hyper.ai/docs/developer-documentation/multi-modality/)

- [指南](https://vllm.hyper.ai/docs/developer-documentation/multi-modality/#%E6%8C%87%E5%8D%97)

- [模块内容](https://vllm.hyper.ai/docs/developer-documentation/multi-modality/#%E6%A8%A1%E5%9D%97%E5%86%85%E5%AE%B9)

[Docker 文件](https://vllm.hyper.ai/docs/developer-documentation/dockerfile)

[vLLM 性能分析](https://vllm.hyper.ai/docs/developer-documentation/profiling-vllm)

- [示例命令和用法](https://vllm.hyper.ai/docs/developer-documentation/profiling-vllm#%E5%91%BD%E4%BB%A4%E5%92%8C%E4%BD%BF%E7%94%A8%E7%A4%BA%E4%BE%8B)

- [离线推理](https://vllm.hyper.ai/docs/developer-documentation/profiling-vllm#%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86)

- [OpenAI 服务器](https://vllm.hyper.ai/docs/developer-documentation/profiling-vllm#openai-%E6%9C%8D%E5%8A%A1%E5%99%A8)


## 社区

[vLLM 聚会](https://vllm.hyper.ai/docs/community/vllm-meetups)

[赞助商](https://vllm.hyper.ai/docs/community/sponsors)


# [索引和表格](https://vllm.hyper.ai/docs/indices-and-tables/index)

* [索引](https://vllm.hyper.ai/docs/indices-and-tables/index)

* [模块索引](https://vllm.hyper.ai/docs/indices-and-tables/python-module-index)

