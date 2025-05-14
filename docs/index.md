---
title: 欢迎来到 vLLM！

sidebar_position: 0
---

# 欢迎来到 vLLM！

![图片](/img/vllm-logo.png)

---

vLLM 是一个快速、易于使用的 LLM 推理和服务库。

最初 vLLM 是在加州大学伯克利分校的[天空计算实验室 (Sky Computing Lab) ](https://sky.cs.berkeley.edu/)开发的，如今已发展成为一个由学术界和工业界共同贡献的社区驱动项目。

vLLM 具有以下功能：

- 最先进的服务吞吐量

- 使用 [PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) 高效管理注意力键和值的内存

- 连续批处理传入请求

- 使用 CUDA/HIP 图实现快速执行模型

- 量化：[GPTQ](https://arxiv.org/abs/2210.17323)、[AWQ](https://arxiv.org/abs/2306.00978)、INT4、INT8 和 FP8

- 优化 CUDA 内核，包括与 FlashAttention 和 FlashInfer 的集成

- 推测性解码

- 分块预填充

vLLM 在以下方面非常灵活且易于使用：

- 无缝集成流行的 HuggingFace 模型

- 使用各种解码算法实现高吞吐量服务，包括*并行采样*、*束搜索*等

- 支持张量并行和流水线并行的分布式推理

- 流式输出

- OpenAI 兼容 API 服务器

- 支持 NVIDIA GPU、AMD CPU 和 GPU、Intel CPU 和 GPU、PowerPC CPU、TPU 以及 AWS Neuron

- 前缀缓存支持

- 多 LoRA 支持

欲了解更多信息，请参阅以下内容：

- [vLLM announcing blog post](https://vllm.ai) (PagedAttention 教程)

- [vLLM paper](https://arxiv.org/abs/2309.06180) (SOSP 2023)

- [How continuous batching enables 23x throughput in LLM inference
  ](https://www.anyscale.com/blog/continuous-batching-llm-inference) [while reducing p50
  ](https://www.anyscale.com/blog/continuous-batching-llm-inference)[ ](https://www.anyscale.com/blog/continuous-batching-llm-inference)[latency](https://www.anyscale.com/blog/continuous-batching-llm-inference)
  by Cade Daniel et al.

- [vLLM 会议](https://docs.vllm.ai/en/latest/community/meetups.html#meetups)

## 文档

### 快速开始

[安装](https://vllm.hyper.ai/docs/getting-started/installation/)  
[快速开始](https://vllm.hyper.ai/docs/getting-started/quickstart)  
[示例](https://vllm.hyper.ai/docs/getting-started/examples/offline-inference/)  
[故障排除](https://vllm.hyper.ai/docs/getting-started/troubleshooting)  
[常见问题](https://vllm.hyper.ai/docs/getting-started/faq)  
[vLLM V1 用户指南](https://vllm.hyper.ai/docs/getting-started/v1-user-guide)  

### 支持模型

[支持模型列表](https://vllm.hyper.ai/docs/models/supported_models)  
[生成模型](https://vllm.hyper.ai/docs/models/generative_models)  
[池化模型](https://vllm.hyper.ai/docs/models/Pooling%20Models)  
[内置扩展](https://vllm.hyper.ai/docs/models/extensions/)

### 功能特性

[量化](https://vllm.hyper.ai/docs/features/quantization/)  
[LoRA 适配器](https://vllm.hyper.ai/docs/features/lora)  
[工具调用](https://vllm.hyper.ai/docs/features/tool_calling)  
[推理输出](https://vllm.hyper.ai/docs/features/reasoning_outputs)  
[结构化输出](https://vllm.hyper.ai/docs/features/structured_outputs)  
[自动前缀缓存](https://vllm.hyper.ai/docs/features/automatic_prefix_caching)  
[分离式预填充（实验性功能）](https://vllm.hyper.ai/docs/features/disagg_prefill)  
[分离式预填充（实验性功能）](https://vllm.hyper.ai/docs/features/spec_decode)  
[兼容矩阵](https://vllm.hyper.ai/docs/features/compatibility_matrix)

### 训练

[Transformers 强化学习](https://vllm.hyper.ai/docs/training/trl)  
[RLHF 基于人类反馈的强化学习](https://vllm.hyper.ai/docs/training/rlhf)  

### 推理与服务

[离线推理](https://vllm.hyper.ai/docs/inference-and-serving/offline_inference)  
[兼容 OpenAI 的服务器](https://vllm.hyper.ai/docs/inference-and-serving/openai_compatible_server)  
[多模态输入](https://vllm.hyper.ai/docs/inference-and-serving/multimodal_inputs)  
[分布式推理与服务](https://vllm.hyper.ai/docs/inference-and-serving/distributed_serving_new)  
[生产指标](https://vllm.hyper.ai/docs/inference-and-serving/metrics)  
[引擎参数](https://vllm.hyper.ai/docs/inference-and-serving/engine_args)  
[环境变量](https://vllm.hyper.ai/docs/inference-and-serving/env_vars)  
[使用统计数据收集](https://vllm.hyper.ai/docs/inference-and-serving/usage_stats)  
[外部集成](https://vllm.hyper.ai/docs/inference-and-serving/integrations/)

### 部署

[使用 Docker](https://vllm.hyper.ai/docs/deployment/docker)  
[使用 Kubernetes](https://vllm.hyper.ai/docs/deployment/k8s)  
[使用 Nginx](https://vllm.hyper.ai/docs/deployment/nginx)  
[使用其他框架](https://vllm.hyper.ai/docs/deployment/framworks/)  
[外部集成](https://vllm.hyper.ai/docs/deployment/integrations/)  

### 性能

[优化与调优](https://vllm.hyper.ai/docs/performance/optimization)  

[基准测试套件](https://vllm.hyper.ai/docs/performance/benchmarks)  

### 设计文档

[架构概览](https://vllm.hyper.ai/docs/design/arch_overview)  
- [入口点](https://vllm.hyper.ai/docs/design/arch_overview#%E5%85%A5%E5%8F%A3%E7%82%B9)  
- [LLM 引擎](https://vllm.hyper.ai/docs/design/arch_overview#llm-%E5%BC%95%E6%93%8E)  
- [工作进程 (Worker)](https://vllm.hyper.ai/docs/design/arch_overview#worker)  
- [模型运行 (Model Runner)](https://vllm.hyper.ai/docs/design/arch_overview#%E6%A8%A1%E5%9E%8B%E8%BF%90%E8%A1%8C%E5%99%A8)  
- [模型](https://vllm.hyper.ai/docs/design/arch_overview#%E6%A8%A1%E5%9E%8B)  
- [类层次结构](https://vllm.hyper.ai/docs/design/arch_overview#%E7%B1%BB%E5%B1%82%E6%AC%A1%E7%BB%93%E6%9E%84)
 
[与 HuggingFace 集成](https://vllm.hyper.ai/docs/design/huggingface_integration) 

[vLLM 插件系统](https://vllm.hyper.ai/docs/design/plugin_system)  
- [vLLM 插件中的工作原理](https://vllm.hyper.ai/docs/design/plugin_system#vllm-%E4%B8%AD%E6%8F%92%E4%BB%B6%E7%9A%84%E5%B7%A5%E4%BD%9C%E5%8E%9F%E7%90%86)  
- [vLLM 如何发现插件](https://vllm.hyper.ai/docs/design/plugin_system#vllm-%E5%A6%82%E4%BD%95%E5%8F%91%E7%8E%B0%E6%8F%92%E4%BB%B6)  
- [支持的插件类型](https://vllm.hyper.ai/docs/design/plugin_system#%E6%94%AF%E6%8C%81%E7%9A%84%E6%8F%92%E4%BB%B6%E7%B1%BB%E5%9E%8B)  
- [插件编写指南](https://vllm.hyper.ai/docs/design/plugin_system#%E7%BC%96%E5%86%99%E6%8F%92%E4%BB%B6%E6%8C%87%E5%8D%97)  
- [兼容性保证](https://vllm.hyper.ai/docs/design/plugin_system#%E5%85%BC%E5%AE%B9%E6%80%A7%E4%BF%9D%E8%AF%81)

[vLLM 分页注意力](https://vllm.hyper.ai/docs/design/paged_attention)  
- [输入](https://vllm.hyper.ai/docs/design/paged_attention#%E8%BE%93%E5%85%A5)  
- [概念](https://vllm.hyper.ai/docs/design/paged_attention#%E6%A6%82%E5%BF%B5)  
- [查询](https://vllm.hyper.ai/docs/design/paged_attention#%E6%9F%A5%E8%AF%A2)  
- [键](https://vllm.hyper.ai/docs/design/paged_attention#key)  
- [QK](https://vllm.hyper.ai/docs/design/paged_attention#qk)  
- [Softmax](https://vllm.hyper.ai/docs/design/paged_attention#softmax)  
- [值](https://vllm.hyper.ai/docs/design/paged_attention#%E5%80%BC)  
- [LV](https://vllm.hyper.ai/docs/design/paged_attention#lv)  
- [输出](https://vllm.hyper.ai/docs/design/paged_attention#%E8%BE%93%E5%87%BA)

[多模态数据处理](https://vllm.hyper.ai/docs/design/mm_processing)  
- [提示更新检测](https://vllm.hyper.ai/docs/design/mm_processing#%E6%8F%90%E7%A4%BA%E6%9B%B4%E6%96%B0%E6%A3%80%E6%B5%8B)  
- [分词后提示输入](https://vllm.hyper.ai/docs/design/mm_processing#%E5%88%86%E8%AF%8D%E5%90%8E%E7%9A%84%E6%8F%90%E7%A4%BA%E8%BE%93%E5%85%A5)  
- [处理器输出缓存](https://vllm.hyper.ai/docs/design/mm_processing#%E5%A4%84%E7%90%86%E5%99%A8%E8%BE%93%E5%87%BA%E7%BC%93%E5%AD%98)

[自动前缀缓存](https://vllm.hyper.ai/docs/design/automatic_prefix_caching)  
- 通用缓存策略

[Python 多进程](https://vllm.hyper.ai/docs/design/multiprocessing)  
- [调试](https://vllm.hyper.ai/docs/design/multiprocessing#%E8%B0%83%E8%AF%95)  
- [介绍](https://vllm.hyper.ai/docs/design/multiprocessing#%E4%BB%8B%E7%BB%8D)  
- [多进程方法](https://vllm.hyper.ai/docs/design/multiprocessing#%E5%A4%9A%E8%BF%9B%E7%A8%8B%E6%96%B9%E6%B3%95)  
- [依赖项兼容性](https://vllm.hyper.ai/docs/design/multiprocessing#%E4%B8%8E%E4%BE%9D%E8%B5%96%E9%A1%B9%E7%9A%84%E5%85%BC%E5%AE%B9%E6%80%A7)  
- [当前状态 (v0)](https://vllm.hyper.ai/docs/design/multiprocessing#%E5%BD%93%E5%89%8D%E7%8A%B6%E6%80%81v0)  
- [v1 之前的状态](https://vllm.hyper.ai/docs/design/multiprocessing#v1-%E4%B8%AD%E7%9A%84%E5%85%88%E5%89%8D%E7%8A%B6%E6%80%81)  
- [考虑的替代方案](https://vllm.hyper.ai/docs/design/multiprocessing#%E8%80%83%E8%99%91%E7%9A%84%E6%9B%BF%E4%BB%A3%E6%96%B9%E6%A1%88)  
- [未来工作](https://vllm.hyper.ai/docs/design/multiprocessing#%E6%9C%AA%E6%9D%A5%E5%B7%A5%E4%BD%9C)  

### V1 设计文档

[vLLM 的 `torch.compile` 集成](https://vllm.hyper.ai/docs/design-v1/torch_compile) 

[自动前缀缓存](https://vllm.hyper.ai/docs/design-v1/prefix_caching)  
- [数据结构](https://vllm.hyper.ai/docs/design-v1/prefix_caching#%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84)  
- [操作](https://vllm.hyper.ai/docs/design-v1/prefix_caching#%E6%93%8D%E4%BD%9C)  
- [示例](https://vllm.hyper.ai/docs/design-v1/prefix_caching#%E7%A4%BA%E4%BE%8B)

[指标](https://vllm.hyper.ai/docs/design-v1/metrics)  
- [目标](https://vllm.hyper.ai/docs/design-v1/metrics#%E7%9B%AE%E6%A0%87)  
- [背景](https://vllm.hyper.ai/docs/design-v1/metrics#%E8%83%8C%E6%99%AF)  
- [v1 设计](https://vllm.hyper.ai/docs/design-v1/metrics#v1-%E8%AE%BE%E8%AE%A1)  
- [已弃用的量度](https://vllm.hyper.ai/docs/design-v1/metrics#%E5%B7%B2%E5%BC%83%E7%94%A8%E7%9A%84%E6%8C%87%E6%A0%87)  
- [未来的工作](https://vllm.hyper.ai/docs/design-v1/metrics#%E6%9C%AA%E6%9D%A5%E5%B7%A5%E4%BD%9C)  
- [跟踪 OpenTelemetry](https://vllm.hyper.ai/docs/design-v1/metrics#%E8%B7%9F%E8%B8%AA---opentelemetry)  

### 开发者指南

[为 vLLM 做出贡献](https://vllm.hyper.ai/docs/contributing/overview)  
- [许可证](https://vllm.hyper.ai/docs/contributing/overview#%E8%AE%B8%E5%8F%AF%E8%AF%81)  
- [开发指南](https://vllm.hyper.ai/docs/contributing/overview#%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97)  
- [测试](https://vllm.hyper.ai/docs/contributing/overview#%E6%B5%8B%E8%AF%95)  
- [问题报告](https://vllm.hyper.ai/docs/contributing/overview#%E9%97%AE%E9%A2%98%E6%8A%A5%E5%91%8A)  
- [拉取请求与代码审查](https://vllm.hyper.ai/docs/contributing/overview#%E6%8B%89%E5%8F%96%E8%AF%B7%E6%B1%82%E4%B8%8E%E4%BB%A3%E7%A0%81%E5%AE%A1%E6%9F%A5)  
- [致谢](https://vllm.hyper.ai/docs/contributing/overview#%E8%87%B4%E8%B0%A2)

[vLLM 性能分析](https://vllm.hyper.ai/docs/contributing/profiling_index)  
- [使用 PyTorch Profiler 进行分析](https://vllm.hyper.ai/docs/contributing/profiling_index#%E4%BD%BF%E7%94%A8-pytorch-profiler-%E8%BF%9B%E8%A1%8C%E5%88%86%E6%9E%90)  
- [使用 NVIDIA Nsight Systems 进行配置文件](https://vllm.hyper.ai/docs/contributing/profiling_index#%E4%BD%BF%E7%94%A8-nvidia-nsight-systems-%E8%BF%9B%E8%A1%8C%E5%88%86%E6%9E%90)

[Dockerfile](https://vllm.hyper.ai/docs/contributing/dockerfile)  

[添加新模型](https://vllm.hyper.ai/docs/contributing/model/)  
- [实现基础模型](https://vllm.hyper.ai/docs/contributing/model/basic)  
- [在 vLLM 中注册模型](https://vllm.hyper.ai/docs/contributing/model/registration)  
- [编写单元测试](https://vllm.hyper.ai/docs/contributing/model/tests)  
- [多模态支持](https://vllm.hyper.ai/docs/contributing/model/multimodal)

[漏洞管理](https://vllm.hyper.ai/docs/contributing/vulnerability_management)  
- [报告漏洞](https://vllm.hyper.ai/docs/contributing/vulnerability_management#%E6%8A%A5%E5%91%8A%E6%BC%8F%E6%B4%9E)  
- [漏洞管理团队](https://vllm.hyper.ai/docs/contributing/vulnerability_management#%E6%BC%8F%E6%B4%9E%E7%AE%A1%E7%90%86%E5%9B%A2%E9%98%9F)  
- [Slack 讨论](https://vllm.hyper.ai/docs/contributing/vulnerability_management#slack-%E8%AE%A8%E8%AE%BA)  
- [漏洞披露](https://vllm.hyper.ai/docs/contributing/vulnerability_management#%E6%BC%8F%E6%B4%9E%E6%8A%AB%E9%9C%B2)  

### API 参考

[离线推理](https://vllm.hyper.ai/docs/api/offline_interence/)  
- [LLM 类](https://vllm.hyper.ai/docs/api/offline_interence/LLM)  
- [LLM 输入](https://vllm.hyper.ai/docs/api/offline_interence/llm_inputs)

[vLLM 引擎](https://vllm.hyper.ai/docs/api/engine/)  
- [LLMEngine](https://vllm.hyper.ai/docs/api/engine/llm_engine)  
- [AsyncLLMEngine](https://vllm.hyper.ai/docs/api/engine/async_llm_engine)

[推理参数](https://vllm.hyper.ai/docs/api/inference_params)  
- [采样参数](https://vllm.hyper.ai/docs/api/inference_params#%E9%87%87%E6%A0%B7%E5%8F%82%E6%95%B0)  
- [池化参数](https://vllm.hyper.ai/docs/api/inference_params#%E6%B1%A0%E5%8C%96%E5%8F%82%E6%95%B0)

- [多模态支持](https://vllm.hyper.ai/docs/api/multimodal/)  
- [模块内容](https://vllm.hyper.ai/docs/api/multimodal/#%E6%A8%A1%E5%9D%97%E5%86%85%E5%AE%B9)  

- [子模块](https://vllm.hyper.ai/docs/api/multimodal/#%E5%AD%90%E6%A8%A1%E5%9D%97)

[模型开发](https://vllm.hyper.ai/docs/api/model/interfaces)  
- 子模块

### 社区

[vLLM 博客](https://vllm.hyper.ai/docs/community/blog)  
[vLLM 会议](https://vllm.hyper.ai/docs/community/meetups)  
[赞助商](https://vllm.hyper.ai/docs/community/sponsors)
