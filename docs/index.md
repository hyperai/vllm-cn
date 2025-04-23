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

### 入门

安装

快速开始

示例

故障排除

常见问题

### 模型

生成模型

池化模型

支持的模型清单

内置扩展

### 功能

量化

LoRA 适配器

工具调用

推理输出

结构化输出

自动前缀缓存

解耦 (Disaggregated) 预填充（实验性）

推测解码

兼容性矩阵

### 推理与服务

离线推理

兼容 OpenAI 的服务器

多模态输入

分布式推理与服务

生产指标

引擎参数

环境变量

使用统计收集

外部集成

### 部署

使用 Docker

使用 Kubernetes

使用 Nginx

使用其他框架

外部集成

### 性能

优化与调优

基准测试套件

### 设计文档

架构概览

- 入口点

- LLM 引擎

- 工作进程 (Worker)

- 模型运行 (Model Runner)

- 模型

- 类层次结构

与 HuggingFace 集成

vLLM 插件系统

- 插件在 vLLM 中的工作原理

- vLLM 如何发现插件

- 支持的插件类型

- 插件编写指南

- 兼容性保证

vLLM 分页注意力

- 输入

- 概念

- 查询

- 键

- QK

- Softmax

- 值

- LV

- 输出

多模态数据处理

- 提示替换检测

- 标记化提示输入

- 处理器输出缓存

自动前缀缓存

- 通用缓存策略

Python 多进程

- 调试

- 介绍

- 多进程方法

- 依赖项兼容性

- 当前状态 (v0)

- v1 之前的状态

- 考虑的替代方案

- 未来工作

### V1 设计文档

自动前缀缓存

- 数据结构

- 操作

- 示例

指标

- 目标

- 背景

- v1 设计

- 已弃用的量度

- 未来的工作

- 跟踪 OpenTelemetry

### 开发者指南

为 vLLM 做出贡献

- 许可证

- 开发

- 测试

- 问题

- 拉取请求与代码审查

- 感谢

vLLM 性能分析

- 使用 PyTorch Profiler 进行分析

- 使用 NVIDIA Nsight Systems 进行配置文件

Dockerfile

添加新模型

- 基本模型实现

- 在 vLLM 中注册模型

- 编写单元测试

- 多模态支持

漏洞管理

- 报告漏洞

- 漏洞管理团队

- Slack 讨论

- 漏洞披露

### API 参考

离线推理

- LLM 类

- LLM 输入

vLLM 引擎

- LLMEngine

- AsyncLLMEngine

推理参数

- 采样参数

- 池化参数

多模态

- 模块内容

- 子模块

模型开发

- 子模块

### 社区

vLLM 博客

vLLM 见面会

赞助商

# 索引和表格

- 索引

- 模块索引
