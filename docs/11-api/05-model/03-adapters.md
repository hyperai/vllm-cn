---
title: 模型适配器
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

## 模块内容

**vllm.model\_executor.models.adapters.as\_embedding\_model(_cls: \_T_) → \_T**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/adapters.py#L116)

继承现有的 vLLM 模型以支持嵌入。

默认情况下，整个提示的嵌入是从最后一个令牌对应的归一化隐藏状态中提取的。

> **注意**
> 
> 我们假设原始模型没有添加额外的层；如果不是这种情况，请实现您自己的模型。

**vllm.model\_executor.models.adapters.as\_classification\_model(_cls: \_T_) → \_T**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/adapters.py#L146)

继承现有的 vLLM 模型以支持分类。

默认情况下，类别概率是从最后一个令牌对应的 softmax 隐藏状态中提取的。

> **注意**
> 
> 我们假设分类头是一个单独的线性层，存储为顶层模型的属性 score；如果不是这种情况，请实现您自己的模型。

**vllm.model\_executor.models.adapters.as\_reward\_model(_cls: \_T_) → \_T**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/adapters.py#L219)

继承现有的 vLLM 模型以支持奖励建模。

默认情况下，我们直接返回每个令牌的隐藏状态。

> **注意**
> 
> 我们假设原始模型没有添加额外的层；如果不是这种情况，请实现您自己的模型。
