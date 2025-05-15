---
title: 基本模型接口
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

## 模块内容

**_class_ vllm.model\_executor.models.interfaces\_base.VllmModel(_vllm\_config: VllmConfig_, _prefix: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") \= ''_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces_base.py#L33)

vLLM 中所有模型所需的接口。

**_class_ vllm.model\_executor.models.interfaces\_base.VllmModelForTextGeneration(_vllm\_config: VllmConfig_, _prefix: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") \= ''_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces_base.py#L94)

vLLM 中所有生成模型所需的接口。

**compute\_logits(_hidden\_states: T_, _sampling\_metadata: SamplingMetadata_) → T | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces_base.py#L98)

如果 TP rank > 0，则返回 `None`。

**_class_ vllm.model\_executor.models.interfaces\_base.VllmModelForPooling(_vllm\_config: VllmConfig_, _prefix: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") \= ''_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces_base.py#L140)

vLLM 中所有池化模型所需的接口。

**pooler(_hidden\_states: T_, _pooling\_metadata: PoolingMetadata_) → PoolerOutput**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces_base.py#L144)

仅在 TP rank 0 上调用。
