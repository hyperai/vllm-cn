---

title: 基本模型接口

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


## 模块内容

***class*****vllm.model_executor.models.interfaces_base.****VllmModel****(*****vllm_config: VllmConfig*****,*****prefix:***[str](https://docs.python.org/3/library/stdtypes.html#str) ***=*** ***''*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces_base.py#L33)

vLLM 中所有模型所需的接口。


***class*****vllm.model_executor.models.interfaces_base.****VllmModelForTextGeneration****(*****vllm_config: VllmConfig*****,*****prefix:***[str](https://docs.python.org/3/library/stdtypes.html#str) ***=*** ***''*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces_base.py#L94)

vLLM 中所有生成模型所需的接口。


**compute_logits****(*****hidden_states: T*****,*****sampling_metadata: SamplingMetadata*****)****→ T |**[None](https://docs.python.org/3/library/constants.html#None)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces_base.py#L98)

如果 TP rank > 0，则返回 `None`。


**sample****(*****logits: T*****,*****sampling_metadata: SamplingMetadata*****)****→ SamplerOutput**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces_base.py#L106)

仅在 TP rank 0 上调用。


***class*****vllm.model_executor.models.interfaces_base.****VllmModelForPooling****(*****vllm_config: VllmConfig*****,*****prefix:***[str](https://docs.python.org/3/library/stdtypes.html#str) ***=*** ***''*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces_base.py#L140)

vLLM 中所有池化模型所需的接口。


**pooler****(*****hidden_states: T*****,*****pooling_metadata: PoolingMetadata*****)****→ PoolerOutput**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces_base.py#L144)

仅在 TP rank 0 上调用。


