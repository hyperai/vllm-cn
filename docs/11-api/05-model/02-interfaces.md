---
title: 可选接口
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

## 模块内容

**vllm.model\_executor.models.interfaces.MultiModalEmbeddings**

[[source]](https://github.com/vllm-project/vllm/blob/main/#L1588)

输出嵌入必须是以下格式之一：

- **2D 张量的列表或元组，其中每个张量对应于**
  - 每个输入多模态数据项（例如，图像）。
- 单个 3D 张量，批次维度将 2D 张量分组。

别名为 `Union`[`list`[`Tensor`], `Tensor`, `tuple`[`Tensor`, …]]

**_class_ vllm.model\_executor.models.interfaces.SupportsMultiModal(_\*args_, _\*\*kwargs_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L33)

所有多模态模型所需的接口。

**supports\_multimodal_: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar "(in Python v3.13)")\[[Literal](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.13)")\[True\]\]_ _\= True_**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L33)

指示该模型支持多模态输入的标志。

> **注意**
> 
> 如果此类在模型类的 MRO 中，则无需重新定义此标志。

**get\_multimodal\_embeddings(_\*\*kwargs: [object](https://docs.python.org/3/library/functions.html#object "(in Python v3.13)")_) → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)")\] | [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)") | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")\[[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)"), ...\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L46)

返回从多模态 kwargs 生成的多模态嵌入，以便与文本嵌入合并。

> **注意**
> 
> 返回的多模态嵌入必须与输入提示中其对应的多模态数据项的出现顺序相同。

**get\_language\_model() → [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "(in PyTorch v2.7)")**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L59)[#](https://docs.vllm.ai/en/stable/api/model/interfaces.html#vllm.model_executor.models.interfaces.SupportsMultiModal.get_language_model "Permalink to this definition")

返回用于文本生成的基础语言模型。

这通常是 torch.nn.Module 实例，负责处理合并的多模态嵌入并生成隐藏状态。

返回：  
核心语言模型组件。

返回类型：  
[torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "(in PyTorch v2.7)")

**get\_input\_embeddings(_input\_ids: Tensor_, _multimodal\_embeddings: MultiModalEmbeddings | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _attn\_metadata: 'AttentionMetadata' | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → Tensor**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L2492)

**get\_input\_embeddings(_input\_ids: Tensor_, _multimodal\_embeddings: MultiModalEmbeddings | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_) → Tensor**

用于 @overload 的辅助函数，调用时抛出异常。

**_class_ vllm.model\_executor.models.interfaces.SupportsLoRA(_\*args_, _\*\*kwargs_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L111)

所有支持 LoRA 的模型所需的接口。

**supports\_lora_: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar "(in Python v3.13)")\[[Literal](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.13)")\[True\]\]_ _\= True**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L111)

指示该模型支持 LoRA 的标志。

> **注意**
> 
> 如果此类在模型类的 MRO 中，则无需重新定义此标志。

**_class_ vllm.model\_executor.models.interfaces.SupportsPP(_\*args_, _\*\*kwargs_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L189)

所有支持管道并行的模型所需的接口。

**supports\_pp_: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar "(in Python v3.13)")\[[Literal](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.13)")\[True\]\]_ _\= True**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L189)

指示该模型支持流水线并行的标志。

> **注意**
> 
> 如果此类在模型类的 MRO 中，则无需重新定义此标志。

**make\_empty\_intermediate\_tensors(_batch\_size: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_, _dtype: [torch.dtype](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype "(in PyTorch v2.7)")_, _device: [torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device "(in PyTorch v2.7)")_) → IntermediateTensors**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L202)

当 PP rank > 0 时调用，用于分析目的。

**forward(_\*_, _intermediate\_tensors: IntermediateTensors | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")_) → [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)") | IntermediateTensors**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L211)

当 PP rank > 0 时接受 `IntermediateTensors`。

仅在最后一个 PP rank 返回 `IntermediateTensors`。

**_class_ vllm.model\_executor.models.interfaces.HasInnerState(_\*args_, _\*\*kwargs_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L304)

所有具有内部状态的模型所需的接口。

**has\_inner\_state_: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar "(in Python v3.13)")\[[Literal](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.13)")\[True\]\]_ _\= True**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L304)

指示该模型具有内部状态的标志。具有内部状态的模型通常需要访问 `scheduler_config` 以获取 `max_num_seqs` 等信息。例如，Mamba 和 Jamba 都为 True。

**_class_ vllm.model\_executor.models.interfaces.IsAttentionFree(_\*args_, _\*\*kwargs_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L340)

所有像 Mamba 这样没有注意力机制但具有状态（其大小与 token 数量无关）的模型所需的接口。

**is\_attention\_free_: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar "(in Python v3.13)")\[[Literal](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.13)")\[True\]\]_ _\= True**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L340)

指示该模型没有注意力机制的标志。用于块管理器和注意力后端选择。Mamba 为 True，但 Jamba 不为 True。

**_class_ vllm.model\_executor.models.interfaces.IsHybrid(_\*args_, _\*\*kwargs_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L377)

所有像 Jamba 这样同时具有注意力和 Mamba 块的模型所需的接口，指示 `hf_config` 具有 `layers_block_type`。

**is\_hybrid_: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar "(in Python v3.13)")\[[Literal](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.13)")\[True\]\]_ _\= True**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L377)

指示该模型同时具有 Mamba 和注意力块的标志，还指示模型的 `hf_config` 具有 `layers_block_type`。

**_class_ vllm.model\_executor.models.interfaces.HasNoOps(_\*args_, _\*\*kwargs_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L414)

**_class_ vllm.model\_executor.models.interfaces.SupportsCrossEncoding(_\*args_, _\*\*kwargs_)**

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L455)[#](https://docs.vllm.ai/en/stable/api/model/interfaces.html#vllm.model_executor.models.interfaces.SupportsCrossEncoding "Permalink to this definition")

所有支持交叉编码的模型所需的接口。

**_class_ vllm.model\_executor.models.interfaces.SupportsQuant(_\*args_, _\*\*kwargs_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L448)

所有支持量化的模型所需的接口。

**_class_ vllm.model\_executor.models.interfaces.SupportsTranscription(_\*args_, _\*\*kwargs_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L478)

所有支持转录的模型所需的接口。

**_class_ vllm.model\_executor.models.interfaces.SupportsV0Only(_\*args_, _\*\*kwargs_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L505)

具有此接口的模型与 V1 vLLM 不兼容。
