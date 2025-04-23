---

title: 可选接口

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


## 模块内容

**vllm.model_executor.models.interfaces.****MultiModalEmbeddings**

[[source]](https://github.com/vllm-project/vllm/blob/main/#L1588)

输出嵌入必须是以下格式之一：

* **2D 张量的列表或元组，其中每个张量对应于**

   * 每个输入多模态数据项（例如，图像）。

* 单个 3D 张量，批次维度将 2D 张量分组。

别名为 `Union`[`list`[`Tensor`], `Tensor`, `tuple`[`Tensor`, …]]


***class*****vllm.model_executor.models.interfaces.****SupportsMultiModal****(************args*****,*************kwargs*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L33)

所有多模态模型所需的接口。


**supports_multimodal*****:***[ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)***[***[Literal](https://docs.python.org/3/library/typing.html#typing.Literal)***[True]] = True***

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L33)

指示该模型支持多模态输入的标志。


>**注意**
>如果此类在模型类的 MRO 中，则无需重新定义此标志。

**get_multimodal_embeddings****(*************kwargs:***[object](https://docs.python.org/3/library/functions.html#object)**)****→**[list](https://docs.python.org/3/library/stdtypes.html#list)**[**[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)**] |**[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)**|**[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)**[**[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)**, ...] |**[None](https://docs.python.org/3/library/constants.html#None)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L46)

返回从多模态 kwargs 生成的多模态嵌入，以便与文本嵌入合并。


>**注意**
>返回的多模态嵌入必须与输入提示中其对应的多模态数据项的出现顺序相同。

**get_input_embeddings****(*****input_ids: Tensor*****,*****multimodal_embeddings: MultiModalEmbeddings |***[None](https://docs.python.org/3/library/constants.html#None) ***=*** ***None*****,*****attn_metadata: 'AttentionMetadata' |***[None](https://docs.python.org/3/library/constants.html#None) ***=*** ***None*****)****→ Tensor**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L2492)

**get_input_embeddings****(*****input_ids: Tensor*****,*****multimodal_embeddings: MultiModalEmbeddings |***[None](https://docs.python.org/3/library/constants.html#None) ***=*** ***None*****)****→ Tensor**

用于 @overload 的辅助函数，调用时抛出异常。


***class*****vllm.model_executor.models.interfaces.****SupportsLoRA****(************args*****,*************kwargs*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L111)

所有支持 LoRA 的模型所需的接口。


**supports_lora*****:***[ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)***[***[Literal](https://docs.python.org/3/library/typing.html#typing.Literal)***[True]] = True***

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L111)

指示该模型支持 LoRA 的标志。


>**注意**
>如果此类在模型类的 MRO 中，则无需重新定义此标志。

***class*****vllm.model_executor.models.interfaces.****SupportsPP****(************args*****,*************kwargs*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L189)

所有支持管道并行的模型所需的接口。


**supports_pp*****:***[ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)***[***[Literal](https://docs.python.org/3/library/typing.html#typing.Literal)***[True]] = True***

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L189)

指示该模型支持流水线并行的标志。


>**注意**
>如果此类在模型类的 MRO 中，则无需重新定义此标志。

**make_empty_intermediate_tensors****(*****batch_size:***[int](https://docs.python.org/3/library/functions.html#int)**,*****dtype:***[torch.dtype](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)**,*****device:***[torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)**)****→ IntermediateTensors**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L202)

当 PP rank > 0 时调用，用于分析目的。


**forward****(***********,*****intermediate_tensors: IntermediateTensors |***[None](https://docs.python.org/3/library/constants.html#None)**)****→**[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)**| IntermediateTensors**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L211)

当 PP rank > 0 时接受 `IntermediateTensors`。

仅在最后一个 PP rank 返回 `IntermediateTensors`。


***class*****vllm.model_executor.models.interfaces.****HasInnerState****(************args*****,*************kwargs*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L304)

所有具有内部状态的模型所需的接口。


**has_inner_state*****:***[ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)***[***[Literal](https://docs.python.org/3/library/typing.html#typing.Literal)***[True]] = True***

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L304)

指示该模型具有内部状态的标志。具有内部状态的模型通常需要访问 `scheduler_config` 以获取 `max_num_seqs` 等信息。例如，Mamba 和 Jamba 都为 True。


***class*****vllm.model_executor.models.interfaces.****IsAttentionFree****(************args*****,*************kwargs*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L340)

所有像 Mamba 这样没有注意力机制但具有状态（其大小与 token 数量无关）的模型所需的接口。


**is_attention_free*****:***[ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)***[***[Literal](https://docs.python.org/3/library/typing.html#typing.Literal)***[True]] = True***

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L340)

指示该模型没有注意力机制的标志。用于块管理器和注意力后端选择。Mamba 为 True，但 Jamba 不为 True。


***class*****vllm.model_executor.models.interfaces.****IsHybrid****(************args*****,*************kwargs*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L377)

所有像 Jamba 这样同时具有注意力和 Mamba 块的模型所需的接口，指示 `hf_config` 具有 `layers_block_type`。


**is_hybrid*****:***[ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)***[***[Literal](https://docs.python.org/3/library/typing.html#typing.Literal)***[True]] = True***

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L377)

指示该模型同时具有 Mamba 和注意力块的标志，还指示模型的 `hf_config` 具有 `layers_block_type`。


***class*****vllm.model_executor.models.interfaces.****SupportsCrossEncoding****(************args*****,*************kwargs*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L414)

所有支持交叉编码的模型所需的接口。


***class*****vllm.model_executor.models.interfaces.****SupportsQuant****(************args*****,*************kwargs*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L448)

所有支持量化的模型所需的接口。


***class*****vllm.model_executor.models.interfaces.****SupportsTranscription****(************args*****,*************kwargs*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L478)

所有支持转录的模型所需的接口。


***class*****vllm.model_executor.models.interfaces.****SupportsV0Only****(************args*****,*************kwargs*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py#L505)

具有此接口的模型与 V1 vLLM 不兼容。


