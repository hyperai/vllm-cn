---
title: 输入定义
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

## 面向用户的输入

**vllm.multimodal.inputs.MultiModalDataDict**

[[source]](https://github.com/vllm-project/vllm/blob/main/#L0)

一个包含每种模态类型输入的字典。

内置的模态类型由 `MultiModalDataBuiltins` 定义。

别名为 `Mapping`[`str`, `Union`[`Any`, `list`[`Any`]]]

## 内部数据结构

**class vllm.multimodal.inputs.PlaceholderRange(_offset: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_, _length: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_, _is\_embed: [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.6)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L112)

基类：`TypedDict`

多模态数据的占位符位置信息。

**示例**

提示：`AAAA BBBB What is in these images?`

图像 A 和 B 将具有：

```plain
A: { "offset": 0, "length": 4 }
B: { "offset": 5, "length": 4 }
```

is\_embed_: [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")_ _\= None

[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L112)[#](https://docs.vllm.ai/en/stable/api/multimodal/inputs.html#vllm.multimodal.inputs.PlaceholderRange.is_embed "Permalink to this definition")

一个形状为 (length,) 的布尔掩码，用于指示在偏移量 offset 和 offset + length 之间的哪些位置需要分配嵌入向量。

**length: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L112)

占位符的长度。

**offset: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L112)

占位符在提示中的起始索引。

**vllm.multimodal.inputs.NestedTensors**

[[source]](https://github.com/vllm-project/vllm/blob/main/#L1588)

如果每个元素的维度不匹配，则使用列表而不是张量。

别名为 `Union`[`list`[NestedTensors], `list`[`Tensor`], `Tensor`, `tuple`[`Tensor`, …]]

**class vllm.multimodal.inputs.MultiModalFieldElem(_modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _key: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _data: list\[typing.Union\[list\[ForwardRef('NestedTensors')\], list\[torch.Tensor\], torch.Tensor, tuple\[torch.Tensor, ...\]\]\] | list\[torch.Tensor\] | torch.Tensor | tuple\[torch.Tensor, ...\]_, _field: BaseMultiModalField_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L167)

表示 `MultiModalKwargs` 中多模态项的对应关键字参数。

**data: list\[typing.Union\[list\[ForwardRef('NestedTensors')\], list\[torch.Tensor\], torch.Tensor, tuple\[torch.Tensor, ...\]\]\] | list\[torch.Tensor\] | torch.Tensor | tuple\[torch.Tensor, ...\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L167)

该字段在 `MultiModalKwargs` 中的张量数据，即传递给模型的关键字参数的值。

**field: BaseMultiModalField**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L167)

定义如何将该字段的张量数据与其他字段组合，以便为模型推理批量处理多模态项。

**key: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L167)

该字段在 `MultiModalKwargs` 中的键，即传递给模型的关键字参数的名称。

**modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L167)

对应多模态项的模态类型。每个多模态项可以由多个关键字参数组成。

**class vllm.multimodal.inputs.MultiModalFieldConfig(_field: BaseMultiModalField_, _modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L343)

**static batched(_modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L345)

定义一个字段，其中批次中的元素通过索引底层数据的第一维获得。

**参数:**

**modality** – 使用此关键字参数的多模态项的模态类型。

**示例**

```plain
Input:
    Data: [[AAAA]
        [BBBB]
        [CCCC]]


Output:
    Element 1: [AAAA]
    Element 2: [BBBB]
    Element 3: [CCCC]
```

**static flat(_modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _slices: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[slice](https://docs.python.org/3/library/functions.html#slice "(in Python v3.13)")\] | [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[slice](https://docs.python.org/3/library/functions.html#slice "(in Python v3.13)")\]\]_, _dim: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") \= 0_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L374)

定义一个字段，其中批次中的元素通过沿底层数据的第一维切片获得。

**参数:**

- **modality** – 使用此关键字参数的多模态项的模态类型。
- **slices** – 对于每个多模态项，用于提取其对应数据的切片。

**示例**

```plain
Given:
    slices: [slice(0, 3), slice(3, 7), slice(7, 9)]


Input:
    Data: [AAABBBBCC]


Output:
    Element 1: [AAA]
    Element 2: [BBBB]
    Element 3: [CC]
```

**static flat\_from\_sizes(_modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _size\_per\_item: [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)")_, _dim: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") \= 0_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L406)

定义一个字段，其中批次中的元素通过沿底层数据的第一维切片获得。

**参数:**

- **modality** – 使用此关键字参数的多模态项的模态类型。
- **slices** – 对于每个多模态项，用于提取其对应数据的切片大小。

**示例**

```plain
Given:
    size_per_item: [3, 4, 2]


Input:
    Data: [AAABBBBCC]


Output:
    Element 1: [AAA]
    Element 2: [BBBB]
    Element 3: [CC]
```

> **另请参阅**
>
>`MultiModalFieldConfig.flat()`

**static shared(_modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _batch\_size: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L449)

定义一个字段，其中批次中的元素通过获取底层数据的全部内容获得。

这意味着批次中的每个元素的数据是相同的。

**参数:**

- **modality** – 使用此关键字参数的多模态项的模态类型。
- **batch_size** – 共享此数据的多模态项的数量。

**示例**

```plain
Given:
    batch_size: 4


Input:
    Data: [XYZ]


Output:
    Element 1: [XYZ]
    Element 2: [XYZ]
    Element 3: [XYZ]
    Element 4: [XYZ]
```

**class vllm.multimodal.inputs.MultiModalKwargsItem(_dict\=None_, _/_, _\*\*kwargs_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L497)

基类：`UserDict`[`str`, `MultiModalFieldElem`]

与 `MultiModalDataItems` 中的数据项对应的 `MultiModalFieldElem` 集合。

**class_ vllm.multimodal.inputs.MultiModalKwargs(_data: \]\]_, _\*_, _items: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[MultiModalKwargsItem](https://docs.vllm.ai/en/stable/api/multimodal/inputs.html#vllm.multimodal.inputs.MultiModalKwargsItem "vllm.multimodal.inputs.MultiModalKwargsItem")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L516)

基类：`UserDict`[`str`, `Union`[`list`[NestedTensors], `list`[`Tensor`], `Tensor`, `tuple`[`Tensor`, …]]]

表示 `forward()` 的关键字参数的字典。

元数据 `items` 使我们能够通过 `get_item()` 和 `get_items()` 获取与 `MultiModalDataItems` 中每个数据项对应的关键字参数。

**static batch(_inputs\_list: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[vllm.multimodal.inputs.MultiModalKwargs](https://docs.vllm.ai/en/stable/api/multimodal/inputs.html#vllm.multimodal.inputs.MultiModalKwargs "vllm.multimodal.inputs.MultiModalKwargs")\]_) → \]\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L626)

将多个输入批量处理为一个字典。

生成的字典具有与输入相同的键。如果每个输入的对应值是一个张量且它们共享相同的形状，则输出值是一个批处理后的张量；否则，输出值是一个包含每个输入原始值的列表。

**static from\_items(_items: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[MultiModalKwargsItem](https://docs.vllm.ai/en/stable/api/multimodal/inputs.html#vllm.multimodal.inputs.MultiModalKwargsItem "vllm.multimodal.inputs.MultiModalKwargsItem")\]_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L560)

从多个项构造一个新的 `MultiModalKwargs`。

**get\_item(_modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _item\_index: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_) → [MultiModalKwargsItem](https://docs.vllm.ai/en/stable/api/multimodal/inputs.html#vllm.multimodal.inputs.MultiModalKwargsItem "vllm.multimodal.inputs.MultiModalKwargsItem")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L694)

获取由其模态类型和索引标识的项对应的关键字参数。

**get\_item\_count(_modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_) → [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L689)

获取属于某个模态类型的项的数量。

**get\_items(_modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_) → [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[MultiModalKwargsItem](https://docs.vllm.ai/en/stable/api/multimodal/inputs.html#vllm.multimodal.inputs.MultiModalKwargsItem "vllm.multimodal.inputs.MultiModalKwargsItem")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L702)

获取属于某个模态类型的每个项对应的关键字参数。

**class vllm.multimodal.inputs.MultiModalInputs**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L717)

基类：`TypedDict`

表示 `vllm.multimodal.processing.BaseMultiModalProcessor` 的输出，准备传递给 vLLM 内部。

**mm\_hashes_: MultiModalHashDict | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L717)

多模态数据的哈希值。

**mm\_kwargs_: [MultiModalKwargs](https://docs.vllm.ai/en/stable/api/multimodal/inputs.html#vllm.multimodal.inputs.MultiModalKwargs "vllm.multimodal.inputs.MultiModalKwargs")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L717)

批处理后直接传递给模型的关键字参数。

**mm\_placeholders_: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[PlaceholderRange](https://docs.vllm.ai/en/stable/api/multimodal/inputs.html#vllm.multimodal.inputs.PlaceholderRange "vllm.multimodal.inputs.PlaceholderRange")\]\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L717)

对于每种模态类型，`prompt_token_ids` 中占位符 token 的信息。

**prompt: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L717)

处理后的提示文本。

**prompt\_token\_ids_: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L717)

包含占位符 token 的处理后的 token ID。

**token\_type\_ids_: [NotRequired](https://docs.python.org/3/library/typing.html#typing.NotRequired "(in Python v3.13)")\[[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L717)

提示的 token 类型 ID。

**type_: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.13)")\['multimodal'\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L717)

输入的类型。
