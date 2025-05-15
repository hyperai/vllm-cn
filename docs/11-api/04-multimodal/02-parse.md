---
title: 数据解析
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

## 模块内容

**class vllm.multimodal.parse.ModalityDataItems(_data: \_T_, _modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L26)

表示 `MultiModalDataItems` 中某个模态的数据项。

**abstract get\_count() → [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L52)

获取数据项的数量。

**abstract get(_index: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_) → \_I**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L57)

通过索引获取数据项。

**get\_all() → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[\_I\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L62)

获取所有数据项。

**abstract get\_processor\_data() → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [object](https://docs.python.org/3/library/functions.html#object "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L66)

获取传递给 HF 处理器的数据。

**abstract get\_passthrough\_data() → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [object](https://docs.python.org/3/library/functions.html#object "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L71)

获取直接传递给模型的数据。

**class vllm.multimodal.parse.ProcessorBatchItems(_data: \_T_, _modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L77)

数据项以列表形式排列的基类。

**get\_count() → [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L80)

获取数据项的数量。

**get(_index: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_) → \_T**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L83)

通过索引获取数据项。

**get\_processor\_data() → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [object](https://docs.python.org/3/library/functions.html#object "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L86)

获取传递给 HF 处理器的数据。

**get\_passthrough\_data() → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [object](https://docs.python.org/3/library/functions.html#object "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L89)

获取直接传递给模型的数据。

**class vllm.multimodal.parse.EmbeddingItems(_data: \_T_, _modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L93)

数据项表示为批处理嵌入张量或嵌入张量列表（每个数据项一个）的基类。

**get\_count() → [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L100)

获取数据项的数量。

**get(_index: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_) → [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L103)

通过索引获取数据项。

**get\_processor\_data() → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [object](https://docs.python.org/3/library/functions.html#object "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L106)

获取传递给 HF 处理器的数据。

**get\_passthrough\_data() → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [object](https://docs.python.org/3/library/functions.html#object "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L109)

获取直接传递给模型的数据。

**class vllm.multimodal.parse.DictEmbeddingItems(_data: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)")\]_, _modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _required\_fields: [set](https://docs.python.org/3/library/stdtypes.html#set "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")\]_, _fields\_factory: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable "(in Python v3.13)")\[\[[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)")\]\], [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [MultiModalFieldConfig](https://docs.vllm.ai/en/stable/api/multimodal/inputs.html#vllm.multimodal.inputs.MultiModalFieldConfig "vllm.multimodal.inputs.MultiModalFieldConfig")\]\]_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L116)

数据项表示为张量字典的基类。

通常，字典键对应于 HF 处理器的输出。

**get\_count() → [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L158)

获取数据项的数量。

**get(_index: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_) → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L161)

通过索引获取数据项。

**get\_processor\_data() → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [object](https://docs.python.org/3/library/functions.html#object "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L167)

获取传递给 HF 处理器的数据。

**get\_passthrough\_data() → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [object](https://docs.python.org/3/library/functions.html#object "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L170)

获取直接传递给模型的数据。

**class vllm.multimodal.parse.AudioProcessorItems(_data: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")\] | [numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.2)") | [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)")\]_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L174)

**class vllm.multimodal.parse.AudioEmbeddingItems(_data: [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)")\]_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L184)

**class vllm.multimodal.parse.ImageSize(_width_, _height_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L190)

**width: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L190)

字段编号 0 的别名。

**height: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L190)

字段编号 1 的别名。

**class vllm.multimodal.parse.ImageProcessorItems(_data: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[Image](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image "(in Pillow (PIL Fork) v11.2.1)") | [numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.2)") | [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)")\]_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L195)

**class vllm.multimodal.parse.ImageEmbeddingItems(_data: [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)")\]_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L212)

**class vllm.multimodal.parse.VideoProcessorItems(_data: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[PIL.Image.Image](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image "(in Pillow (PIL Fork) v11.2.1)")\] | [numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.2)") | [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.2)")\] | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)")\]\]_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L218)

**class vllm.multimodal.parse.VideoEmbeddingItems(_data: [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)")\]_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L238)

**class vllm.multimodal.parse.MultiModalDataItems(_dict\=None_, _/_, _\*\*kwargs_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L247)

与 `MultiModalDataDict` 类似，但经过规范化，使得每个条目对应一个列表。

**get\_count(_modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _\*_, _strict: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_) → [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L253)

获取属于某个模态的数据项的数量。

如果 `strict=False`，即使未找到模态，也返回 `0` 而不是抛出 `KeyError`。

**get\_all\_counts() → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L270)

获取每个模态的数据项数量。

**get\_items(_modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _typ: [type](https://docs.python.org/3/library/functions.html#type "(in Python v3.13)")\[\_D\] | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")\[[type](https://docs.python.org/3/library/functions.html#type "(in Python v3.13)")\[\_D\], ...\]_) → \_D**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L274)

获取属于某个模态的数据项，并要求它们属于特定类型。

**class vllm.multimodal.parse.MultiModalDataParser(_\*_, _target\_sr: [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _audio\_resample\_method: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.13)")\['librosa', 'scipy'\] \= 'librosa'_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L301)

解析 `MultiModalDataDict`  到 `MultiModalDataItems` 中。

**参数：**

**target_sr**（[float](https://docs.python.org/3/library/functions.html#float)_, 可选_） – 启用自动重采样，将音频项调整为模型期望的采样率。
