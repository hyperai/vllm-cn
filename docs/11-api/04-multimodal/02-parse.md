---

title: 数据解析

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


## 模块内容

***class*****vllm.multimodal.parse.****ModalityDataItems****(*****data: _T*****,*****modality:***[str](https://docs.python.org/3/library/stdtypes.html#str)**)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L26)

表示 `MultiModalDataItems` 中某个模态的数据项。


***abstract*****get_count****()****→**[int](https://docs.python.org/3/library/functions.html#int)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L52)

获取数据项的数量。


***abstract*****get****(*****index:***[int](https://docs.python.org/3/library/functions.html#int)**)****→ _I**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L57)

通过索引获取数据项。


**get_all****()****→**[list](https://docs.python.org/3/library/stdtypes.html#list)**[_I]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L62)

获取所有数据项。


***abstract*****get_processor_data****()****→**[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[object](https://docs.python.org/3/library/functions.html#object)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L66)

获取传递给 HF 处理器的数据。


***abstract*****get_passthrough_data****()****→**[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[object](https://docs.python.org/3/library/functions.html#object)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L71)

获取直接传递给模型的数据。


***class*****vllm.multimodal.parse.****ProcessorBatchItems****(*****data: _T*****,*****modality:***[str](https://docs.python.org/3/library/stdtypes.html#str)**)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L77)

数据项以列表形式排列的基类。


**get_count****()****→**[int](https://docs.python.org/3/library/functions.html#int)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L80)

获取数据项的数量。


**get****(*****index:***[int](https://docs.python.org/3/library/functions.html#int)**)****→ _T**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L83)

通过索引获取数据项。


**get_processor_data****()****→**[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[object](https://docs.python.org/3/library/functions.html#object)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L86)

获取传递给 HF 处理器的数据。


**get_passthrough_data****()****→**[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[object](https://docs.python.org/3/library/functions.html#object)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L89)

获取直接传递给模型的数据。


***class*****vllm.multimodal.parse.****EmbeddingItems****(*****data: _T*****,*****modality:***[str](https://docs.python.org/3/library/stdtypes.html#str)**)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L93)

数据项表示为批处理嵌入张量或嵌入张量列表（每个数据项一个）的基类。


**get_count****()****→**[int](https://docs.python.org/3/library/functions.html#int)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L100)

获取数据项的数量。


**get****(*****index:***[int](https://docs.python.org/3/library/functions.html#int)**)****→**[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L103)

通过索引获取数据项。


**get_processor_data****()****→**[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[object](https://docs.python.org/3/library/functions.html#object)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L106)

获取传递给 HF 处理器的数据。


**get_passthrough_data****()****→**[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[object](https://docs.python.org/3/library/functions.html#object)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L109)

获取直接传递给模型的数据。


***class*****vllm.multimodal.parse.****DictEmbeddingItems****(*****data:***[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)***]*****,*****modality:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,*****required_fields:***[set](https://docs.python.org/3/library/stdtypes.html#set)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***]*****,*****fields_factory:***[Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)***[[***[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)***]],***[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[MultiModalFieldConfig](https://docs.vllm.ai/en/latest/api/multimodal/inputs.html#vllm.multimodal.inputs.MultiModalFieldConfig)***]]*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L116)

数据项表示为张量字典的基类。

通常，字典键对应于 HF 处理器的输出。


**get_count****()****→**[int](https://docs.python.org/3/library/functions.html#int)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L158)

获取数据项的数量。


**get****(*****index:***[int](https://docs.python.org/3/library/functions.html#int)**)****→**[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L161)

通过索引获取数据项。


**get_processor_data****()****→**[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[object](https://docs.python.org/3/library/functions.html#object)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L167)

获取传递给 HF 处理器的数据。


**get_passthrough_data****()****→**[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[object](https://docs.python.org/3/library/functions.html#object)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L170)

获取直接传递给模型的数据。


***class*****vllm.multimodal.parse.****AudioProcessorItems****(*****data:***[Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)***[***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[float](https://docs.python.org/3/library/functions.html#float)***] |***[numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)***|***[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)***]*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L174)


***class*****vllm.multimodal.parse.****AudioEmbeddingItems****(*****data:***[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)***]*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L184)


***class*****vllm.multimodal.parse.****ImageSize****(*****width*****,*****height*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L190)


**width*****:***[int](https://docs.python.org/3/library/functions.html#int)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L190)

字段编号 0 的别名。


**height*****:***[int](https://docs.python.org/3/library/functions.html#int)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L190)

字段编号 1 的别名。


***class*****vllm.multimodal.parse.****ImageProcessorItems****(*****data:***[Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)***[***[Image](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image)***|***[numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)***|***[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)***]*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L195)


***class*****vllm.multimodal.parse.****ImageEmbeddingItems****(*****data:***[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)***]*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L212)


***class*****vllm.multimodal.parse.****VideoProcessorItems****(*****data:***[Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)***[***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[PIL.Image.Image](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image)***] |***[numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)***|***[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)***] |***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)***]]*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L218)


***class*****vllm.multimodal.parse.****VideoEmbeddingItems****(*****data:***[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)***]*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L238)


***class*****vllm.multimodal.parse.****MultiModalDataItems****(*****dict******=None*****,*****/*****,*************kwargs*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L247)

与 `MultiModalDataDict` 类似，但经过规范化，使得每个条目对应一个列表。


**get_count****(*****modality:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,***********,*****strict:***[bool](https://docs.python.org/3/library/functions.html#bool) ***=*** ***True*****)****→**[int](https://docs.python.org/3/library/functions.html#int)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L253)

获取属于某个模态的数据项的数量。

如果 `strict=False`，即使未找到模态，也返回 `0` 而不是抛出 `KeyError`。


**get_all_counts****()****→**[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[int](https://docs.python.org/3/library/functions.html#int)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L270)

获取每个模态的数据项数量。


**get_items****(*****modality:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,*****typ:***[type](https://docs.python.org/3/library/functions.html#type)***[_D] |***[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)***[***[type](https://docs.python.org/3/library/functions.html#type)***[_D], ...]*****)****→ _D**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L274)

获取属于某个模态的数据项，并要求它们属于特定类型。


***class*****vllm.multimodal.parse.****MultiModalDataParser****(***********,*****target_sr:***[float](https://docs.python.org/3/library/functions.html#float)***|***[None](https://docs.python.org/3/library/constants.html#None) ***=*** ***None*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/parse.py#L301)

解析 `MultiModalDataDict`  到 `MultiModalDataItems` 中。


**参数：**

**target_sr** ([float](https://docs.python.org/3/library/functions.html#float)*, 可选*) – 启用自动重采样，将音频项调整为模型期望的采样率。


