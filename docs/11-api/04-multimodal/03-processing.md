---
title: 数据处理
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

## 模块内容

**vllm.multimodal.processing.PromptSeq**

[[source]](https://github.com/vllm-project/vllm/blob/main/#L1588)

一个 token 序列（token ID 列表）或文本。

别名为 `Union`[`str`, `list`[`int`]]

**class vllm.multimodal.processing.PromptIndex(_get\_match\_index: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable "(in Python v3.13)")\[\[transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | TokenizerBase, [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]\], [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")\]_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L44)

解析为提示中的索引。

**vllm.multimodal.processing.PromptTarget**

[[source]](https://github.com/vllm-project/vllm/blob/main/#L1588)

要更新的 token 序列或文本。

别名为 `Union`[`str`, `list`[`int`], `PromptIndex`]

**_class_ vllm.multimodal.processing.PromptUpdateDetails(_full: \_S_, _is\_embed: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable "(in Python v3.13)")\[\[\_BoundPromptSequence\], [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)")\] | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L105)

关于更新中包含的 token 序列或文本的详细信息。

**_full_: \_S**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L105)

完整内容。

**features: _S**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L105)

与特征占位符对应的部分内容；在模型推理期间，这部分内容将被视觉编码器的输出替换。

**vllm.multimodal.processing.PromptUpdateInfo**

[[source]](https://github.com/vllm-project/vllm/blob/main/#L1588)

更新中包含的 token 序列或文本。

如果只有部分内容对应于特征占位符，则可以使用 `PromptUpdateDetails` 来指定哪一部分。

别名为 `Union`[`str`, `list`[`int`], `PromptUpdateDetails`]

**vllm.multimodal.processing.PromptUpdateContent**

[[source]](https://github.com/vllm-project/vllm/blob/main/#L1588)

给定 `modality` 中处理项的索引，输出相应的 token 序列（或文本）。

为了方便起见，如果 token 序列（或文本）不依赖于 Importing，则可以直接传入 token 序列（或文本）而不是函数。

别名为 `Union`[`Callable`[`int`, `Union`[`str`, `list`[`int`], `PromptUpdateDetails`]], `str`, `list`[`int`], `PromptUpdateDetails`]

**_class_ vllm.multimodal.processing.UpdateMode(_value_, _names\=\_not\_given_, _\*values_, _module\=None_, _qualname\=None_, _type\=None_, _start\=1_, _boundary\=None_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L143)

**_class_ vllm.multimodal.processing.PromptUpdate(_modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _target: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptIndex](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptIndex "vllm.multimodal.processing.PromptIndex")_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L148)

定义如何使用占位符 token 更新提示。

**_modality_: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L148)

为其进行更新的模态。

**_target_: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptIndex](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptIndex "vllm.multimodal.processing.PromptIndex")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L148)

要更新的 token 序列（或文本）。

**_abstract property_ content_: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable "(in Python v3.13)")\[\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\], [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptUpdateDetails](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails "vllm.multimodal.processing.PromptUpdateDetails")\] | [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptUpdateDetails](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails "vllm.multimodal.processing.PromptUpdateDetails")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L148)

更新中包含的占位符 token。

**abstract property mode_: [UpdateMode](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.UpdateMode "vllm.multimodal.processing.UpdateMode")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L148)

定义如何更新提示。

**_class_ vllm.multimodal.processing.PromptInsertion(_modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _target: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptIndex](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptIndex "vllm.multimodal.processing.PromptIndex")_, _insertion: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable "(in Python v3.13)")\[\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\], [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptUpdateDetails](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails "vllm.multimodal.processing.PromptUpdateDetails")\] | [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptUpdateDetails](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails "vllm.multimodal.processing.PromptUpdateDetails")_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L179)

定义如何将占位符 token 插入提示中。

**示例**

对于每个图像，在 `<s>` token 后插入与视觉编码器特征大小相等的 `<image>` 特征占位符：

```plain
PromptInsertion(
    modality="image",
    target="<s>",
    insertion="<image>" * image_feature_size,
)
```

在提示的开头插入这些 token：

```plain
PromptInsertion(
    modality="image",
    target=PromptIndexTargets.start(),
    insertion="<image>" * image_feature_size,
)
```

在前缀 `Images:` 后插入这些 token：

```plain
PromptInsertion(
    modality="image",
    target=PromptIndexTargets.prefix("Images:"),
    insertion="<image>" * image_feature_size,
)
```

在提示的末尾插入这些 token：

```plain
PromptInsertion(
    modality="image",
    target=PromptIndexTargets.end(),
    insertion="<image>" * image_feature_size,
)
```

**insertion_: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable "(in Python v3.13)")\[\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\], [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptUpdateDetails](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails "vllm.multimodal.processing.PromptUpdateDetails")\] | [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptUpdateDetails](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails "vllm.multimodal.processing.PromptUpdateDetails")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L179)

给定 `modality` 中处理项的索引，输出要在 `target` 后插入的 token 序列（或文本）。

为了方便起见，如果 token 序列（或文本）不依赖于 Importing，则可以直接传入 token 序列（或文本）而不是函数。

**_property_ content_: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable "(in Python v3.13)")\[\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\], [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptUpdateDetails](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails "vllm.multimodal.processing.PromptUpdateDetails")\] | [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptUpdateDetails](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails "vllm.multimodal.processing.PromptUpdateDetails")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L179)

更新中包含的占位符 token。

**_property_ mode_: [UpdateMode](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.UpdateMode "vllm.multimodal.processing.UpdateMode")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L179)

定义如何更新提示。

**_class_ vllm.multimodal.processing.PromptReplacement(_modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _target: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptIndex](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptIndex "vllm.multimodal.processing.PromptIndex")_, _replacement: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable "(in Python v3.13)")\[\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\], [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptUpdateDetails](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails "vllm.multimodal.processing.PromptUpdateDetails")\] | [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptUpdateDetails](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails "vllm.multimodal.processing.PromptUpdateDetails")_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L246)

定义如何用占位符 token 替换输入提示的部分内容。

**示例**

对于每个图像，将提示中的一个 `<image>` 输入占位符替换为与视觉编码器特征大小相等的 `<image>` 特征占位符：

```plain
PromptReplacement(
    modality="image",
    target="<image>",
    replacement="<image>" * image_feature_size,
)
```

如上所述，但进一步用 `<image_bos>` 和 `<image_eos>` 填充特征占位符，这些 token 不应传递给视觉编码器：

```plain
PromptReplacement(
    modality="image",
    target="<image>",
    replacement=PromptUpdateDetails(
        full="".join([
            "<image_bos>",
            "<image>" * image_feature_size,
            "<image_eos>",
        ]),
        features="<image>" * image_feature_size,
    ),
)
```

为了避免在提示替换期间不必要的 token 化，建议传递 token 序列而不是文本：

```plain
PromptReplacement(
    modality="image",
    target=[image_token_id],
    replacement=PromptUpdateDetails(
        full=([image_bos_id] + [image_token_id] * image_feature_size
              + [image_eos_id]),
        features=[image_token_id] * image_feature_size,
    ),
)
```

**replacement_: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable "(in Python v3.13)")\[\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\], [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptUpdateDetails](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails "vllm.multimodal.processing.PromptUpdateDetails")\] | [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptUpdateDetails](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails "vllm.multimodal.processing.PromptUpdateDetails")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L246)

给定 `modality` 中处理项的索引，输出要替换 `target` 的 token 序列（或文本）。

为了方便起见，如果 token 序列（或文本）不依赖于 Importing，则可以直接传入 token 序列（或文本）而不是函数。

**_property_ content_: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable "(in Python v3.13)")\[\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\], [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptUpdateDetails](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails "vllm.multimodal.processing.PromptUpdateDetails")\] | [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptUpdateDetails](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails "vllm.multimodal.processing.PromptUpdateDetails")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L246)

更新中包含的占位符 token。

**_property_ mode_: [UpdateMode](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.UpdateMode "vllm.multimodal.processing.UpdateMode")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L246)

定义如何更新提示。

**vllm.multimodal.processing.full\_groupby\_modality(_values: [Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable "(in Python v3.13)")\[\_M\]_) → [ItemsView](https://docs.python.org/3/library/collections.abc.html#collections.abc.ItemsView "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[\_M\]\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L356)

便利函数，基于模态应用 `full_groupby()`。

**_class_ vllm.multimodal.processing.BoundPromptUpdate(_\_origin: [PromptUpdate](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdate "vllm.multimodal.processing.PromptUpdate")_, _tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | TokenizerBase_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L413)

一个绑定到 tokenizer 的 `PromptUpdate`，用于自动在 token 序列和文本表示之间转换 `target` 和 `get_content()` 的结果。

_property_ target_: \_BoundPromptSequence | [PromptIndex](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptIndex "vllm.multimodal.processing.PromptIndex")

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L413)

要更新的 token 序列（或文本）。

_property_ content_: [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable "(in Python v3.13)")\[\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\], [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptUpdateDetails](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails "vllm.multimodal.processing.PromptUpdateDetails")\] | [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\] | [PromptUpdateDetails](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails "vllm.multimodal.processing.PromptUpdateDetails")

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L413)

更新中包含的占位符 token。

**_property_ mode_: [UpdateMode](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.UpdateMode "vllm.multimodal.processing.UpdateMode")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L413)

定义如何更新提示。

**get\_content(_item\_idx: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_) → \_BoundPromptContent**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L450)

给定 `modality` 中处理项的索引，输出要更新的 token 序列（或文本）。

**vllm.multimodal.processing.iter\_token\_matches(_token\_ids: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_, _match\_ids: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_) → [Generator](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator "(in Python v3.13)")\[\_TokenMatch\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L486)

生成 `token_ids` 中 `match_ids` 的每次出现。

注意，空匹配会被忽略。

**vllm.multimodal.processing.replace\_token\_matches(_token\_ids: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_, _match\_ids: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_, _new\_ids: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_) → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L514)

将 `token_ids` 中 `match_ids` 的每次出现替换为 `new_ids`。

注意，空匹配会被忽略。

**_class_ vllm.multimodal.processing.PromptTargetMatch(_\_origin: [vllm.multimodal.processing.BoundPromptUpdate](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.BoundPromptUpdate "vllm.multimodal.processing.BoundPromptUpdate")_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L541)

**_class_ vllm.multimodal.processing.PlaceholderFeaturesInfo(_modality: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _item\_idx: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_, _start\_idx: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")_, _tokens: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_, _is\_embed: [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.7)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L603)

**vllm.multimodal.processing.find\_token\_matches(_prompt: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_, _prompt\_updates: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[BoundPromptUpdate](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.BoundPromptUpdate "vllm.multimodal.processing.BoundPromptUpdate")\]_) → [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[PromptTargetMatch](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptTargetMatch "vllm.multimodal.processing.PromptTargetMatch")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L621)

返回在 `prompt` 中找到的 `prompt_updates` 的每个目标。

**vllm.multimodal.processing.find\_text\_matches(_prompt: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _prompt\_updates: [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[BoundPromptUpdate](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.BoundPromptUpdate "vllm.multimodal.processing.BoundPromptUpdate")\]_) → [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[PromptTargetMatch](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptTargetMatch "vllm.multimodal.processing.PromptTargetMatch")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L647)

返回在 `prompt` 中找到的 `prompt_updates` 的每个目标。

**vllm.multimodal.processing.apply\_token\_matches(_prompt: [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_, _mm\_matches: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[PromptTargetMatch](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptTargetMatch "vllm.multimodal.processing.PromptTargetMatch")\]\]_, _mm\_item\_counts: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_) → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L746)

将 `mm_matches` 中的更新应用到 `prompt`。

**vllm.multimodal.processing.apply\_text\_matches(_prompt: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")_, _mm\_matches: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")\[[PromptTargetMatch](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.PromptTargetMatch "vllm.multimodal.processing.PromptTargetMatch")\]\]_, _mm\_item\_counts: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_) → [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L760)

将 `mm_matches` 中的更新应用到 `prompt`。

**_class_ vllm.multimodal.processing.BaseProcessingInfo(_ctx: InputProcessingContext_)[\[source\]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L1001)[#](https://docs.vllm.ai/en/stable/api/multimodal/processing.html#vllm.multimodal.processing.BaseProcessingInfo "Permalink to this definition")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L974)

提供数据处理所需信息的基类。

**get\_hf\_processor(_\*\*kwargs: [object](https://docs.python.org/3/library/functions.html#object "(in Python v3.13)")_) → transformers.ProcessorMixin**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L992)

子类可以重写此方法以处理来自模型配置或用户输入的特定 kwargs。

**_abstract_ get\_supported\_mm\_limits() → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L999)

返回每个模态支持的最大项数。

值为 `None` 表示项数无限制。

如果返回的字典中省略了某个模态，则表示完全不支持该模态。

**get\_allowed\_mm\_limits() → [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L1011)

获取每个模态的每个数据项的最大可能 token 数。

此方法返回的字典应与 `get_supported_mm_limits()` 返回的字典具有相同的键。

**_class_ vllm.multimodal.processing.BaseMultiModalProcessor(_info: \_I_, _dummy\_inputs: [BaseDummyInputsBuilder](https://docs.vllm.ai/en/stable/api/multimodal/profiling.html#vllm.multimodal.profiling.BaseDummyInputsBuilder "vllm.multimodal.profiling.BaseDummyInputsBuilder")\[\_I\]_, _\*_, _cache: ProcessingCache | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _enable\_sanity\_checks: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L1030)

处理多模态输入以用于 vLLM 的抽象基类。

不要与 `transformers.ProcessorMixin` 混淆。

**apply(_prompt: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_, _mm\_data: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [Any](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[Any](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")\]\]_, _hf\_processor\_mm\_kwargs: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [object](https://docs.python.org/3/library/functions.html#object "(in Python v3.13)")\]_, _return\_mm\_hashes: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= False_) → [MultiModalInputs](https://docs.vllm.ai/en/stable/api/multimodal/inputs.html#vllm.multimodal.inputs.MultiModalInputs "vllm.multimodal.inputs.MultiModalInputs")**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L1546)

处理多模态输入以用于 vLLM。

主要步骤包括：

1. 将 HF 处理器应用于提示文本和多模态数据，输出 token ID 和处理后的张量。

2. 在 token ID 中找到并用占位符 token 更新序列。占位符 token 的数量等于多模态编码器输出的多模态数据的特征大小。

3. 从处理后的 token ID 中提取占位符 token 的信息。

**_class_ vllm.multimodal.processing.EncDecMultiModalProcessor(_info: \_I_, _dummy\_inputs: [BaseDummyInputsBuilder](https://docs.vllm.ai/en/stable/api/multimodal/profiling.html#vllm.multimodal.profiling.BaseDummyInputsBuilder "vllm.multimodal.profiling.BaseDummyInputsBuilder")\[\_I\]_, _\*_, _cache: ProcessingCache | [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.13)") \= None_, _enable\_sanity\_checks: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= True_)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L1644)

**_abstract_ create\_encoder\_prompt(_prompt: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_, _mm\_data: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [Any](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[Any](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")\]\]_) → [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L1646)

为编码器创建输入提示。在分析和生成期间，HF 处理器将应用于此提示。

**create\_decoder\_prompt(_prompt: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_, _mm\_data: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [Any](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[Any](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")\]\]_) → [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L1658)

为解码器创建输入提示。

**apply(_prompt: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")\]_, _mm\_data: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [Any](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)") | [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.13)")\[[Any](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")\]\]_, _hf\_processor\_mm\_kwargs: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)"), [object](https://docs.python.org/3/library/functions.html#object "(in Python v3.13)")\]_, _return\_mm\_hashes: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") \= False_) → MultiModalEncDecInputs**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L1666)

处理多模态输入以用于 vLLM。主要处理步骤修改为适应编码器-解码器模型：1. 从输入提示文本创建编码器提示。2. 将 HF 处理器应用于编码器提示。3. 将输入提示文本复制为解码器提示输入。
