---

title: 数据处理

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


## 模块内容

**vllm.multimodal.processing.****PromptSeq**

[[source]](https://github.com/vllm-project/vllm/blob/main/#L1588)

一个 token 序列（token ID 列表）或文本。

别名为 `Union`[`str`, `list`[`int`]]


***class*****vllm.multimodal.processing.****PromptIndex****(*****get_match_index:***[Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)***[[transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | TokenizerBase,***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***]],***[int](https://docs.python.org/3/library/functions.html#int)***|***[None](https://docs.python.org/3/library/constants.html#None)***]*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L44)

解析为提示中的索引。


**vllm.multimodal.processing.****PromptTarget**

[[source]](https://github.com/vllm-project/vllm/blob/main/#L1588)

要更新的 token 序列或文本。

别名为 `Union`[`str`, `list`[`int`], `PromptIndex`]


***class*****vllm.multimodal.processing.****PromptUpdateDetails****(*****full: _S*****,*****features: _S*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L105)

关于更新中包含的 token 序列或文本的详细信息。


**full*****: _S***

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L105)

完整内容。


**features*****: _S***

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L105)

与特征占位符对应的部分内容；在模型推理期间，这部分内容将被视觉编码器的输出替换。


**vllm.multimodal.processing.****PromptUpdateInfo**

[[source]](https://github.com/vllm-project/vllm/blob/main/#L1588)

更新中包含的 token 序列或文本。

如果只有部分内容对应于特征占位符，则可以使用 `PromptUpdateDetails` 来指定哪一部分。

别名为 `Union`[`str`, `list`[`int`], `PromptUpdateDetails`]


**vllm.multimodal.processing.****PromptUpdateContent**

[[source]](https://github.com/vllm-project/vllm/blob/main/#L1588)

给定 `modality` 中处理项的索引，输出相应的 token 序列（或文本）。

为了方便起见，如果 token 序列（或文本）不依赖于 Importing，则可以直接传入 token 序列（或文本）而不是函数。

别名为 `Union`[`Callable`[`int`, `Union`[`str`, `list`[`int`], `PromptUpdateDetails`]], `str`, `list`[`int`], `PromptUpdateDetails`]


***class*****vllm.multimodal.processing.****UpdateMode****(*****value*****,*****names******=_not_given*****,************values*****,*****module******=None*****,*****qualname******=None*****,*****type******=None*****,*****start******=1*****,*****boundary******=None*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L143)


***class*****vllm.multimodal.processing.****PromptUpdate****(*****modality:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,*****target:***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptIndex](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptIndex)**)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L148)

定义如何使用占位符 token 更新提示。


**modality*****:***[str](https://docs.python.org/3/library/stdtypes.html#str)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L148)

为其进行更新的模态。


**target*****:***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptIndex](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptIndex)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L148)

要更新的 token 序列（或文本）。


***abstract property*****content*****:***[Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)***[[***[int](https://docs.python.org/3/library/functions.html#int)***],***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptUpdateDetails](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails)***] |***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptUpdateDetails](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L148)

更新中包含的占位符 token。


***abstract property*****mode*****:***[UpdateMode](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.UpdateMode)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L148)

定义如何更新提示。


***class*****vllm.multimodal.processing.****PromptInsertion****(*****modality:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,*****target:***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptIndex](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptIndex)**,*****insertion:***[Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)***[[***[int](https://docs.python.org/3/library/functions.html#int)***],***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptUpdateDetails](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails)***] |***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptUpdateDetails](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails)**)**

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


**insertion*****:***[Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)***[[***[int](https://docs.python.org/3/library/functions.html#int)***],***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptUpdateDetails](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails)***] |***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptUpdateDetails](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L179)

给定 `modality` 中处理项的索引，输出要在 `target` 后插入的 token 序列（或文本）。

为了方便起见，如果 token 序列（或文本）不依赖于 Importing，则可以直接传入 token 序列（或文本）而不是函数。


***property*****content*****:***[Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)***[[***[int](https://docs.python.org/3/library/functions.html#int)***],***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptUpdateDetails](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails)***] |***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptUpdateDetails](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L179)

更新中包含的占位符 token。


***property*****mode*****:***[UpdateMode](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.UpdateMode)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L179)

定义如何更新提示。


***class*****vllm.multimodal.processing.****PromptReplacement****(*****modality:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,*****target:***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptIndex](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptIndex)**,*****replacement:***[Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)***[[***[int](https://docs.python.org/3/library/functions.html#int)***],***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptUpdateDetails](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails)***] |***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptUpdateDetails](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails)**)**

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


**replacement*****:***[Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)***[[***[int](https://docs.python.org/3/library/functions.html#int)***],***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptUpdateDetails](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails)***] |***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptUpdateDetails](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L246)

给定 `modality` 中处理项的索引，输出要替换 `target` 的 token 序列（或文本）。

为了方便起见，如果 token 序列（或文本）不依赖于 Importing，则可以直接传入 token 序列（或文本）而不是函数。


***property*****content*****:***[Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)***[[***[int](https://docs.python.org/3/library/functions.html#int)***],***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptUpdateDetails](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails)***] |***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptUpdateDetails](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L246)

更新中包含的占位符 token。


***property*****mode*****:***[UpdateMode](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.UpdateMode)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L246)

定义如何更新提示。


**vllm.multimodal.processing.****full_groupby_modality****(*****values:***[Iterable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)***[_M]*****)****→**[ItemsView](https://docs.python.org/3/library/collections.abc.html#collections.abc.ItemsView)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[list](https://docs.python.org/3/library/stdtypes.html#list)**[_M]]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L356)

便利函数，基于模态应用 `full_groupby()`。


***class*****vllm.multimodal.processing.****BoundPromptUpdate****(*****_origin:***[PromptUpdate](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdate)**,*****tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | TokenizerBase*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L413)

一个绑定到 tokenizer 的 `PromptUpdate`，用于自动在 token 序列和文本表示之间转换 `target` 和 `get_content()` 的结果。


***property*****target*****: _BoundPromptSequence |***[PromptIndex](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptIndex)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L413)

要更新的 token 序列（或文本）。


***property*****content*****:***[Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)***[[***[int](https://docs.python.org/3/library/functions.html#int)***],***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptUpdateDetails](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails)***] |***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***] |***[PromptUpdateDetails](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptUpdateDetails)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L413)

更新中包含的占位符 token。


***property*****mode*****:***[UpdateMode](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.UpdateMode)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L413)

定义如何更新提示。


**get_content****(*****item_idx:***[int](https://docs.python.org/3/library/functions.html#int)**)****→ _BoundPromptContent**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L450)

给定 `modality` 中处理项的索引，输出要更新的 token 序列（或文本）。


**vllm.multimodal.processing.****iter_token_matches****(*****token_ids:***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***]*****,*****match_ids:***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***]*****)****→**[Generator](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator)**[_TokenMatch]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L486)

生成 `token_ids` 中 `match_ids` 的每次出现。

注意，空匹配会被忽略。


**vllm.multimodal.processing.****replace_token_matches****(*****token_ids:***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***]*****,*****match_ids:***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***]*****,*****new_ids:***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***]*****)****→**[list](https://docs.python.org/3/library/stdtypes.html#list)**[**[int](https://docs.python.org/3/library/functions.html#int)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L514)

将 `token_ids` 中 `match_ids` 的每次出现替换为 `new_ids`。

注意，空匹配会被忽略。


***class*****vllm.multimodal.processing.****PromptTargetMatch****(*****_origin:***[vllm.multimodal.processing.BoundPromptUpdate](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.BoundPromptUpdate)**)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L541)


***class*****vllm.multimodal.processing.****PlaceholderFeaturesInfo****(*****modality:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,*****item_idx:***[int](https://docs.python.org/3/library/functions.html#int)**,*****start_idx:***[int](https://docs.python.org/3/library/functions.html#int)**,*****tokens:***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***]*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L603)


**vllm.multimodal.processing.****find_token_matches****(*****prompt:***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***]*****,*****prompt_updates:***[Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)***[***[BoundPromptUpdate](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.BoundPromptUpdate)***]*****)****→**[Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)**[**[PromptTargetMatch](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptTargetMatch)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L621)

返回在 `prompt` 中找到的 `prompt_updates` 的每个目标。


**vllm.multimodal.processing.****find_text_matches****(*****prompt:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,*****prompt_updates:***[Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)***[***[BoundPromptUpdate](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.BoundPromptUpdate)***]*****)****→**[Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)**[**[PromptTargetMatch](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptTargetMatch)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L647)

返回在 `prompt` 中找到的 `prompt_updates` 的每个目标。


**vllm.multimodal.processing.****apply_token_matches****(*****prompt:***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***]*****,*****mm_matches:***[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)***[***[PromptTargetMatch](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptTargetMatch)***]]*****,*****mm_item_counts:***[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[int](https://docs.python.org/3/library/functions.html#int)***]*****)****→**[list](https://docs.python.org/3/library/stdtypes.html#list)**[**[int](https://docs.python.org/3/library/functions.html#int)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L746)

将 `mm_matches` 中的更新应用到 `prompt`。


**vllm.multimodal.processing.****apply_text_matches****(*****prompt:***[str](https://docs.python.org/3/library/stdtypes.html#str)**,*****mm_matches:***[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)***[***[PromptTargetMatch](https://docs.vllm.ai/en/latest/api/multimodal/processing.html#vllm.multimodal.processing.PromptTargetMatch)***]]*****,*****mm_item_counts:***[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[int](https://docs.python.org/3/library/functions.html#int)***]*****)****→**[str](https://docs.python.org/3/library/stdtypes.html#str)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L760)

将 `mm_matches` 中的更新应用到 `prompt`。


***class*****vllm.multimodal.processing.****BaseProcessingInfo****(*****ctx: InputProcessingContext*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L974)

提供数据处理所需信息的基类。


**get_hf_processor****(*************kwargs:***[object](https://docs.python.org/3/library/functions.html#object)**)****→ transformers.ProcessorMixin**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L992)

子类可以重写此方法以处理来自模型配置或用户输入的特定 kwargs。


***abstract*****get_supported_mm_limits****()****→**[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[int](https://docs.python.org/3/library/functions.html#int)**|**[None](https://docs.python.org/3/library/constants.html#None)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L999)

返回每个模态支持的最大项数。

值为 `None` 表示项数无限制。

如果返回的字典中省略了某个模态，则表示完全不支持该模态。


***abstract*****get_mm_max_tokens_per_item****(*****seq_len:***[int](https://docs.python.org/3/library/functions.html#int)**,*****mm_counts:***[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[int](https://docs.python.org/3/library/functions.html#int)***]*****)****→**[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)**[**[str](https://docs.python.org/3/library/stdtypes.html#str)**,**[int](https://docs.python.org/3/library/functions.html#int)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L1011)

获取每个模态的每个数据项的最大可能 token 数。

此方法返回的字典应与 `get_supported_mm_limits()` 返回的字典具有相同的键。


***class*****vllm.multimodal.processing.****BaseMultiModalProcessor****(*****info: _I*****,*****dummy_inputs:***[BaseDummyInputsBuilder](https://docs.vllm.ai/en/latest/api/multimodal/profiling.html#vllm.multimodal.profiling.BaseDummyInputsBuilder)***[_I]*****,***********,*****cache: ProcessingCache |***[None](https://docs.python.org/3/library/constants.html#None) ***=*** ***None*****,*****enable_sanity_checks:***[bool](https://docs.python.org/3/library/functions.html#bool) ***=*** ***True*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L1030)

处理多模态输入以用于 vLLM 的抽象基类。

不要与 `transformers.ProcessorMixin` 混淆。


**apply****(*****prompt:***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***]*****,*****mm_data:***[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[Any](https://docs.python.org/3/library/typing.html#typing.Any)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[Any](https://docs.python.org/3/library/typing.html#typing.Any)***]]*****,*****hf_processor_mm_kwargs:***[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[object](https://docs.python.org/3/library/functions.html#object)***]*****,*****return_mm_hashes:***[bool](https://docs.python.org/3/library/functions.html#bool) ***=*** ***False*****)****→**[MultiModalInputs](https://docs.vllm.ai/en/latest/api/multimodal/inputs.html#vllm.multimodal.inputs.MultiModalInputs)

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L1546)

处理多模态输入以用于 vLLM。

主要步骤包括：

1. 将 HF 处理器应用于提示文本和多模态数据，输出 token ID 和处理后的张量。

2. 在 token ID 中找到并用占位符 token 更新序列。占位符 token 的数量等于多模态编码器输出的多模态数据的特征大小。

3. 从处理后的 token ID 中提取占位符 token 的信息。


***class*****vllm.multimodal.processing.****EncDecMultiModalProcessor****(*****info: _I*****,*****dummy_inputs:***[BaseDummyInputsBuilder](https://docs.vllm.ai/en/latest/api/multimodal/profiling.html#vllm.multimodal.profiling.BaseDummyInputsBuilder)***[_I]*****,***********,*****cache: ProcessingCache |***[None](https://docs.python.org/3/library/constants.html#None) ***=*** ***None*****,*****enable_sanity_checks:***[bool](https://docs.python.org/3/library/functions.html#bool) ***=*** ***True*****)**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L1644)

***abstract*****create_encoder_prompt****(*****prompt:***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***]*****,*****mm_data:***[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[Any](https://docs.python.org/3/library/typing.html#typing.Any)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[Any](https://docs.python.org/3/library/typing.html#typing.Any)***]]*****)****→**[str](https://docs.python.org/3/library/stdtypes.html#str)**|**[list](https://docs.python.org/3/library/stdtypes.html#list)**[**[int](https://docs.python.org/3/library/functions.html#int)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L1646)

为编码器创建输入提示。在分析和生成期间，HF 处理器将应用于此提示。


**create_decoder_prompt****(*****prompt:***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***]*****,*****mm_data:***[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[Any](https://docs.python.org/3/library/typing.html#typing.Any)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[Any](https://docs.python.org/3/library/typing.html#typing.Any)***]]*****)****→**[str](https://docs.python.org/3/library/stdtypes.html#str)**|**[list](https://docs.python.org/3/library/stdtypes.html#list)**[**[int](https://docs.python.org/3/library/functions.html#int)**]**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L1658)

为解码器创建输入提示。


**apply****(*****prompt:***[str](https://docs.python.org/3/library/stdtypes.html#str)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[int](https://docs.python.org/3/library/functions.html#int)***]*****,*****mm_data:***[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[Any](https://docs.python.org/3/library/typing.html#typing.Any)***|***[list](https://docs.python.org/3/library/stdtypes.html#list)***[***[Any](https://docs.python.org/3/library/typing.html#typing.Any)***]]*****,*****hf_processor_mm_kwargs:***[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)***[***[str](https://docs.python.org/3/library/stdtypes.html#str)***,***[object](https://docs.python.org/3/library/functions.html#object)***]*****,*****return_mm_hashes:***[bool](https://docs.python.org/3/library/functions.html#bool) ***=*** ***False*****)****→ MultiModalEncDecInputs**

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing.py#L1666)

处理多模态输入以用于 vLLM。主要处理步骤修改为适应编码器-解码器模型：1. 从输入提示文本创建编码器提示。2. 将 HF 处理器应用于编码器提示。3. 将输入提示文本复制为解码器提示输入。


