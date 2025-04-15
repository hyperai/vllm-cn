---
title: 启用多模态输入
---


本文档将引导您完成扩展 vLLM 模型的步骤，以便它接受「[多模态](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html#multi-modality) 」 输入。


**另见**

`添加新模型`


## 1. 更新基础 vLLM 模型

假设您已经按照「这些步骤（添加新模型）」在 vLLM 中实现了模型。进一步更新模型的步骤如下：

* 实现 `SupportsMultiModal` 接口。

```diff
    + from vllm.model_executor.models.interfaces import SupportsMultiModal


    - class YourModelForImage2Seq(nn.Module):
    + class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
```


**注意：**

模型类不必命名为 `*ForCausalLM`。可查看 [HuggingFace Transformers 文档](https://huggingface.co/docs/transformers/model_doc/auto#multimodal) 获取一些示例。


* 如果您还没有这样做，请为对应于多模态输入的每个输入张量在 `forward()` 中保留一个关键字参数，如下例所示: 

```diff
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    +     pixel_values: torch.Tensor,
    ) -> SamplerOutput:
```


## 2. 注册输入映射器

对于模型接受的每种模态类型，您需要使用 `MULTIMODAL_REGISTRY.register_input_mapper` 来装饰模型类。该装饰器会接收一个函数，该函数负责将多模态输入映射到您之前在 `forward()` 中预定义的关键字参数。

```diff
from vllm.model_executor.models.interfaces import SupportsMultiModal
+ from vllm.multimodal import MULTIMODAL_REGISTRY


+ @MULTIMODAL_REGISTRY.register_image_input_mapper()
class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
```
核心 vLLM 库中的每种模式都有一个默认映射器。如果您没有提供自己的函数，将使用此输入映射器。
 

**另见**

`输入处理管道`


## 3. 注册多模态 token 最大数量

对于模型作为输入所接受的每种模态类型，您需要计算每个数据实例中可能的最大 token 数，并通过 `INPUT_REGISTRY.register_dummy_data` 进行注册。

```diff
from vllm.inputs import INPUT_REGISTRY
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_image_input_mapper()
+ @MULTIMODAL_REGISTRY.register_max_image_tokens(<your_calculation>)
@INPUT_REGISTRY.register_dummy_data(<your_dummy_data_factory>)
class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
```
以下是一些示例：
* 图像输入（静态特征尺寸）: [LLaVA-1.5 模型](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llava.py)

* 图像输入（动态特征尺寸）: [LLaVA-NeXT 模型](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llava_next.py)


**另见**

`输入处理管道`


## 4.（可选）注册虚拟数据

在启动过程中，虚拟数据被传递到 vLLM 模型以分配内存。默认情况下仅包含文本输入，这可能不适用于多模态模型。在这种情况下，您可以通过 `INPUT_REGISTRY.register_dummy_data` 注册一个工厂方法来定义自己的虚拟数据。

```diff
from vllm.inputs import INPUT_REGISTRY
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(<your_calculation>)
+ @INPUT_REGISTRY.register_dummy_data(<your_dummy_data_factory>)
class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
```
**注意：**
虚拟数据应具有尽可能多的多模态标记，如上一步所述。


以下是一些示例：

* 图像输入（静态特征尺寸）: [LLaVA-1.5 模型](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llava.py)

* 图像输入（动态特征尺寸）: [LLaVA-NeXT 模型](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llava_next.py)


**另见**

`输入处理管道`


## 5.（可选）注册输入处理器

有时，需要在将输入传递给模型执行器之前，在 `LLMEngine` 层面对它们进行处理，这通常是因为与 HuggingFace Transformers 中的实现不同，多模态嵌入的重塑和/或扩展需要在模型的 `forward()` 调用之外进行。您可以通过 `INPUT_REGISTRY.register_input_processor` 注册输入处理器。

```diff
from vllm.inputs import INPUT_REGISTRY
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(<your_calculation>)
@INPUT_REGISTRY.register_dummy_data(<your_dummy_data_factory>)
+ @INPUT_REGISTRY.register_input_processor(<your_input_processor>)
class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
```
一个常见的输入处理器用例是插入占位符 tokens，以利用 vLLM 框架生成注意力掩码。以下是一些示例：

* 插入静态图像 token 数量：[LLaVA-1.5模型](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llava.py)

* 插入动态数量的图像标记：[LLaVA-NeXT 模型](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llava_next.py)


**另见**

`输入处理管道`

