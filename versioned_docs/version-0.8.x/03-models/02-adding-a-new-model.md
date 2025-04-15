---
title: 添加新模型
---


本文档提供将 [HuggingFace Transformers](https://github.com/huggingface/transformers) 模型集成到 vLLM 的高级指南。

**注意：**

添加新模型的复杂性在很大程度上取决于模型的架构。如果模型与 vLLM 中的现有模型具有相似的架构，则该过程相当简单。然而，对于包含新运算符（例如，新的注意力机制）的模型，该过程可能会更复杂一些。


**注意：**


默认情况下，vLLM 模型不支持多模态输入。要启用多模式支持，请在此处实现模型后遵循 `本指南 <enabling_multimodal_inputs>` 。


**提示：**

如果您在将模型集成到 vLLM 时遇到问题，请随时在我们的 [GitHub](https://github.com/vllm-project/vllm/issues) 存储库上提出问题。我们很乐意为您提供帮助！


## 0. Fork vLLM 存储库

首先 fork 我们的 [GitHub](https://github.com/vllm-project/vllm/issues) 存储库，然后[从源代码构建](https://docs.vllm.ai/en/latest/getting_started/installation.html#build-from-source)。这使您能够修改代码库并测试您的模型。


**提示：**

如果您不想 fork 存储库并修改 vLLM 的代码库，请参阅下面的「树外模型集成」部分。


## 1. 引入你的模型代码

从 HuggingFace Transformers 存储库克隆 PyTorch 模型代码，并将其放入 [vllm/model_executor/models](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models) 目录中。例如，vLLM 的 [OPT 模型](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/opt.py) 就是从 HuggingFace 的 [modeling_opt .py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py) 文件中改编而来的。


**警告**

复制模型代码时，请务必查看并遵守代码的版权和许可条款。


## 2. 重写 `forward` 的方法

接下来，您需要按照以下步骤重写模型的 `forward()` 方法：


1. 删除任何不必要的代码，例如仅用于训练的代码。

2. 更改输入参数: 

```diff
def forward(
    self,
    input_ids: torch.Tensor,
-     attention_mask: Optional[torch.Tensor] = None,
-     position_ids: Optional[torch.LongTensor] = None,
-     past_key_values: Optional[List[torch.FloatTensor]] = None,
-     inputs_embeds: Optional[torch.FloatTensor] = None,
-     labels: Optional[torch.LongTensor] = None,
-     use_cache: Optional[bool] = None,
-     output_attentions: Optional[bool] = None,
-     output_hidden_states: Optional[bool] = None,
-     return_dict: Optional[bool] = None,
- ) -> Union[Tuple, CausalLMOutputWithPast]:
+     positions: torch.Tensor,
+     kv_caches: List[torch.Tensor],
+     attn_metadata: AttentionMetadata,
+ ) -> Optional[SamplerOutput]:
```


1. 更新代码，考虑到 `input_ids` 和 `positions` 现在是扁平化的张量。

2. 根据模型的架构，用 `PagedAttention`、`PagedAttentionWithRoPE` 或 `PagedAttentionWithALiBi` 替换原有的注意力操作。


**注意：**

目前，vLLM 支持基础的多头注意力机制及其带有旋转位置嵌入的变体。如果你的模型使用了不同的注意力机制，则需要在 vLLM 中实现一个新的注意力层。


## 3.（可选）实现张量并行和量化支持

如果您的模型太大而无法在单个GPU上容纳，可以考虑采用张量并行技术来解决这一问题。具体来说，您需要将模型中的线性层和嵌入层替换为支持张量并行的对应版本。对于嵌入层，您只需将 `torch.nn.Embedding` 替换为 `VocabParallelEmbedding` 即可。对于输出的语言模型头部，您可以使用 `ParallelLMHead` 。对于线性层，我们提供以下选项来并行化它们：


* `ReplicatedLinear`：跨多个 GPU 复制输入和权重。此方法不节省内存。

* `RowParallelLinear`：将输入张量在隐藏维度上进行划分，而权重矩阵则沿行（输入维度）进行划分。在完成矩阵乘法之后，通过执行 all-reduce 操作来减少结果。通常用于第二个 FFN 层和注意力层的输出线性变换。

* `ColumnParallelLinear`：复制输入张量。权重矩阵则按照列（输出维度）进行分割，而计算结果也沿列维度进行分割。这种技术通常用于原始 Transformer 中的第一个 FFN 层和注意力层的分离 QKV 变换。

* `MergedColumnParallelLinear`：合并多个 *ColumnParallelLinear* 运算符的列并行线性。通常用于具有加权激活函数（例如 SiLU）的第一个 FFN 层。该类处理了多个权重矩阵的分片权重加载逻辑。

* `QKVParallelLinear`: 用于多头和分组查询注意机制的查询、键和值投影的并行线性层。当键/值头的数量小于世界大小时，此类会正确复制键/值头。此类负责处理权重矩阵的权重加载和复制。


请注意，上面的所有线性层均采用 [linear_method] 作为输入。vLLM 会根据不同的量化方案设置该参数，以支持权重量化。


1. 实现权重加载逻辑

您现在需要在 `*ForCausalLM` 类中实现 `load_weights` 方法。此方法应该从 HuggingFace 的检查点文件中加载权重，并将它们分配给模型中的相应层。具体来说，对于  `MergedColumnParallelLinear` 和 `QKVParallelLinear` 层，如果原始模型具有分离的权重矩阵，则需要分别加载不同的部分。


## 5. 注册模型

最后，将你的`*ForCausalLM`类注册到 [vllm/model_executor/models/__init__.py](https://github.com/vllm-project/vllm/blob/) 的`_MODELS`中。


## 6. 树外模型集成

我们还提供了一种无需修改 vLLM 代码库即可集成模型的方法。步骤 2、3、4 仍然是必需的，但您可以跳过步骤 1 和 5。


只需在代码中添加以下行：

```python
from vllm import ModelRegistry
from your_code import YourModelForCausalLM
ModelRegistry.register_model("YourModelForCausalLM", YourModelForCausalLM)
```


如果您的模型导入了初始化 CUDA 的模块，建议您改为延迟加载这些模块，以避免出现类似`RuntimeError: Cannot re-initialize CUDA in forked subprocess` 的错误。

```plain
from vllm import ModelRegistry


ModelRegistry.register_model("YourModelForCausalLM", "your_code:YourModelForCausalLM")


```


**重要**

如果您的模型是多模态模型，请确保模型类实现了 `SupportsMultiModal` 接口。

更多信息可点击[此处](https://docs.vllm.ai/en/latest/models/enabling_multimodal_inputs.html#enabling-multimodal-inputs)查阅。


如果您使用 `vllmserve <args>` 运行 API 服务器，则可以使用以下代码包装入口点：

```python
from vllm import ModelRegistry
from your_code import YourModelForCausalLM
ModelRegistry.register_model("YourModelForCausalLM", YourModelForCausalLM)
import runpy
runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')
```
将上述代码保存在文件中并使用 `python your_file.py <args>` 运行它。
