---
title: 实现基础模型
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

本指南将带您逐步实现一个基础的 vLLM 模型。

## 1. 引入您的模型代码

首先，从源仓库中克隆 PyTorch 模型代码。例如，vLLM 的 [OPT 模型](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/opt.py) 是从 HuggingFace 的 [modeling_opt.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py) 文件改编而来的。

> **警告**
> 请确保审查并遵守原始代码的版权和许可条款！

## 2. 使您的代码与 vLLM 兼容

为了确保与 vLLM 的兼容性，您的模型必须满足以下要求：

### 初始化代码

模型中的所有 vLLM 模块必须在构造函数中包含一个 `prefix` 参数。这个 `prefix` 通常是模块在模型状态字典中的全名，对于以下情况至关重要：

- 运行时支持：vLLM 的注意力操作符通过其全名注册在模型的状态中。每个注意力操作符必须有一个唯一的 `prefix` 作为其层名，以避免冲突。
- 非均匀量化支持：量化检查点可以选择性地量化某些层，同时保持其他层的全精度。通过在初始化时提供 `prefix`，vLLM 可以将当前层的 `prefix` 与量化配置匹配，以确定是否应以量化模式初始化该层。

初始化代码应如下所示：

```plain
from torch import nn
from vllm.config import VllmConfig
from vllm.attention import Attention


class MyAttention(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.attn = Attention(prefix=f"{prefix}.attn")


class MyDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.self_attn = MyAttention(prefix=f"{prefix}.self_attn")


class MyModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.layers = nn.ModuleList(
            [MyDecoderLayer(vllm_config, prefix=f"{prefix}.layers.{i}") for i in range(vllm_config.model_config.hf_config.num_hidden_layers)]
        )


class MyModelForCausalLM(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model = MyModel(vllm_config, prefix=f"{prefix}.model")
```

### 计算代码

- 在 `MyModel` 模块中添加一个 `get_input_embeddings` 方法，该方法根据 `input_ids` 返回文本嵌入。这相当于直接调用文本嵌入层，但提供了一个统一的接口，以防 `MyModel` 用于复合多模态模型中。

```plain
class MyModel(nn.Module):
        ...


    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...
```

- 重写模型的 `forward()` 方法，删除任何不必要的代码，例如训练专用代码。修改输入参数，将 `input_ids` 和 `positions` 视为具有单一批量大小维度的展平张量，而没有最大序列长度维度。

```plain
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    ...
```

> **注意**
> 目前，vLLM 支持基础的多头注意力机制及其带有旋转位置嵌入的变体。如果您的模型采用不同的注意力机制，您需要在 vLLM 中实现一个新的注意力层。

作为参考，请查看我们的 [Llama 实现](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py)。vLLM 已经支持大量模型。建议找到一个与您的模型相似的模型，并根据您的模型架构进行调整。更多示例请查看 [vllm/model_executor/models](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models)。

## 3. （可选）实现张量并行和量化支持

如果您的模型太大，无法放入单个 GPU，您可以使用张量并行来管理它。为此，请将模型的线性和嵌入层替换为其张量并行版本。对于嵌入层，您可以直接将 `torch.nn.Embedding` 替换为 `VocabParallelEmbedding`。对于输出 LM 头，您可以使用 `ParallelLMHead`。对于线性层，我们提供以下选项来并行化它们：

- `ReplicatedLinear`：在多个 GPU 上复制输入和权重。不节省内存。
- `RowParallelLinear`：输入张量沿隐藏维度分区。权重矩阵沿行（输入维度）分区。矩阵乘法后执行 *all-reduce* 操作以减少结果。通常用于第二个 FFN 层和注意力层的输出线性变换。
- `ColumnParallelLinear`：输入张量被复制。权重矩阵沿列（输出维度）分区。结果沿列维度分区。通常用于第一个 FFN 层和原始 Transformer 中注意力层的分离 QKV 变换。
- `MergedColumnParallelLinear`：合并多个 `ColumnParallelLinear` 操作符的列并行线性层。通常用于带有加权激活函数（例如 SiLU）的第一个 FFN 层。此类处理多个权重矩阵的分片权重加载逻辑。
- `QKVParallelLinear`：用于多头和分组查询注意力机制的查询、键和值投影的并行线性层。当键/值头的数量少于世界大小时，此类会正确复制键/值头。此类处理权重矩阵的加载和复制。

请注意，上述所有线性层都将 `linear_method` 作为输入。vLLM 将根据不同的量化方案设置此参数从而支持权重量化。

## 4. 实现权重加载逻辑

您现在需要在 `*ForCausalLM` 类中实现 `load_weights` 方法。此方法应从 HuggingFace 的检查点文件加载权重并将其分配给模型中的相应层。具体来说，对于 `MergedColumnParallelLinear` 和 `QKVParallelLinear` 层，如果原始模型具有分离的权重矩阵，则需要分别加载不同的部分。

## 5. 注册您的模型

有关如何将新模型注册到 vLLM 的说明，请参阅[此页面](https://docs.vllm.ai/en/latest/contributing/model/registration.html#new-model-registration)。

## 常见问题

### 如何支持具有交错滑动窗口的模型？

对于具有交错滑动窗口的模型（例如 `google/gemma-2-2b-it` 和 `mistralai/Ministral-8B-Instruct-2410`），调度程序会将模型视为全注意力模型，即所有 token 的 kv-cache 都不会被丢弃。这是为了确保前缀缓存与这些模型一起工作。滑动窗口仅作为注意力内核计算的参数出现。

要支持具有交错滑动窗口的模型，我们需要注意以下细节：

- 确保[此行](https://github.com/vllm-project/vllm/blob/996357e4808ca5eab97d4c97c7d25b3073f46aab/vllm/config.py#L308)将 `has_interleaved_attention` 评估为 `True`，并将 `self.hf_text_config.interleaved_sliding_window` 设置为模型可以理解的交错滑动窗口格式。然后，`self.hf_text_config.sliding_window` 将被删除，模型将被视为全注意力模型。
- 在建模代码中，解析每一层的正确滑动窗口值，并将其传递给注意力层的 `per_layer_sliding_window` 参数。作为参考，请查看[此行](https://github.com/vllm-project/vllm/blob/996357e4808ca5eab97d4c97c7d25b3073f46aab/vllm/model_executor/models/llama.py#L171)。

通过这两个步骤，交错滑动窗口应该可以正常工作。
