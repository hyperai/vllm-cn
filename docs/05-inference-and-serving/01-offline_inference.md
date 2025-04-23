---
title: 离线推理
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

您可以在自己的代码中运行 vLLM 来处理一组提示。

离线 API 基于 `LLM` 类。要初始化 vLLM 引擎，请创建一个新的 `LLM` 实例并指定要运行的模型。

例如，以下代码从 HuggingFace 下载 `facebook/opt-125m` 模型，并使用默认配置在 vLLM 中运行它。

```plain
llm = LLM(model="facebook/opt-125m")
```

初始化 `LLM` 实例后，您可以使用各种 API 执行模型推理。可用的 API 取决于运行的模型类型：

- [生成模型](https://docs.vllm.ai/en/latest/models/generative_models.html#generative-models)：输出对数概率，通过采样获得最终输出文本。
- [池化模型](https://docs.vllm.ai/en/latest/models/pooling_models.html#pooling-models)：直接输出其隐藏状态。

有关每个 API 的更多详细信息，请参阅上述页面。

另请参阅：[API 参考](https://docs.vllm.ai/en/latest/api/offline_inference/index.html)

## 配置选项

本节列出了运行 vLLM 引擎时最常见的选项。完整列表请参阅[引擎参数](https://docs.vllm.ai/en/latest/serving/engine_args.html#engine-args)页面。

### 模型解析

vLLM 通过检查模型仓库中 `config.json` 的 `architectures` 字段并找到注册到 vLLM 的相应实现来加载与 HuggingFace 兼容的模型。然而，我们的模型解析可能会因以下原因失败：

- 模型仓库的 `config.json` 缺少 `architectures` 字段。
- 非官方仓库使用未在 vLLM 中记录的替代名称引用模型。
- 相同的架构名称用于多个模型，导致加载哪个模型存在歧义。

要解决此问题，可以通过向 `hf_overrides` 选项传递 `config.json` 的覆盖值来显式指定模型架构。例如：

```plain
model = LLM(
    model="cerebras/Cerebras-GPT-1.3B",
    hf_overrides={"architectures": ["GPT2LMHeadModel"]},  # GPT-2
)
```

[支持模型列表](https://docs.vllm.ai/en/latest/models/supported_models.html#supported-models)显示了 vLLM 识别的模型架构。

### 减少内存使用

大型模型可能会导致您的机器内存不足 (OOM)。以下是一些有助于缓解此问题的选项。

#### 张量并行（TP）

张量并行（`tensor_parallel_size` 选项）可用于将模型拆分到多个 GPU 上。

以下代码将模型拆分到 2 个 GPU 上。

```plain
llm = LLM(model="ibm-granite/granite-3.1-8b-instruct",
          tensor_parallel_size=2)
```

> **重要\*\***提示\*\*
> 为确保 vLLM 正确初始化 CUDA，您应避免在初始化 vLLM 之前调用相关函数（例如 `torch.cuda.set_device()`）。否则，您可能会遇到类似 `RuntimeError: Cannot re-initialize CUDA in forked subprocess` 的错误。
>
> 要控制使用哪些设备，请设置 `CUDA_VISIBLE_DEVICES` 环境变量。

#### 量化

量化模型以降低精度为代价占用更少的内存。

静态量化模型可以从 HF Hub 下载（一些流行的模型可在 [Neural Magic](https://huggingface.co/neuralmagic) 找到），并直接使用而无需额外配置。

动态量化也通过 `quantization` 选项支持——更多详细信息请参阅[此处](https://docs.vllm.ai/en/latest/features/quantization/index.html#quantization-index)。

#### 上下文长度和批量大小

您可以通过限制模型的上下文长度（`max_model_len` 选项）和最大批量大小（`max_num_seqs` 选项）来进一步减少内存使用。

```plain
llm = LLM(model="adept/fuyu-8b",
          max_model_len=2048,
          max_num_seqs=2)
```

#### 调整缓存大小

如果您的 CPU RAM 不足，请尝试以下选项：

- （仅限多模态模型）您可以使用环境变量（默认为 4 GiB）设置多模态输入缓存的大小 `VLLM_MM_INPUT_CACHE_GIB`。
- （仅限 CPU 后端）您可以使用环境变量（默认 4 GiB）`VLLM_CPU_KVCACHE_SPACE` 设置 KV 缓存的大小。

### 性能优化和调优

您可以通过微调各种选项来潜在地提高 vLLM 的性能。更多详细信息请参阅[本指南](https://docs.vllm.ai/en/latest/performance/optimization.html#optimization-and-tuning)。
