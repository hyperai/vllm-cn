---
title: 架构概述
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

本文档提供了 vLLM 架构的概述。

**目录**

- [入口点](https://docs.vllm.ai/en/latest/design/arch_overview.html#entrypoints)

  - [LLM 类](https://docs.vllm.ai/en/latest/design/arch_overview.html#llm-class)

  - [OpenAI 兼容的 API 服务器](https://docs.vllm.ai/en/latest/design/arch_overview.html#openai-compatible-api-server)

- [LLM 引擎](https://docs.vllm.ai/en/latest/design/arch_overview.html#llm-engine)

  - [LLMEngine](https://docs.vllm.ai/en/latest/design/arch_overview.html#llmengine)

  - [AsyncLLMEngine](https://docs.vllm.ai/en/latest/design/arch_overview.html#asyncllmengine)

- [Worker](https://docs.vllm.ai/en/latest/design/arch_overview.html#worker)

- [模型运行器](https://docs.vllm.ai/en/latest/design/arch_overview.html#model-runner)

- [模型](https://docs.vllm.ai/en/latest/design/arch_overview.html#model)

- [类层次结构](https://docs.vllm.ai/en/latest/design/arch_overview.html#class-hierarchy)

## [入口点](https://docs.vllm.ai/en/latest/design/arch_overview.html#id1)

vLLM 提供了多个与系统交互的入口点。下图展示了它们之间的关系。

![图片](/img/docs/v1-design/01-arch_overview_1.png)

### [LLM 类](https://docs.vllm.ai/en/latest/design/arch_overview.html#id2)

LLM 类提供了主要的 Python 接口，用于进行离线推理，即在不使用单独的模型推理服务器的情况下与模型交互。

以下是 `LLM` 类的使用示例：

```plain
from vllm import LLM, SamplingParams


# Define a list of input prompts
# 定义输入提示列表
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The largest ocean is",
]


# Define sampling parameters
# 定义采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


# Initialize the LLM engine with the OPT-125M model
# 使用 OPT-125M 模型初始化 LLM 引擎
llm = LLM(model="facebook/opt-125m")


# Generate outputs for the input prompts
# 为输入提示生成输出
outputs = llm.generate(prompts, sampling_params)


# Print the generated outputs
# 打印生成的输出
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

更多 API 详细信息可以在 API 文档的[离线推理](https://docs.vllm.ai/en/latest/api/offline_inference/index.html)部分找到。

`LLM` 类的代码可以在 [vllm/entrypoints/llm.py](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py) 中找到。

### [OpenAI 兼容 API 服务器](https://docs.vllm.ai/en/latest/design/arch_overview.html#id3)

vLLM 的第二个主要接口是通过其 OpenAI 兼容的 API 服务器。可以使用 `vllm serve` 命令启动此服务器。

```plain
vllm serve <model>
```

`vllm` CLI 的代码可以在 [vllm/entrypoints/cli/main.py](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/cli/main.py) 中找到。

有时你可能会看到直接使用 API 服务器入口点，而不是通过 `vllm` CLI 命令使用。例如：

```plain
python -m vllm.entrypoints.openai.api_server --model <model>
```

有关 API 服务器的更多详细信息可以在 [OpenAI 兼容服务器](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#openai-compatible-server)文档中找到。

## [LLM 引擎](https://docs.vllm.ai/en/latest/design/arch_overview.html#id4)

`LLMEngine` 和 `AsyncLLMEngine` 类是 vLLM 系统的核心，负责模型推理和异步请求处理。

![图片](/img/docs/v1-design/01-arch_overview_2.png)
[LLMEngine](https://docs.vllm.ai/en/latest/design/arch_overview.html#id5)

`LLMEngine` 类是 vLLM 引擎的核心组件。它负责接收来自客户端的请求并生成模型的输出。`LLMEngine` 包括输入处理、模型执行（可能分布在多个主机和/或 GPU 上）、调度和输出处理。

- **输入处理**：使用指定的分词器处理输入文本的分词。

- **调度**：选择在每个步骤中处理的请求。

- **模型执行**：管理语言模型的执行，包括跨多个 GPU 的分布式执行。

- **输出处理**：处理模型生成的输出，将语言模型的 token ID 解码为人类可读的文本。

`LLMEngine` 的代码可以在 [vllm/engine/llm_engine.py](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py) 中找到。

### [AsyncLLMEngine](https://docs.vllm.ai/en/latest/design/arch_overview.html#id6)

`AsyncLLMEngine` 类是 `LLMEngine` 类的异步封装。它使用 `asyncio` 创建一个后台循环，持续处理传入的请求。`AsyncLLMEngine` 专为在线服务设计，可以处理多个并发请求并将输出流式传输给客户端。

OpenAI 兼容的 API 服务器使用 `AsyncLLMEngine`。还有一个演示 API 服务器作为更简单的示例，代码位于 [vllm/entrypoints/api_server.py](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/api_server.py)。

`AsyncLLMEngine` 的代码可以在 [vllm/engine/async_llm_engine.py](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py) 中找到。

## [Worker](https://docs.vllm.ai/en/latest/design/arch_overview.html#id7)

Worker 是运行模型推理的进程。vLLM 遵循常见的做法，即使用一个进程来控制一个加速器设备，例如 GPU。例如，如果我们使用大小为 2 的张量并行和大小为 2 的流水线并行，我们将总共有 4 个 Worker。Worker 通过它们的 `rank` 和 `local_rank` 来标识。`rank` 用于全局协调，而 `local_rank` 主要用于分配加速器设备和访问本地资源，例如文件系统和共享内存。

## [模型运行器](https://docs.vllm.ai/en/latest/design/arch_overview.html#id8)

每个 Worker 都有一个模型运行器对象，负责加载和运行模型。许多模型执行的逻辑都在这里，例如准备输入张量和捕获 cudagraphs。

## [模型](https://docs.vllm.ai/en/latest/design/arch_overview.html#id9)

每个模型运行器对象都有一个模型对象，它是实际的 `torch.nn.Module` 实例。有关各种配置如何影响我们最终获得的类，请参阅 [huggingface_integration](https://docs.vllm.ai/en/latest/design/huggingface_integration.html#huggingface-integration)。

## [类层次结构](https://docs.vllm.ai/en/latest/design/arch_overview.html#id10)

下图展示了 vLLM 的类层次结构：

> ![图片](/img/docs/v1-design/01-arch_overview_3.png)

这个类层次结构背后有几个重要的设计选择：

1. **可扩展性**：层次结构中的所有类都接受一个包含所有必要信息的配置对象。[VllmConfig](https://github.com/vllm-project/vllm/blob/d1c6799b8870e513bf4f2305cbf6cda9fc3d773b/vllm/config.py#L2036) 类是主要的配置对象，它在各个类之间传递。类层次结构相当深，每个类都需要读取它感兴趣的配置。通过将所有配置封装在一个对象中，我们可以轻松地传递配置对象并访问所需的配置。假设我们想添加一个新功能（鉴于 LLM 推理领域的快速发展，这种情况经常发生），该功能仅涉及模型运行器。我们只需要在 `VllmConfig` 类中添加一个新的配置选项。由于我们传递了整个配置对象，模型运行器可以直接访问它。我们不需要更改引擎、Worker 或模型类的构造函数来传递新的配置选项。

1. **统一性**：模型运行器需要一个统一的接口来创建和初始化模型。vLLM 支持超过 50 种流行的开源模型。每个模型都有自己的初始化逻辑。如果构造函数签名因模型而异，模型运行器将不知道如何相应地调用构造函数，而没有复杂且容易出错的检查逻辑。通过使模型类的构造函数统一，模型运行器可以轻松创建和初始化模型，而无需知道具体的模型类型。这对于组合模型也很有用。视觉语言模型通常由视觉模型和语言模型组成。通过使构造函数统一，我们可以轻松创建视觉模型和语言模型，并将它们组合成视觉语言模型。

> **注意**
> 为了支持这一更改，所有 vLLM 模型的签名已更新为：
> def **init**(self, \*, vllm_config: VllmConfig, prefix: str = ""):
>
> 为了避免意外传递错误的参数，构造函数现在是仅关键字参数。这确保了如果传递了旧的配置，构造函数将引发错误。vLLM 开发者已经为 vLLM 中的所有模型进行了此更改。对于树外注册的模型，开发者需要更新他们的模型，例如通过添加适配代码将旧的构造函数签名适配到新的签名：
> class MyOldModel(nn.Module):
> def **init**(
> self,
> config,
> cache_config: Optional[CacheConfig] = None,
> quant_config: Optional[QuantizationConfig] = None,
> lora_config: Optional[LoRAConfig] = None,
> prefix: str = "",
> ) -> None:
> ...
>
> from vllm.config import VllmConfig
> class MyNewModel(MyOldModel):
> def **init**(self, \*, vllm_config: VllmConfig, prefix: str = ""):
> config = vllm_config.model_config.hf_config
> cache_config = vllm_config.cache_config
> quant_config = vllm_config.quant_config
> lora_config = vllm_config.lora_config
> super().**init**(config, cache_config, quant_config, lora_config, prefix)
>
> if **version** >= "0.6.4":
> MyModel = MyNewModel
> else:
> MyModel = MyOldModel
>
> 这样，模型可以与新旧版本的 vLLM 一起使用。

1. **初始化的分片和量化**：某些功能需要更改模型权重。例如，张量并行需要对模型权重进行分片，量化需要对模型权重进行量化。有两种可能的方法来实现此功能。一种方法是在模型初始化后更改模型权重。另一种方法是在模型初始化期间更改模型权重。vLLM 选择后者。第一种方法对于大型模型不可扩展。假设我们想在 16 个 H100 80GB GPU 上运行一个 405B 模型（大约 810GB 权重）。理想情况下，每个 GPU 应该只加载 50GB 权重。如果我们在模型初始化后更改模型权重，我们需要将完整的 810GB 权重加载到每个 GPU 上，然后对权重进行分片，导致巨大的内存开销。相反，如果我们在模型初始化期间对权重进行分片，每一层只会创建它需要的权重分片，从而大大减少内存开销。同样的想法适用于量化。注意，我们还向模型的构造函数添加了一个额外的参数 `prefix`，以便模型可以根据前缀以不同的方式初始化自己。这对于非均匀量化很有用，其中模型的不同部分以不同的方式量化。`prefix` 通常是顶级模型的空字符串，子模型的字符串如 `"vision"` 或 `"language"`。通常，它与检查点文件中模块的状态字典的名称匹配。

这种设计的一个缺点是很难为 vLLM 中的各个组件编写单元测试，因为每个组件都需要由完整的配置对象初始化。我们通过提供一个默认的初始化函数来解决这个问题，该函数创建一个所有字段都设置为 `None` 的默认配置对象。如果我们想要测试的组件只关心配置对象中的几个字段，我们可以创建一个默认配置对象并设置我们关心的字段。这样，我们可以隔离测试组件。注意，vLLM 中的许多测试是端到端测试，测试整个系统，所以这不是一个大问题。

总之，完整的配置对象 `VllmConfig` 可以被视为引擎级别的全局状态，在所有 vLLM 类之间共享。
