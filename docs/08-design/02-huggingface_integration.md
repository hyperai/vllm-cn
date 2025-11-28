---
title: 与 HuggingFace 集成
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

本文档描述了 vLLM 如何与 HuggingFace 库集成。我们将逐步解释在运行 `vllm serve` 时，后台会发生什么。

假设我们希望通过运行 `vllm serve Qwen/Qwen2-7B` 来提供流行的 QWen 模型。

1. `model` 参数是 `Qwen/Qwen2-7B`。vLLM 通过检查相应的配置文件 `config.json` 来确定该模型是否存在。实现细节请参见此[代码片段](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L162-L182)。在此过程中：

   1. 如果 `model` 参数对应于现有的本地路径，vLLM 将直接从该路径加载配置文件。

   2. 如果 `model` 参数是由用户名和模型名称组成的 HuggingFace 模型 ID，vLLM 将首先尝试使用 HuggingFace 本地缓存中的配置文件，使用 `model` 参数作为模型名称，`--revision` 参数作为版本。有关 HuggingFace 缓存如何工作的更多信息，请参阅[他们的网站](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome)。

   3. 如果 `model` 参数是 HuggingFace 模型 ID，但在缓存中未找到，vLLM 将从 HuggingFace 模型中心下载配置文件。实现细节请参见[此函数](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L91)。输入参数包括 `model` 参数作为模型名称，`--revision` 参数作为版本，环境变量 `HF_TOKEN` 作为访问模型中心的令牌。在我们的例子中，vLLM 将下载 [config.json](https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json) 文件。

2. 确认模型存在后，vLLM 加载其配置文件并将其转换为字典。实现细节请参见此[代码片段](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L185-L186)。

3. 接下来，vLLM [检查](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L189) 配置字典中的 `model_type` 字段，以[生成](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L190-L216) 要使用的配置对象。vLLM 直接支持一些 `model_type` 值；列表请参见[此处](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L48)。如果 `model_type` 不在列表中，vLLM 将使用 [AutoConfig.from_pretrained](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained) 加载配置类，参数为 `model`、`--revision` 和 `--trust_remote_code`。请注意：

   1. HuggingFace 也有自己的逻辑来确定要使用的配置类。它将再次使用 `model_type` 字段在 transformers 库中搜索类名；支持的模型列表请参见[此处](https://github.com/huggingface/transformers/tree/main/src/transformers/models)。如果未找到 `model_type`，HuggingFace 将使用配置 JSON 文件中的 `auto_map` 字段来确定类名。具体来说，它是 `auto_map` 下的 `AutoConfig` 字段。示例请参见 [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V2.5/blob/main/config.json)。

   2. `auto_map` 下的 `AutoConfig` 字段指向模型仓库中的模块路径。为了创建配置类，HuggingFace 将导入该模块并使用 `from_pretrained` 方法加载配置类。这通常会导致任意代码执行，因此仅在启用 `--trust_remote_code` 时执行。

4. 随后，vLLM 对配置对象应用一些历史补丁。这些补丁主要与 RoPE 配置有关；实现细节请参见[此处](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/transformers_utils/config.py#L244)。

5. 最后，vLLM 可以到达我们要初始化的模型类。vLLM 使用配置对象中的 `architectures` 字段来确定要初始化的模型类，因为它在[其注册表](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/model_executor/models/registry.py#L80) 中维护了从架构名称到模型类的映射。如果注册表中未找到架构名称，则意味着 vLLM 不支持此模型架构。对于 `Qwen/Qwen2-7B`，`architectures` 字段是 `["Qwen2ForCausalLM"]`，对应于 [vLLM 代码](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/model_executor/models/qwen2.py#L364) 中的 `Qwen2ForCausalLM` 类。该类将根据各种配置初始化自身。

除此之外，vLLM 还有两处依赖于 HuggingFace。

1. **Tokenizer**: vLLM 使用 HuggingFace 的 tokenizer 对输入文本进行分词。tokenizer 使用 [AutoTokenizer.from_pretrained](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained) 加载，参数为 `model` 作为模型名称，`--revision` 作为版本。也可以通过指定 `vllm serve` 命令中的 `--tokenizer` 参数来使用另一个模型的 tokenizer。其他相关参数是 `--tokenizer-revision` 和 `--tokenizer-mode`。请参阅 HuggingFace 的文档以了解这些参数的含义。这部分逻辑可以在 [get_tokenizer](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/transformers_utils/tokenizer.py#L87) 函数中找到。获取 tokenizer 后，值得注意的是，vLLM 将在 [get_cached_tokenizer](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/transformers_utils/tokenizer.py#L24) 中缓存 tokenizer 的一些高开销属性。

2. **模型权重**: vLLM 使用 `model` 参数作为模型名称，`--revision` 参数作为版本，从 HuggingFace 模型中心下载模型权重。vLLM 提供了 `--load-format` 参数来控制从模型中心下载哪些文件。默认情况下，它将尝试以 safetensors 格式加载权重，如果 safetensors 格式不可用，则回退到 PyTorch bin 格式。我们还可以传递 `--load-format dummy` 来跳过下载权重。

   1. 建议使用 safetensors 格式，因为它在分布式推理中加载效率高，并且可以避免任意代码执行。有关 safetensors 格式的更多信息，请参阅[文档](https://huggingface.co/docs/safetensors/en/index)。这部分逻辑可以在[此处](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/model_executor/model_loader/loader.py#L385) 找到。请注意：

这就完成了 vLLM 与 HuggingFace 的集成。

总之，vLLM 从 HuggingFace 模型中心或本地目录读取配置文件 `config.json`、tokenizer 和模型权重。它使用来自 vLLM、HuggingFace transformers 的配置类，或从模型仓库加载配置类。
