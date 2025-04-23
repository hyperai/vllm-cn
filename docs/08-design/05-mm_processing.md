---

title: 多模态数据处理

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


# Multi-Modal Data Processing

为了实现 vLLM 中的各种优化，例如[分块预填充](https://docs.vllm.ai/en/latest/performance/optimization.html#chunked-prefill)和[前缀缓存](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html#automatic-prefix-caching)，我们使用 `BaseMultiModalProcessor` 来提供占位符特征 token（例如 `<image>`）与多模态输入（例如原始输入图像）之间的对应关系，基于 HF 处理器的输出。


以下是 `BaseMultiModalProcessor` 的主要功能：


## 提示更新检测

HF 处理器的主要职责之一是用占位符 token 更新提示。例如：

* 在字符串的开头插入特征占位符 token（例如 `<image><image>...<image>`，其数量等于特征大小）。

* 将现有的输入占位符 token（例如单个图像的 `<image>`）替换为特征占位符 token（例如 `<image><image>...<image>`，其数量等于特征大小）。

关于哪些 token 已被更新的信息是找到占位符特征 token 与多模态输入之间对应关系的关键。


在 vLLM 中，此信息通过 `_get_prompt_updates()` 中的 `PromptUpdate` 指定。我们可以通过检查更新后的 token 是否存在来自动检测 HF 是否更新了提示。


## 分词后的提示输入

为了支持在单独进程中进行分词，我们支持将输入 token ID 与多模态数据一起传递。


### 问题

考虑 HF 处理器遵循以下主要步骤：

1. 标记文本

2. 处理多模态输入

3. 执行提示更新


我们要求：

* 对于文本 + 多模态输入，应用所有步骤 1–3。

* 对于已分词的 + 多模态输入，仅应用步骤 2–3。


如何在不重写 HF 处理器的情况下实现这一点？我们可以尝试在不同输入上多次调用 HF 处理器：

* 对于文本 + 多模态输入，直接调用 HF 处理器。

* 对于已分词的 + 多模态输入，仅对多模态输入调用处理器。


虽然 HF 处理器原生支持文本 + 多模态输入，但对于已分词的 + 多模态输入则不然：如果输入占位符 token 的数量与多模态输入的数量不对应，则会抛出错误。


此外，由于已分词的文本未通过 HF 处理器，我们必须自己应用步骤 3，以保持输出 token 和多模态数据之间的一致性。


### 虚拟文本

我们通过要求每个模型定义如何根据多模态输入的数量生成虚拟文本来解决第一个问题，通过 `get_dummy_processor_inputs()` 实现。这使我们能够生成与多模态输入对应的虚拟文本，并将它们一起输入以获得处理后的多模态数据。


### 自动提示更新

我们通过在 `_apply_prompt_updates()` 中实现与模型无关的代码来解决第二个问题，以根据 `_get_prompt_updates()` 输出的规范自动更新提示中的特征占位符 token。


### 总结

借助虚拟文本和自动提示更新，我们的多模态处理器最终可以接受带有多模态数据的文本和 token 提示。详细逻辑如 `_apply_hf_processor_main()` 所示。


## 处理器输出缓存

一些 HF 处理器，例如 Qwen2-VL 的处理器，[非常慢](https://github.com/vllm-project/vllm/issues/9238#)。为了缓解这个问题，我们缓存 HF 处理器的多模态输出，以避免再次处理相同的多模态输入（例如图像）。


当传入新数据时，我们首先检查哪些项在缓存中，哪些项缺失。缺失的项会被批量传递给 HF 处理器并缓存，然后与缓存中的现有项合并。


由于我们只处理缺失的多模态数据项，输入占位符 token 的数量不再与多模态输入的数量对应，因此它们不能与文本提示一起传递给 HF 处理器。因此，我们分别处理文本和多模态输入，使用[虚拟文本](https://docs.vllm.ai/en/latest/design/mm_processing.html#mm-dummy-text)来避免 HF 错误。由于这跳过了 HF 的提示更新代码，我们在之后应用[自动提示更新](https://docs.vllm.ai/en/latest/design/mm_processing.html#mm-automatic-prompt-updating)，以保持输出 token 和多模态数据之间的一致性。

