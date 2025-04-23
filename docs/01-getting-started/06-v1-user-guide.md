---

title: vLLM V1 用户指南

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


V1 现已默认启用所有支持的使用场景，我们将逐步为计划支持的每个场景启用该版本。请在 [GitHub](https://github.com/vllm-project/vllm) 或 [vLLM Slack](https://inviter.co/vllm-slack) 分享反馈。


如需禁用 V1，请设置环境变量：`VLLM_USE_V1=0`，并向我们提交 GitHub Issue 说明原因！

## 

## 为什么选择 vLLM V1？

vLLM V0 成功支持了广泛的模型和硬件，但随着新功能的独立开发，系统复杂性逐渐增加。这种复杂性使得集成新功能更加困难，并积累了技术债务，揭示了对更精简统一设计的需求。


在 V0 成功的基础上，vLLM V1 保留了 V0 中稳定且经过验证的组件（如模型、GPU 内核和实用工具）。同时，它显著重构了核心系统（包括调度器、KV 缓存管理器、工作器、采样器和 API 服务器），以提供一个更易于维护的框架，更好地支持持续增长和创新。


具体而言，V1 旨在：

* 提供**简单、模块化且易于修改的代码库**。

* 通过接近零的 CPU 开销确保**高性能**。

* 将**关键优化整合到统一架构**中。

* 通过默认启用功能/优化实现**零配置**。


从升级到 V1 核心引擎中我们观察到显著的性能提升，尤其是在长上下文场景中。请参阅性能基准测试（待补充）。更多细节请查看 vLLM V1 博客文章 [vLLM V1：核心架构的重大升级](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)（发布于 2025 年 1 月 27 日）。


本动态用户指南概述了 vLLM V1 引入的若干 **重要变更与限制**。团队正积极推动 V1 成为默认引擎，因此随着更多功能在 V1 上获得支持，本指南将持续更新。

### 

### 支持概览

#### 硬件

|硬件|状态|
|:----|:----|
|**NVIDIA**|🚀 原生支持|
|**AMD**|🚧 开发中|
|**TPU**|🚧 开发中|


#### 功能/模型

|功能/模型|状态|
|:----|:----|
|**前缀缓存**|🚀 已优化|
|**分块预填充**|🚀 已优化|
|**logprobs 计算**|🟢 功能正常|
|**LoRA**|🟢 功能正常（[PR #13096](https://github.com/vllm-project/vllm/pull/13096)）|
|**多模态模型**|🟢 功能正常|
|**FP8 KV 缓存**|🟢 Hopper 设备上功能正常（[PR #15191](https://github.com/vllm-project/vllm/pull/15191)）|
|**推测解码**|🚧 开发中（[PR #13933](https://github.com/vllm-project/vllm/pull/13933)）|
|**带前缀缓存的提示 logprobs**|🟡 计划中（[RFC #13414](https://github.com/vllm-project/vllm/issues/13414)）|
|**结构化输出替代后端**|🟡 计划中|
|**嵌入模型**|🟡 计划中（[RFC #12249](https://github.com/vllm-project/vllm/issues/12249)）|
|**Mamba 模型**|🟡 计划中|
|**编码器-解码器模型**|🟡 计划中|
|**请求级结构化输出后端**|🔴 已弃用|
|**best_of**|🔴 已弃用（[RFC #13361](https://github.com/vllm-project/vllm/issues/13361)）|
|**逐请求 logits 处理器**|🔴 已弃用（[RFC #13360](https://github.com/vllm-project/vllm/pull/13360)）|
|**GPU <> CPU KV 缓存交换**|🔴 已弃用|


* **🚀 已优化**：接近完全优化，当前无进一步计划。

* **🟢 功能正常**：完全可用，持续优化中。

* **🚧 开发中**：积极开发中。

* **🟡 计划中**：计划未来实现（部分可能有开放 PR/RFC）。

* **🔴 已弃用**：除非有强烈需求，否则 V1 中不计划支持。


**注意**：vLLM V1 的统一调度器通过使用简单字典（例如 `{request_id: num_tokens}`）动态分配固定 token 预算，将提示和输出 token 同等对待，从而支持分块预填充、前缀缓存和推测解码等功能，无需严格区分预填充和解码阶段。

### 

### 语义变更与弃用功能

#### Logprobs

vLLM V1 支持 logprobs 和提示 logprobs，但与 V0 相比存在重要语义差异：


**Logprobs 计算**

V1 中的 logprobs 现在直接从模型的原始输出计算后立即返回（即在应用任何 logits 后处理如温度缩放或惩罚调整之前）。因此，返回的 logprobs 不反映采样期间使用的最终调整概率。


支持带采样后调整的 logprobs 正在开发中，将在未来更新中添加。


**带前缀缓存的提示 Logprobs**

当前提示 logprobs 仅支持通过 `--no-enable-prefix-caching` 关闭前缀缓存的情况。未来版本中，提示 logprobs 将与前缀缓存兼容，但即使命中前缀缓存，也会触发重新计算以恢复完整提示 logprobs。详见 [RFC #13414](https://github.com/vllm-project/vllm/issues/13414)。

#### 

#### 弃用功能

作为 vLLM V1 架构重大重构的一部分，若干旧功能已被弃用。


**采样功能**

* **best_of**：由于使用率低，此功能已弃用。详见 [RFC #13361](https://github.com/vllm-project/vllm/issues/13361)。

* **逐请求 Logits 处理器**：在 V0 中，用户可传递自定义处理函数来逐请求调整 logits。在 vLLM V1 中，此功能已弃用。设计转向支持 **全局 logits 处理器**，该功能团队正积极开发中。详见 [RFC #13360](https://github.com/vllm-project/vllm/pull/13360)。


**KV 缓存功能**

* **GPU <> CPU KV 缓存交换**：借助新简化的核心架构，vLLM V1 不再需要 KV 缓存交换来处理请求抢占。


**结构化输出功能**

* **请求级结构化输出后端**：已弃用，替代后端（outlines、guidance）支持正在开发中。

### 

### 开发中的功能与模型支持

尽管我们已在 vLLM V1 中重新实现并部分优化了 V0 的许多功能和模型，但部分功能的优化仍在进行中，另一些尚未支持。

#### 

#### 待优化功能

这些功能已在 vLLM V1 中支持，但其优化仍在进行。

* **LoRA**：LoRA 在 V1 中功能正常，但性能低于 V0。团队正积极提升其性能（例如参见 [PR #13096](https://github.com/vllm-project/vllm/pull/13096)）。

* **推测解码**：目前 V1 仅支持基于 ngram 的推测解码。后续将支持其他类型的推测解码（例如参见 [PR #13933](https://github.com/vllm-project/vllm/pull/13933)）。我们将优先支持 Eagle、MTP 而非基于草稿模型的推测解码。

* **多模态模型**：V1 基本兼容 V0，但暂不支持交错模态输入。即将推出的功能与优化状态请参见[此处](https://github.com/orgs/vllm-project/projects/8)。


#### 待支持功能

* **结构化输出替代后端**：计划支持结构化输出替代后端（outlines、guidance）。V1 当前仅支持 `xgrammar:no_fallback` 模式，即若输出模式不受 xgrammar 支持将报错。结构化输出详情请参见[此处](https://docs.vllm.ai/en/latest/features/structured_outputs.html)。

#### 

#### 待支持模型

vLLM V1 当前排除了具有 `SupportsV0Only` 协议的模型架构，主要涵盖以下类别。这些模型的 V1 支持将逐步添加。

**嵌入模型**

不再使用单独的模型运行器，而是基于全局 logits 处理器 [RFC #13360](https://github.com/vllm-project/vllm/pull/13360) 提出了隐藏状态处理器 [RFC #12249](https://github.com/vllm-project/vllm/issues/12249)，以在 V1 中实现同一引擎实例同时生成和嵌入。该功能仍在规划阶段。

**Mamba 模型**

使用选择性状态空间机制（非标准 Transformer 注意力）的模型尚未支持（例如 `MambaForCausalLM`、`JambaForCausalLM`）。

**编码器-解码器模型**

vLLM V1 当前针对仅解码器的 Transformer 优化。需要独立编码器与解码器间交叉注意力的模型尚未支持（例如 `BartForConditionalGeneration`、`MllamaForConditionalGeneration`）。

完整支持模型列表请参见[支持模型列表](https://docs.vllm.ai/en/latest/models/supported_models.html)。

## 

## 常见问题


**使用 vLLM V1 时遇到 CUDA OOM 错误，该如何处理？**


V1 中 `max_num_seqs` 的默认值已从 V0 的 `256` 提升至 `1024`。如果仅在 V1 引擎下遇到 CUDA OOM，请尝试设置更低的 `max_num_seqs` 或 `gpu_memory_utilization`。


另一方面，若收到缓存块内存不足的错误，应增加 `gpu_memory_utilization`，这表明 GPU 内存充足但未为 KV 缓存块分配足够内存。

