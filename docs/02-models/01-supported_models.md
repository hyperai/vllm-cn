---
title: 支持模型列表
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM 支持跨多种任务的生成式和池化模型。若模型支持多个任务，可通过 `--task` 参数指定任务。

针对每个任务，我们列出了 vLLM 中已实现的模型架构。每个架构旁附带了使用该架构的热门模型示例。

## 加载模型

### HuggingFace Hub

默认情况下，vLLM 从 [HuggingFace (HF) Hub](https://huggingface.co/models) 加载模型。

要判断某个模型是否原生支持，可检查 HF 仓库内的 `config.json` 文件。若 `"architectures"` 字段包含以下列出的模型架构，则该模型应被原生支持。

模型**无需**原生支持即可在 vLLM 中使用。[Transformers 回退](https://docs.vllm.ai/en/latest/models/supported_models.html#transformers-fallback) 允许您直接使用模型的 Transformers 实现运行（甚至支持 Hugging Face 模型中心的远程代码）。

**提示**

运行时验证模型是否支持的最简单方法是运行以下程序：

```plain
from vllm import LLM


# For generative models (task=generate) only
# 仅适用于生成式模型（task=generate）
llm = LLM(model=..., task="generate")  # Name or path of your model
output = llm.generate("Hello, my name is")
print(output)


# For pooling models (task={embed,classify,reward,score}) only
# 仅适用于池化模型（task={embed,classify,reward,score}）
llm = LLM(model=..., task="embed")  # Name or path of your model
output = llm.encode("Hello, my name is")
print(output)
```

若 vLLM 成功返回文本（生成式模型）或隐藏状态（池化模型），则表明您的模型受支持。

否则，请参考 [添加新模型](https://docs.vllm.ai/en/latest/contributing/model/index.html#new-model) 了解如何在 vLLM 中实现您的模型。您也可 [在 GitHub 提交 Issue](https://github.com/vllm-project/vllm/issues/new/choose) 请求 vLLM 支持。

### Transformers 回退

vLLM 可回退至 Transformers 中可用的模型实现。目前并非所有模型均支持此功能，但大多数解码器语言模型已支持，视觉语言模型支持正在规划中！

要检查是否使用 Transformers 后端，可执行以下操作：

```plain
from vllm import LLM
llm = LLM(model=..., task="generate")  # Name or path of your model # 模型的名称或路径
llm.apply_model(lambda model: print(type(model)))
```

若输出为 `TransformersForCausalLM`，则表示它基于 Transformers 实现！

**提示**

对于 [离线推理](https://docs.vllm.ai/en/latest/serving/offline_inference.html#offline-inference)，可通过设置 `model_impl="transformers"` 强制使用 `TransformersForCausalLM`；对于 [OpenAI 兼容服务器](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#openai-compatible-server)，可使用 `--model-impl transformers`。

**注意**

vLLM 可能无法完全优化 Transformers 实现，因此若在 vLLM 中比较原生模型与 Transformers 模型，可能会观察到性能下降。

#### 支持功能

Transformers 回退明确支持以下功能：

- [量化](https://docs.vllm.ai/en/latest/features/quantization/index.html#quantization-index)（GGUF 除外）
- [LoRA 适配器](https://docs.vllm.ai/en/latest/features/lora.html#lora-adapter)
- [分布式推理与服务](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#distributed-serving)（需 `transformers>=4.49.0`）

#### 远程代码

前文提到，Transformers 回退允许您直接在 vLLM 中运行远程代码模型。若对此功能感兴趣，本节将为您说明！

只需设置 `trust_remote_code=True`，vLLM 即可运行模型中心中与 Transformers 兼容的任何模型。只要模型开发者以兼容方式实现其模型，您便可在 Transformers 或 vLLM 官方支持前运行新模型！

```plain
from vllm import LLM
llm = LLM(model=..., task="generate", trust_remote_code=True)  # 模型的名称或路径
llm.apply_model(lambda model: print(model.__class__))
```

要使您的模型兼容 Transformers 回退，需满足以下要求：

modeling_my_model.py

```plain
from transformers import PreTrainedModel
from torch import nn


class MyAttention(nn.Module):


  def forward(self, hidden_states, **kwargs): # <- kwargs are required # <- 需包含 kwargs
    ...
    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    attn_output, attn_weights = attention_interface(
      self,
      query_states,
      key_states,
      value_states,
      **kwargs,
    )
    ...


class MyModel(PreTrainedModel):
  _supports_attention_backend = True
```

后台流程如下：

1. 加载配置

2. 从 `auto_map` 加载 `MyModel` Python 类，并验证模型 `_supports_attention_backend`

3. 使用 `TransformersForCausalLM` 后端（参见 [vllm/model_executor/models/transformers.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/transformers.py)），该后端利用 `self.config._attn_implementation = "vllm"`，因此需使用 `ALL_ATTENTION_FUNCTION`

要使模型兼容张量并行，需满足：

configuration_my_model.py

```plain
from transformers import PretrainedConfig


class MyConfig(PretrainedConfig):
  base_model_tp_plan = {
    "layers.*.self_attn.q_proj": "colwise",
    ...
  }
```

**提示**

`base_model_tp_plan` 是一个 `dict`，用于将全限定层名模式映射到张量并行风格（当前仅支持 `"colwise"` 和 `"rowwise"`）。

### ModelScope

要使用 [ModelScope](https://www.modelscope.cn/) 的模型而非 HuggingFace Hub，请设置环境变量：

```plain
export VLLM_USE_MODELSCOPE=True
```

并与 `trust_remote_code=True` 配合使用：

```plain
from vllm import LLM


llm = LLM(model=..., revision=..., task=..., trust_remote_code=True)


# For generative models (task=generate) only
# 仅适用于生成式模型（task=generate）
output = llm.generate("Hello, my name is")
print(output)


# For pooling models (task={embed,classify,reward,score}) only
# 仅适用于池化模型（task={embed,classify,reward,score}）
output = llm.encode("Hello, my name is")
print(output)
```

## 纯文本语言模型列表

### 生成式模型

关于生成式模型的使用详情，请参阅[此页面](https://docs.vllm.ai/en/latest/models/generative_models.html#generative-models)。

#### 文本生成 (`--task generate`)

| 架构                                  | 模型                                                | HF 模型示例                                                                                                                                                        | [LoRA](https://docs.vllm.ai/en/latest/features/lora.html#lora-adapter) | [PP](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#distributed-serving) |
| :------------------------------------ | :-------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------- | :---------------------------------------------------------------------------------------- |
| AquilaForCausalLM                     | Aquila, Aquila2                                     | BAAI/Aquila-7B, BAAI/AquilaChat-7B, etc.                                                                                                                           | ✅︎                                                                    | ✅︎                                                                                       |
| ArcticForCausalLM                     | Arctic                                              | Snowflake/snowflake-arctic-base, Snowflake/snowflake-arctic-instruct, etc.                                                                                         |                                                                        | ✅︎                                                                                       |
| BaiChuanForCausalLM                   | Baichuan2, Baichuan                                 | baichuan-inc/Baichuan2-13B-Chat, baichuan-inc/Baichuan-7B, etc.                                                                                                    | ✅︎                                                                    | ✅︎                                                                                       |
| BloomForCausalLM                      | BLOOM, BLOOMZ, BLOOMChat                            | bigscience/bloom, bigscience/bloomz, etc.                                                                                                                          |                                                                        | ✅︎                                                                                       |
| BartForConditionalGeneration          | BART                                                | facebook/bart-base, facebook/bart-large-cnn, etc.                                                                                                                  |                                                                        |                                                                                           |
| ChatGLMModel                          | ChatGLM                                             | THUDM/chatglm2-6b, THUDM/chatglm3-6b, etc.                                                                                                                         | ✅︎                                                                    | ✅︎                                                                                       |
| CohereForCausalLM, Cohere2ForCausalLM | Command-R                                           | CohereForAI/c4ai-command-r-v01, CohereForAI/c4ai-command-r7b-12-2024, etc.                                                                                         | ✅︎                                                                    | ✅︎                                                                                       |
| DbrxForCausalLM                       | DBRX                                                | databricks/dbrx-base, databricks/dbrx-instruct, etc.                                                                                                               |                                                                        | ✅︎                                                                                       |
| DeciLMForCausalLM                     | DeciLM                                              | Deci/DeciLM-7B, Deci/DeciLM-7B-instruct, etc.                                                                                                                      |                                                                        | ✅︎                                                                                       |
| DeepseekForCausalLM                   | DeepSeek                                            | deepseek-ai/deepseek-llm-67b-base, deepseek-ai/deepseek-llm-7b-chat etc.                                                                                           |                                                                        | ✅︎                                                                                       |
| DeepseekV2ForCausalLM                 | DeepSeek-V2                                         | deepseek-ai/DeepSeek-V2, deepseek-ai/DeepSeek-V2-Chat etc.                                                                                                         |                                                                        | ✅︎                                                                                       |
| DeepseekV3ForCausalLM                 | DeepSeek-V3                                         | deepseek-ai/DeepSeek-V3-Base, deepseek-ai/DeepSeek-V3 etc.                                                                                                         |                                                                        | ✅︎                                                                                       |
| ExaoneForCausalLM                     | EXAONE-3                                            | LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct, etc.                                                                                                                         | ✅︎                                                                    | ✅︎                                                                                       |
| FalconForCausalLM                     | Falcon                                              | tiiuae/falcon-7b, tiiuae/falcon-40b, tiiuae/falcon-rw-7b, etc.                                                                                                     |                                                                        | ✅︎                                                                                       |
| FalconMambaForCausalLM                | FalconMamba                                         | tiiuae/falcon-mamba-7b, tiiuae/falcon-mamba-7b-instruct, etc.                                                                                                      | ✅︎                                                                    | ✅︎                                                                                       |
| GemmaForCausalLM                      | Gemma                                               | google/gemma-2b, google/gemma-7b, etc.                                                                                                                             | ✅︎                                                                    | ✅︎                                                                                       |
| Gemma2ForCausalLM                     | Gemma 2                                             | google/gemma-2-9b, google/gemma-2-27b, etc.                                                                                                                        | ✅︎                                                                    | ✅︎                                                                                       |
| Gemma3ForCausalLM                     | Gemma 3                                             | google/gemma-3-1b-it, etc.                                                                                                                                         | ✅︎                                                                    | ✅︎                                                                                       |
| GlmForCausalLM                        | GLM-4                                               | THUDM/glm-4-9b-chat-hf, etc.                                                                                                                                       | ✅︎                                                                    | ✅︎                                                                                       |
| GPT2LMHeadModel                       | GPT-2                                               | gpt2, gpt2-xl, etc.                                                                                                                                                |                                                                        | ✅︎                                                                                       |
| GPTBigCodeForCausalLM                 | StarCoder, SantaCoder, WizardCoder                  | bigcode/starcoder, bigcode/gpt_bigcode-santacoder, WizardLM/WizardCoder-15B-V1.0, etc.                                                                             | ✅︎                                                                    | ✅︎                                                                                       |
| GPTJForCausalLM                       | GPT-J                                               | EleutherAI/gpt-j-6b, nomic-ai/gpt4all-j, etc.                                                                                                                      |                                                                        | ✅︎                                                                                       |
| GPTNeoXForCausalLM                    | GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM | EleutherAI/gpt-neox-20b, EleutherAI/pythia-12b, OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5, databricks/dolly-v2-12b, stabilityai/stablelm-tuned-alpha-7b, etc. |                                                                        | ✅︎                                                                                       |
| GraniteForCausalLM                    | Granite 3.0, Granite 3.1, PowerLM                   | ibm-granite/granite-3.0-2b-base, ibm-granite/granite-3.1-8b-instruct, ibm/PowerLM-3b, etc.                                                                         | ✅︎                                                                    | ✅︎                                                                                       |
| GraniteMoeForCausalLM                 | Granite 3.0 MoE, PowerMoE                           | ibm-granite/granite-3.0-1b-a400m-base, ibm-granite/granite-3.0-3b-a800m-instruct, ibm/PowerMoE-3b, etc.                                                            | ✅︎                                                                    | ✅︎                                                                                       |
| GraniteMoeSharedForCausalLM           | Granite MoE Shared                                  | ibm-research/moe-7b-1b-active-shared-experts (test model)                                                                                                          | ✅︎                                                                    | ✅︎                                                                                       |
| GritLM                                | GritLM                                              | parasail-ai/GritLM-7B-vllm.                                                                                                                                        | ✅︎                                                                    | ✅︎                                                                                       |
| Grok1ModelForCausalLM                 | Grok1                                               | hpcai-tech/grok-1.                                                                                                                                                 | ✅︎                                                                    | ✅︎                                                                                       |
| InternLMForCausalLM                   | InternLM                                            | internlm/internlm-7b, internlm/internlm-chat-7b, etc.                                                                                                              | ✅︎                                                                    | ✅︎                                                                                       |
| InternLM2ForCausalLM                  | InternLM2                                           | internlm/internlm2-7b, internlm/internlm2-chat-7b, etc.                                                                                                            | ✅︎                                                                    | ✅︎                                                                                       |
| InternLM3ForCausalLM                  | InternLM3                                           | internlm/internlm3-8b-instruct, etc.                                                                                                                               | ✅︎                                                                    | ✅︎                                                                                       |
| JAISLMHeadModel                       | Jais                                                | inceptionai/jais-13b, inceptionai/jais-13b-chat, inceptionai/jais-30b-v3, inceptionai/jais-30b-chat-v3, etc.                                                       |                                                                        | ✅︎                                                                                       |
| JambaForCausalLM                      | Jamba                                               | ai21labs/AI21-Jamba-1.5-Large, ai21labs/AI21-Jamba-1.5-Mini, ai21labs/Jamba-v0.1, etc.                                                                             | ✅︎                                                                    | ✅︎                                                                                       |
| LlamaForCausalLM                      | Llama 3.1, Llama 3, Llama 2, LLaMA, Yi              | meta-llama/Meta-Llama-3.1-405B-Instruct, meta-llama/Meta-Llama-3.1-70B, meta-llama/Meta-Llama-3-70B-Instruct, meta-llama/Llama-2-70b-hf, 01-ai/Yi-34B, etc.        | ✅︎                                                                    | ✅︎                                                                                       |
| MambaForCausalLM                      | Mamba                                               | state-spaces/mamba-130m-hf, state-spaces/mamba-790m-hf, state-spaces/mamba-2.8b-hf, etc.                                                                           |                                                                        | ✅︎                                                                                       |
| MiniCPMForCausalLM                    | MiniCPM                                             | openbmb/MiniCPM-2B-sft-bf16, openbmb/MiniCPM-2B-dpo-bf16, openbmb/MiniCPM-S-1B-sft, etc.                                                                           | ✅︎                                                                    | ✅︎                                                                                       |
| MiniCPM3ForCausalLM                   | MiniCPM3                                            | openbmb/MiniCPM3-4B, etc.                                                                                                                                          | ✅︎                                                                    | ✅︎                                                                                       |
| MistralForCausalLM                    | Mistral, Mistral-Instruct                           | mistralai/Mistral-7B-v0.1, mistralai/Mistral-7B-Instruct-v0.1, etc.                                                                                                | ✅︎                                                                    | ✅︎                                                                                       |
| MixtralForCausalLM                    | Mixtral-8x7B, Mixtral-8x7B-Instruct                 | mistralai/Mixtral-8x7B-v0.1, mistralai/Mixtral-8x7B-Instruct-v0.1, mistral-community/Mixtral-8x22B-v0.1, etc.                                                      | ✅︎                                                                    | ✅︎                                                                                       |
| MPTForCausalLM                        | MPT, MPT-Instruct, MPT-Chat, MPT-StoryWriter        | mosaicml/mpt-7b, mosaicml/mpt-7b-storywriter, mosaicml/mpt-30b, etc.                                                                                               |                                                                        | ✅︎                                                                                       |
| NemotronForCausalLM                   | Nemotron-3, Nemotron-4, Minitron                    | nvidia/Minitron-8B-Base, mgoin/Nemotron-4-340B-Base-hf-FP8, etc.                                                                                                   | ✅︎                                                                    | ✅︎                                                                                       |
| OLMoForCausalLM                       | OLMo                                                | allenai/OLMo-1B-hf, allenai/OLMo-7B-hf, etc.                                                                                                                       |                                                                        | ✅︎                                                                                       |
| OLMo2ForCausalLM                      | OLMo2                                               | allenai/OLMo2-7B-1124, etc.                                                                                                                                        |                                                                        | ✅︎                                                                                       |
| OLMoEForCausalLM                      | OLMoE                                               | allenai/OLMoE-1B-7B-0924, allenai/OLMoE-1B-7B-0924-Instruct, etc.                                                                                                  | ✅︎                                                                    | ✅︎                                                                                       |
| OPTForCausalLM                        | OPT, OPT-IML                                        | facebook/opt-66b, facebook/opt-iml-max-30b, etc.                                                                                                                   |                                                                        | ✅︎                                                                                       |
| OrionForCausalLM                      | Orion                                               | OrionStarAI/Orion-14B-Base, OrionStarAI/Orion-14B-Chat, etc.                                                                                                       |                                                                        | ✅︎                                                                                       |
| PhiForCausalLM                        | Phi                                                 | microsoft/phi-1_5, microsoft/phi-2, etc.                                                                                                                           | ✅︎                                                                    | ✅︎                                                                                       |
| Phi3ForCausalLM                       | Phi-4, Phi-3                                        | microsoft/Phi-4-mini-instruct, microsoft/Phi-4, microsoft/Phi-3-mini-4k-instruct, microsoft/Phi-3-mini-128k-instruct, microsoft/Phi-3-medium-128k-instruct, etc.   | ✅︎                                                                    | ✅︎                                                                                       |
| Phi3SmallForCausalLM                  | Phi-3-Small                                         | microsoft/Phi-3-small-8k-instruct, microsoft/Phi-3-small-128k-instruct, etc.                                                                                       |                                                                        | ✅︎                                                                                       |
| PhiMoEForCausalLM                     | Phi-3.5-MoE                                         | microsoft/Phi-3.5-MoE-instruct, etc.                                                                                                                               | ✅︎                                                                    | ✅︎                                                                                       |
| PersimmonForCausalLM                  | Persimmon                                           | adept/persimmon-8b-base, adept/persimmon-8b-chat, etc.                                                                                                             |                                                                        | ✅︎                                                                                       |
| QWenLMHeadModel                       | Qwen                                                | Qwen/Qwen-7B, Qwen/Qwen-7B-Chat, etc.                                                                                                                              | ✅︎                                                                    | ✅︎                                                                                       |
| Qwen2ForCausalLM                      | QwQ, Qwen2                                          | Qwen/QwQ-32B-Preview, Qwen/Qwen2-7B-Instruct, Qwen/Qwen2-7B, etc.                                                                                                  | ✅︎                                                                    | ✅︎                                                                                       |
| Qwen2MoeForCausalLM                   | Qwen2MoE                                            | Qwen/Qwen1.5-MoE-A2.7B, Qwen/Qwen1.5-MoE-A2.7B-Chat, etc.                                                                                                          |                                                                        | ✅︎                                                                                       |
| StableLmForCausalLM                   | StableLM                                            | stabilityai/stablelm-3b-4e1t, stabilityai/stablelm-base-alpha-7b-v2, etc.                                                                                          |                                                                        | ✅︎                                                                                       |
| Starcoder2ForCausalLM                 | Starcoder2                                          | bigcode/starcoder2-3b, bigcode/starcoder2-7b, bigcode/starcoder2-15b, etc.                                                                                         |                                                                        | ✅︎                                                                                       |
| SolarForCausalLM                      | Solar Pro                                           | upstage/solar-pro-preview-instruct, etc.                                                                                                                           | ✅︎                                                                    | ✅︎                                                                                       |
| TeleChat2ForCausalLM                  | TeleChat2                                           | Tele-AI/TeleChat2-3B, Tele-AI/TeleChat2-7B, Tele-AI/TeleChat2-35B, etc.                                                                                            | ✅︎                                                                    | ✅︎                                                                                       |
| TeleFLMForCausalLM                    | TeleFLM                                             | CofeAI/FLM-2-52B-Instruct-2407, CofeAI/Tele-FLM, etc.                                                                                                              | ✅︎                                                                    | ✅︎                                                                                       |
| XverseForCausalLM                     | XVERSE                                              | xverse/XVERSE-7B-Chat, xverse/XVERSE-13B-Chat, xverse/XVERSE-65B-Chat, etc.                                                                                        | ✅︎                                                                    | ✅︎                                                                                       |
| Zamba2ForCausalLM                     | Zamba2                                              | Zyphra/Zamba2-7B-instruct, Zyphra/Zamba2-2.7B-instruct, Zyphra/Zamba2-1.2B-instruct, etc.                                                                          |                                                                        |                                                                                           |

**注意**

当前 ROCm 版本的 vLLM 仅支持 Mistral 和 Mixtral 模型，且上下文长度上限为 4096。

### 池化模型

关于池化模型的使用详情，请参阅[此页面](https://docs.vllm.ai/en/latest/models/pooling_models.html#pooling-models)。

> **重要**
> 由于部分模型架构同时支持生成式和池化任务，您应显式指定任务类型以确保模型以池化模式而非生成模式运行。

#### 文本嵌入 (`--task embed`)

| 架构                                             | 模型              | HF 模型示例                                                                                        | [LoRA](https://docs.vllm.ai/en/latest/features/lora.html#lora-adapter) | [PP](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#distributed-serving) |
| :----------------------------------------------- | :---------------- | :------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------- | :---------------------------------------------------------------------------------------- |
| BertModel                                        | BERT-based        | BAAI/bge-base-en-v1.5, etc.                                                                        |                                                                        |                                                                                           |
| Gemma2Model                                      | Gemma 2-based     | BAAI/bge-multilingual-gemma2, etc.                                                                 |                                                                        | ✅︎                                                                                       |
| GritLM                                           | GritLM            | parasail-ai/GritLM-7B-vllm.                                                                        | ✅︎                                                                    | ✅︎                                                                                       |
| LlamaModel, LlamaForCausalLM, MistralModel, etc. | Llama-based       | intfloat/e5-mistral-7b-instruct, etc.                                                              | ✅︎                                                                    | ✅︎                                                                                       |
| Qwen2Model, Qwen2ForCausalLM                     | Qwen2-based       | ssmits/Qwen2-7B-Instruct-embed-base (see note), Alibaba-NLP/gte-Qwen2-7B-instruct (see note), etc. | ✅︎                                                                    | ✅︎                                                                                       |
| RobertaModel, RobertaForMaskedLM                 | RoBERTa-based     | sentence-transformers/all-roberta-large-v1, sentence-transformers/all-roberta-large-v1, etc.       |                                                                        |                                                                                           |
| XLMRobertaModel                                  | XLM-RoBERTa-based | intfloat/multilingual-e5-large, etc.                                                               |                                                                        |                                                                                           |

> **注意** >`ssmits/Qwen2-7B-Instruct-embed-base` 的 Sentence Transformers 配置定义有误。您需手动设置均值池化：传递 `--override-pooler-config '{"pooling_type": "MEAN"}'`。

> **注意** >`Alibaba-NLP/gte-Qwen2-1.5B-instruct` 的 HF 实现强制使用因果注意力（与 `config.json` 所示不符）。为比较 vLLM 与 HF 结果，应在 vLLM 中设置 `--hf-overrides '{"is_causal": true}'` 以确保两者实现一致。
>
> 对于 1.5B 和 7B 版本，还需启用 `--trust-remote-code` 以加载正确的分词器。详见 [HF Transformers 相关 Issue](https://github.com/huggingface/transformers/issues/34882)。

若您的模型未列于上方，我们将尝试通过 `as_embedding_model()` 自动转换模型。默认情况下，从最后一个 token 对应的归一化隐藏状态提取完整提示的嵌入。

#### 奖励建模 (`--task reward`)

| 架构                       | 模型            | HF 模型示例                                                        | [LoRA](https://docs.vllm.ai/en/latest/features/lora.html#lora-adapter) | [PP](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#distributed-serving) |
| :------------------------- | :-------------- | :----------------------------------------------------------------- | :--------------------------------------------------------------------- | :---------------------------------------------------------------------------------------- |
| InternLM2ForRewardModel    | InternLM2-based | internlm/internlm2-1_8b-reward, internlm/internlm2-7b-reward, etc. | ✅︎                                                                    | ✅︎                                                                                       |
| LlamaForCausalLM           | Llama-based     | peiyi9979/math-shepherd-mistral-7b-prm, etc.                       | ✅︎                                                                    | ✅︎                                                                                       |
| Qwen2ForRewardModel        | Qwen2-based     | Qwen/Qwen2.5-Math-RM-72B, etc.                                     | ✅︎                                                                    | ✅︎                                                                                       |
| Qwen2ForProcessRewardModel | Qwen2-based     | Qwen/Qwen2.5-Math-PRM-7B, Qwen/Qwen2.5-Math-PRM-72B, etc.          | ✅︎                                                                    | ✅︎                                                                                       |

若您的模型不在上述列表中，我们将尝试通过 `as_reward_model()` 自动转换模型。默认直接返回每个 token 的隐藏状态。

> **重要\*\***信息\*\*
> 对于过程监督奖励模型（如 `peiyi9979/math-shepherd-mistral-7b-prm`），需显式设置池化配置，例如：
> `--override-pooler-config '{"pooling_type": "STEP", "step_tag_id": 123, "returned_token_ids": [456, 789]}'`

#### 分类 (`--task classify`)

| 架构                           | 建模        | HF 模型示例                          | [LoRA](https://docs.vllm.ai/en/latest/features/lora.html#lora-adapter) | [PP](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#distributed-serving) |
| :----------------------------- | :---------- | :----------------------------------- | :--------------------------------------------------------------------- | :---------------------------------------------------------------------------------------- |
| JambaForSequenceClassification | Jamba       | ai21labs/Jamba-tiny-reward-dev, etc. | ✅︎                                                                    | ✅︎                                                                                       |
| Qwen2ForSequenceClassification | Qwen2-based | jason9693/Qwen2.5-1.5B-apeach, etc.  | ✅︎                                                                    | ✅︎                                                                                       |

若您的模型不在上述列表中，我们将尝试通过 `as_classification_model()` 自动转换模型。默认从最后一个 token 对应的 softmax 化隐藏状态提取类别概率。

#### 句子对评分 (`--task score`)

| 架构                                | 建模              | HF 模型示例                                | [LoRA](https://docs.vllm.ai/en/latest/features/lora.html#lora-adapter) | [PP](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#distributed-serving) |
| :---------------------------------- | :---------------- | :----------------------------------------- | :--------------------------------------------------------------------- | :---------------------------------------------------------------------------------------- |
| BertForSequenceClassification       | BERT-based        | cross-encoder/ms-marco-MiniLM-L-6-v2, etc. |                                                                        |                                                                                           |
| RobertaForSequenceClassification    | RoBERTa-based     | cross-encoder/quora-roberta-base, etc.     |                                                                        |                                                                                           |
| XLMRobertaForSequenceClassification | XLM-RoBERTa-based | BAAI/bge-reranker-v2-m3, etc.              |                                                                        |                                                                                           |

## 多模态语言模型列表

根据模型不同，支持以下模态组合：

- **T**ext 文本
- **I**mage 图像
- **V**ideo 视频
- **A**udio 音频

支持通过 `+` 连接的任意模态组合：

- 例如 `T + I` 表示模型支持纯文本、纯图像及图文混合输入。

以 `/` 分隔的模态互斥：

- 例如 `T / I` 表示模型支持纯文本或纯图像输入，但不支持图文混合输入。

关于如何传递多模态输入，请参阅[此页面](https://docs.vllm.ai/en/latest/serving/multimodal_inputs.html#multimodal-inputs)。

**重要\*\***信息\*\*

要为每个文本提示启用多个多模态项，需设置 `limit_mm_per_prompt`（离线推理）或 `--limit-mm-per-prompt`（在线服务）。例如允许每个文本提示传递最多 4 张图像：

离线推理：

```plain
llm = LLM(
    model="Qwen/Qwen2-VL-7B-Instruct",
    limit_mm_per_prompt={"image": 4},
)
```

在线服务：

```plain
vllm serve Qwen/Qwen2-VL-7B-Instruct --limit-mm-per-prompt image=4
```

> **注意**
> 当前 vLLM 仅支持对多模态模型的语言主干添加 LoRA。

### 生成式模型

关于生成式模型的使用详情，请参阅[此页面](https://docs.vllm.ai/en/latest/models/generative_models.html#generative-models)。

#### 文本生成 (`--task generate`)

| 架构                                   | 模型                                                       | 输入                      | HF 模型示例                                                                                                          | [LoRA](https://docs.vllm.ai/en/latest/features/lora.html#lora-adapter) | [PP](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#distributed-serving) | [V1](https://github.com/vllm-project/vllm/issues/8779#) |
| :------------------------------------- | :--------------------------------------------------------- | :------------------------ | :------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------- | :---------------------------------------------------------------------------------------- | :------------------------------------------------------ |
| AriaForConditionalGeneration           | Aria                                                       | T + I+                    | rhymes-ai/Aria                                                                                                       |                                                                        | ✅︎                                                                                       | ✅︎                                                     |
| Blip2ForConditionalGeneration          | BLIP-2                                                     | T + IE                    | Salesforce/blip2-opt-2.7b, Salesforce/blip2-opt-6.7b, etc.                                                           |                                                                        | ✅︎                                                                                       | ✅︎                                                     |
| ChameleonForConditionalGeneration      | Chameleon                                                  | T + I                     | facebook/chameleon-7b etc.                                                                                           |                                                                        | ✅︎                                                                                       | ✅︎                                                     |
| DeepseekVLV2ForCausalLM^               | DeepSeek-VL2                                               | T + I+                    | deepseek-ai/deepseek-vl2-tiny, deepseek-ai/deepseek-vl2-small, deepseek-ai/deepseek-vl2 etc.                         |                                                                        | ✅︎                                                                                       | ✅︎                                                     |
| Florence2ForConditionalGeneration      | Florence-2                                                 | T + I                     | microsoft/Florence-2-base, microsoft/Florence-2-large etc.                                                           |                                                                        |                                                                                           |                                                         |
| FuyuForCausalLM                        | Fuyu                                                       | T + I                     | adept/fuyu-8b etc.                                                                                                   |                                                                        | ✅︎                                                                                       | ✅︎                                                     |
| Gemma3ForConditionalGeneration         | Gemma 3                                                    | T + I+                    | google/gemma-3-4b-it, google/gemma-3-27b-it, etc.                                                                    | ✅︎                                                                    | ✅︎                                                                                       | ⚠️                                                      |
| GLM4VForCausalLM^                      | GLM-4V                                                     | T + I                     | THUDM/glm-4v-9b, THUDM/cogagent-9b-20241220 etc.                                                                     | ✅︎                                                                    | ✅︎                                                                                       | ✅︎                                                     |
| H2OVLChatModel                         | H2OVL                                                      | T + IE+                   | h2oai/h2ovl-mississippi-800m, h2oai/h2ovl-mississippi-2b, etc.                                                       |                                                                        | ✅︎                                                                                       | ✅︎\*                                                   |
| Idefics3ForConditionalGeneration       | Idefics3                                                   | T + I                     | HuggingFaceM4/Idefics3-8B-Llama3 etc.                                                                                | ✅︎                                                                    |                                                                                           | ✅︎                                                     |
| InternVLChatModel                      | InternVideo 2.5, InternVL 2.5, Mono-InternVL, InternVL 2.0 | T + IE+                   | OpenGVLab/InternVideo2_5_Chat_8B, OpenGVLab/InternVL2_5-4B, OpenGVLab/Mono-InternVL-2B, OpenGVLab/InternVL2-4B, etc. |                                                                        | ✅︎                                                                                       | ✅︎                                                     |
| LlavaForConditionalGeneration          | LLaVA-1.5                                                  | T + IE+                   | llava-hf/llava-1.5-7b-hf, TIGER-Lab/Mantis-8B-siglip-llama3 (see note), etc.                                         |                                                                        | ✅︎                                                                                       | ✅︎                                                     |
| LlavaNextForConditionalGeneration      | LLaVA-NeXT                                                 | T + IE+                   | llava-hf/llava-v1.6-mistral-7b-hf, llava-hf/llava-v1.6-vicuna-7b-hf, etc.                                            |                                                                        | ✅︎                                                                                       | ✅︎                                                     |
| LlavaNextVideoForConditionalGeneration | LLaVA-NeXT-Video                                           | T + V                     | llava-hf/LLaVA-NeXT-Video-7B-hf, etc.                                                                                |                                                                        | ✅︎                                                                                       | ✅︎                                                     |
| LlavaOnevisionForConditionalGeneration | LLaVA-Onevision                                            | T + I+ + V+               | llava-hf/llava-onevision-qwen2-7b-ov-hf, llava-hf/llava-onevision-qwen2-0.5b-ov-hf, etc.                             |                                                                        | ✅︎                                                                                       | ✅︎                                                     |
| MiniCPMO                               | MiniCPM-O                                                  | T + IE+ + VE+ + AE+       | openbmb/MiniCPM-o-2_6, etc.                                                                                          | ✅︎                                                                    | ✅︎                                                                                       | ✅︎                                                     |
| MiniCPMV                               | MiniCPM-V                                                  | T + IE+ + VE+             | openbmb/MiniCPM-V-2 (see note), openbmb/MiniCPM-Llama3-V-2_5, openbmb/MiniCPM-V-2_6, etc.                            | ✅︎                                                                    | ✅︎                                                                                       | ✅︎                                                     |
| MllamaForConditionalGeneration         | Llama 3.2                                                  | T + I+                    | meta-llama/Llama-3.2-90B-Vision-Instruct, meta-llama/Llama-3.2-11B-Vision, etc.                                      |                                                                        |                                                                                           |                                                         |
| MolmoForCausalLM                       | Molmo                                                      | T + I+                    | allenai/Molmo-7B-D-0924, allenai/Molmo-7B-O-0924, etc.                                                               | ✅︎                                                                    | ✅︎                                                                                       | ✅︎                                                     |
| NVLM_D_Model                           | NVLM-D 1.0                                                 | T + I+                    | nvidia/NVLM-D-72B, etc.                                                                                              |                                                                        | ✅︎                                                                                       | ✅︎                                                     |
| PaliGemmaForConditionalGeneration      | PaliGemma, PaliGemma 2                                     | T + IE                    | google/paligemma-3b-pt-224, google/paligemma-3b-mix-224, google/paligemma2-3b-ft-docci-448, etc.                     |                                                                        | ✅︎                                                                                       | ⚠️                                                      |
| Phi3VForCausalLM                       | Phi-3-Vision, Phi-3.5-Vision                               | T + IE+                   | microsoft/Phi-3-vision-128k-instruct, microsoft/Phi-3.5-vision-instruct, etc.                                        |                                                                        | ✅︎                                                                                       | ✅︎                                                     |
| Phi4MMForCausalLM                      | Phi-4-multimodal                                           | T + I+ / T + A+ / I+ + A+ | microsoft/Phi-4-multimodal-instruct, etc.                                                                            | ✅︎                                                                    |                                                                                           |                                                         |
| PixtralForConditionalGeneration        | Pixtral                                                    | T + I+                    | mistralai/Mistral-Small-3.1-24B-Instruct-2503, mistral-community/pixtral-12b, etc.                                   |                                                                        | ✅︎                                                                                       | ✅︎                                                     |
| QwenVLForConditionalGeneration^        | Qwen-VL                                                    | T + IE+                   | Qwen/Qwen-VL, Qwen/Qwen-VL-Chat, etc.                                                                                | ✅︎                                                                    | ✅︎                                                                                       | ✅︎                                                     |
| Qwen2AudioForConditionalGeneration     | Qwen2-Audio                                                | T + A+                    | Qwen/Qwen2-Audio-7B-Instruct                                                                                         |                                                                        | ✅︎                                                                                       | ✅︎                                                     |
| Qwen2VLForConditionalGeneration        | QVQ, Qwen2-VL                                              | T + IE+ + VE+             | Qwen/QVQ-72B-Preview, Qwen/Qwen2-VL-7B-Instruct, Qwen/Qwen2-VL-72B-Instruct, etc.                                    | ✅︎                                                                    | ✅︎                                                                                       | ✅︎                                                     |
| Qwen2_5_VLForConditionalGeneration     | Qwen2.5-VL                                                 | T + IE+ + VE+             | Qwen/Qwen2.5-VL-3B-Instruct, Qwen/Qwen2.5-VL-72B-Instruct, etc.                                                      | ✅︎                                                                    | ✅︎                                                                                       | ✅︎                                                     |
| UltravoxModel                          | Ultravox                                                   | T + AE+                   | fixie-ai/ultravox-v0_5-llama-3_2-1b                                                                                  | ✅︎                                                                    | ✅︎                                                                                       | ✅︎                                                     |

^ 您需要通过 `--hf-overrides` 设置架构名称以匹配 vLLM 中的定义。

    • 例如使用 DeepSeek-VL2 系列模型：

      `--hf-overrides '{"architectures": ["DeepseekVLV2ForCausalLM"]}'`

E 可为此模态输入预计算嵌入。

- 可为此模态的每个文本提示输入多项内容。

> **重要\*\***信息\*\*
> 使用 Gemma3 系列模型需通过 `pip install git+https://github.com/huggingface/transformers` 从源码安装 Hugging Face Transformers 库。
>
> 平移扫描图像预处理当前仅 V0 支持（V1 不支持）。可通过传递 `--mm-processor-kwargs '{"do_pan_and_scan": True}'` 启用。

> **警告**
> V0 和 V1 均支持 `Gemma3ForConditionalGeneration` 的纯文本输入，但对图文混合输入的处理存在差异：
>
> V0 正确实现模型的注意力模式：
> 对同一图像的图像 token 使用双向注意力
> 对其他 token 使用因果注意力
> 通过（原生）PyTorch SDPA 与掩码张量实现
> 注意：长提示含图像时可能占用显存较多
>
> V1 当前使用简化注意力模式：
> 对所有 token（含图像 token）使用因果注意力
> 生成合理输出但与原始模型的图文混合输入注意力模式不匹配（尤其当 `{"do_pan_and_scan": True}` 时）
> 未来将更新以支持正确行为
>
> 此限制源于模型的混合注意力模式（图像双向/其他因果）尚未被 vLLM 注意力后端支持。

> **注意** >`h2oai/h2ovl-mississippi-2b` 将在 V1 支持非 FlashAttention 后端后可用。

> **注意**
> 使用 `TIGER-Lab/Mantis-8B-siglip-llama3` 需在运行 vLLM 时传递 `--hf_overrides '{"architectures": ["MantisForConditionalGeneration"]}'`。

> **注意**
> 官方 `openbmb/MiniCPM-V-2` 暂不可用，需使用分支版本（`HwwwH/MiniCPM-V-2`）。详情见 [PR #4087](https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630)。

> **警告**
> 我们的 PaliGemma 实现在 V0 和 V1 中均存在与 Gemma3 相同的问题（见上文）。

### 池化模型

关于池化模型的使用详情，请参阅 [此页面](https://docs.vllm.ai/en/latest/models/pooling_models.html#pooling-models)。

> **重要\*\***信息\*\*
> 由于部分模型架构同时支持生成式和池化任务，您应显式指定任务类型以确保模型以池化模式而非生成模式运行。

#### 文本嵌入 (`--task embed`)

任何文本生成模型均可通过传递 `--task embed` 转换为嵌入模型。

> **注意**
> 为获得最佳效果，应使用专门训练的池化模型。下表列出 vLLM 中已验证的模型。
> |架构|模型|输入|HF 模型示例|[LoRA](https://docs.vllm.ai/en/latest/features/lora.html#lora-adapter)|[PP](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#distributed-serving)|
> |:----|:----|:----|:----|:----|:----|
> |LlavaNextForConditionalGeneration|LLaVA-NeXT-based|T / I|royokong/e5-v||✅︎|
> |Phi3VForCausalLM|Phi-3-Vision-based|T + I|TIGER-Lab/VLM2Vec-Full|🚧|✅︎|
> |Qwen2VLForConditionalGeneration|Qwen2-VL-based|T + I|MrLight/dse-qwen2-2b-mrl-v1||✅︎|

#### 语音转录 (`--task transcription`)

专为自动语音识别训练的 Speech2Text 模型。

| 架构    | 模型          | 模型示例                      | [LoRA](https://docs.vllm.ai/en/latest/features/lora.html#lora-adapter) | [PP](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#distributed-serving) |
| :------ | :------------ | :---------------------------- | :--------------------------------------------------------------------- | :---------------------------------------------------------------------------------------- |
| Whisper | Whisper-based | openai/whisper-large-v3-turbo | 🚧                                                                     | 🚧                                                                                        |

---

## 模型支持政策

vLLM 致力于促进第三方模型在生态中的集成与支持。我们的方法旨在平衡鲁棒性需求与广泛支持模型的实际限制。以下是第三方模型的管理方式：

1. **社区驱动支持**：鼓励社区贡献添加新模型。当用户请求支持新模型时，我们欢迎社区提交 PR。这些贡献主要评估生成输出的合理性，而非与 transformers 等现有实现的严格一致性。**贡献呼吁**：来自模型供应商的直接 PR 尤为欢迎！

2. **尽力保证一致性**：虽然我们力求保持 vLLM 实现的模型与其他框架（如 transformers）的一致性，但完全对齐并非总能实现。加速技术和低精度计算等因素可能引入差异。我们的承诺是确保实现的模型功能正常且输出合理。

> **提示**
> 比较 HuggingFace Transformers 的 `model.generate` 与 vLLM 的 `llm.generate` 输出时，请注意前者读取模型的生成配置文件（即 [generation_config.json](https://github.com/huggingface/transformers/blob/19dabe96362803fb0a9ae7073d03533966598b17/src/transformers/generation/utils.py#L1945)）并应用默认生成参数，而后者仅使用函数传递的参数。比较输出时需确保所有采样参数一致。

1. **问题解决与模型更新**：鼓励用户报告第三方模型的任何缺陷。修复建议应通过 PR 提交，并清晰说明问题及解决方案依据。若某模型的修复影响其他模型，我们依赖社区指出并解决这些跨模型依赖。注意：提交修复 PR 时，通知原作者征求意见是良好实践。

2. **监控与更新**：关注特定模型的用户应监控这些模型的提交历史（例如跟踪 main/vllm/model_executor/models 目录的变更）。这种主动方式有助于用户及时了解可能影响所用模型的更新。

3. **选择性聚焦**：我们的资源主要投向用户关注度高、影响大的模型。使用较少的模型可能获得较少关注，这些模型的维护和改进更依赖社区积极参与。

通过这种方式，vLLM 培育了一个协作环境，核心开发团队与广大社区共同贡献于生态中第三方模型的鲁棒性和多样性。

注意：作为推理引擎，vLLM 不引入新模型。因此，所有 vLLM 支持的模型在此意义上均为第三方模型。

我们对模型的测试分为以下级别：

1. **严格一致性**：在贪婪解码下比较模型输出与 HuggingFace Transformers 库的输出。这是最严格的测试。通过此测试的模型见 [模型测试](https://github.com/vllm-project/vllm/blob/main/tests/models)。

2. **输出合理性**：通过测量输出的困惑度及检查明显错误，验证模型输出是否合理连贯。此为较宽松的测试。

3. **运行时功能性**：检查模型能否无错误加载运行。此为最宽松的测试。通过此测试的模型见 [功能测试](https://github.com/vllm-project/vllm/tree/main/tests) 和 [示例](https://github.com/vllm-project/vllm/tree/main/main/examples)。

4. **社区反馈**：依赖社区提供模型反馈。若模型存在故障或未达预期，鼓励用户提交 Issue 报告或提交 PR 修复。其余模型归入此类。
