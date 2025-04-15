---
title: 支持的模型
---


vLLM 支持 [HuggingFace Transformers](https://huggingface.co/models) 中的各种生成性 Transformer 模型。以下是 vLLM 目前支持的模型架构列表。除了每种架构之外，我们还提供了一些使用它的流行模型。


## 仅文本语言模型

### 文本生成

|架构|模型|HF 模型案例|[LoRA](https://LoRA)|[PP](https://PP)|
|:----|:----|:----|:----|:----|
|AquilaForCausalLM|Aquila, Aquila2|BAAI/Aquila-7B, BAAI/AquilaChat-7B, etc.|✅︎|✅︎|
|ArcticForCausalLM|Arctic|Snowflake/snowflake-arctic-base, Snowflake/snowflake-arctic-instruct, etc.||✅︎|
|BaiChuanForCausalLM|Baichuan2, Baichuan|baichuan-inc/Baichuan2-13B-Chat, baichuan-inc/Baichuan-7B, etc.|✅︎|✅︎|
|BloomForCausalLM|BLOOM, BLOOMZ, BLOOMChat|bigscience/bloom, bigscience/bloomz, etc.||✅︎|
|BartForConditionalGeneration|BART|facebook/bart-base, facebook/bart-large-cnn, etc.|||
|ChatGLMModel|ChatGLM|THUDM/chatglm2-6b, THUDM/chatglm3-6b, etc.|✅︎|✅︎|
|CohereForCausalLM|Command-R|CohereForAI/c4ai-command-r-v01, etc.|✅︎|✅︎|
|DbrxForCausalLM|DBRX|databricks/dbrx-base, databricks/dbrx-instruct, etc.||✅︎|
|DeciLMForCausalLM|DeciLM|Deci/DeciLM-7B, Deci/DeciLM-7B-instruct, etc.||✅︎|
|DeepseekForCausalLM|DeepSeek|deepseek-ai/deepseek-llm-67b-base, deepseek-ai/deepseek-llm-7b-chat etc.||✅︎|
|DeepseekV2ForCausalLM|DeepSeek-V2|deepseek-ai/DeepSeek-V2, deepseek-ai/DeepSeek-V2-Chat etc.||✅︎|
|ExaoneForCausalLM|EXAONE-3|LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct, etc.|✅︎|✅︎|
|FalconForCausalLM|Falcon|tiiuae/falcon-7b, tiiuae/falcon-40b, tiiuae/falcon-rw-7b, etc.||✅︎|
|GemmaForCausalLM|Gemma|google/gemma-2b, google/gemma-7b, etc.|✅︎|✅︎|
|Gemma2ForCausalLM|Gemma2|google/gemma-2-9b, google/gemma-2-27b, etc.|✅︎|✅︎|
|GPT2LMHeadModel|GPT-2|gpt2, gpt2-xl, etc.||✅︎|
|GPTBigCodeForCausalLM|StarCoder, SantaCoder, WizardCoder|bigcode/starcoder, bigcode/gpt_bigcode-santacoder, WizardLM/WizardCoder-15B-V1.0, etc.|✅︎|✅︎|
|GPTJForCausalLM|GPT-J|EleutherAI/gpt-j-6b, nomic-ai/gpt4all-j, etc.||✅︎|
|GPTNeoXForCausalLM|GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM|EleutherAI/gpt-neox-20b, EleutherAI/pythia-12b, OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5, databricks/dolly-v2-12b, stabilityai/stablelm-tuned-alpha-7b, etc.||✅︎|
|GraniteForCausalLM|PowerLM|ibm/PowerLM-3b etc.|✅︎|✅︎|
|GraniteMoeForCausalLM|PowerMoE|ibm/PowerMoE-3b etc.|✅︎|✅︎|
|InternLMForCausalLM|InternLM|internlm/internlm-7b, internlm/internlm-chat-7b, etc.|✅︎|✅︎|
|InternLM2ForCausalLM|InternLM2|internlm/internlm2-7b, internlm/internlm2-chat-7b, etc.||✅︎|
|JAISLMHeadModel|Jais|core42/jais-13b, core42/jais-13b-chat, core42/jais-30b-v3, core42/jais-30b-chat-v3, etc.||✅︎|
|JambaForCausalLM|Jamba|ai21labs/AI21-Jamba-1.5-Large, ai21labs/AI21-Jamba-1.5-Mini, ai21labs/Jamba-v0.1, etc.|✅︎||
|LlamaForCausalLM|Llama 3.1, Llama 3, Llama 2, LLaMA, Yi|meta-llama/Meta-Llama-3.1-405B-Instruct, meta-llama/Meta-Llama-3.1-70B, meta-llama/Meta-Llama-3-70B-Instruct, meta-llama/Llama-2-70b-hf, 01-ai/Yi-34B, etc.|✅︎|✅︎|
|MambaForCausalLM|Mamba|state-spaces/mamba-130m-hf, state-spaces/mamba-790m-hf, state-spaces/mamba-2.8b-hf, etc.|||
|MiniCPMForCausalLM|MiniCPM|openbmb/MiniCPM-2B-sft-bf16, openbmb/MiniCPM-2B-dpo-bf16, openbmb/MiniCPM-S-1B-sft, etc.|✅︎|✅︎|
|MiniCPM3ForCausalLM|MiniCPM3|openbmb/MiniCPM3-4B, etc.|✅︎|✅︎|
|MistralForCausalLM|Mistral, Mistral-Instruct|mistralai/Mistral-7B-v0.1, mistralai/Mistral-7B-Instruct-v0.1, etc.|✅︎|✅︎|
|MixtralForCausalLM|Mixtral-8x7B, Mixtral-8x7B-Instruct|mistralai/Mixtral-8x7B-v0.1, mistralai/Mixtral-8x7B-Instruct-v0.1, mistral-community/Mixtral-8x22B-v0.1, etc.|✅︎|✅︎|
|MPTForCausalLM|MPT, MPT-Instruct, MPT-Chat, MPT-StoryWriter|mosaicml/mpt-7b, mosaicml/mpt-7b-storywriter, mosaicml/mpt-30b, etc.||✅︎|
|NemotronForCausalLM|Nemotron-3, Nemotron-4, Minitron|nvidia/Minitron-8B-Base, mgoin/Nemotron-4-340B-Base-hf-FP8, etc.|✅︎|✅︎|
|OLMoForCausalLM|OLMo|allenai/OLMo-1B-hf, allenai/OLMo-7B-hf, etc.||✅︎|
|OLMoEForCausalLM|OLMoE|allenai/OLMoE-1B-7B-0924, allenai/OLMoE-1B-7B-0924-Instruct, etc.|✅︎|✅︎|
|OPTForCausalLM|OPT, OPT-IML|facebook/opt-66b, facebook/opt-iml-max-30b, etc.||✅︎|
|OrionForCausalLM|Orion|OrionStarAI/Orion-14B-Base, OrionStarAI/Orion-14B-Chat, etc.||✅︎|
|PhiForCausalLM|Phi|microsoft/phi-1_5, microsoft/phi-2, etc.|✅︎|✅︎|
|Phi3ForCausalLM|Phi-3|microsoft/Phi-3-mini-4k-instruct, microsoft/Phi-3-mini-128k-instruct, microsoft/Phi-3-medium-128k-instruct, etc.|✅︎|✅︎|
|Phi3SmallForCausalLM|Phi-3-Small|microsoft/Phi-3-small-8k-instruct, microsoft/Phi-3-small-128k-instruct, etc.||✅︎|
|PhiMoEForCausalLM|Phi-3.5-MoE|microsoft/Phi-3.5-MoE-instruct, etc.|✅︎|✅︎|
|PersimmonForCausalLM|Persimmon|adept/persimmon-8b-base, adept/persimmon-8b-chat, etc.||✅︎|
|QWenLMHeadModel|Qwen|Qwen/Qwen-7B, Qwen/Qwen-7B-Chat, etc.||✅︎|
|Qwen2ForCausalLM|Qwen2|Qwen/Qwen2-beta-7B, Qwen/Qwen2-beta-7B-Chat, etc.|✅︎|✅︎|
|Qwen2MoeForCausalLM|Qwen2MoE|Qwen/Qwen1.5-MoE-A2.7B, Qwen/Qwen1.5-MoE-A2.7B-Chat, etc.||✅︎|
|StableLmForCausalLM|StableLM|stabilityai/stablelm-3b-4e1t, stabilityai/stablelm-base-alpha-7b-v2, etc.||✅︎|
|Starcoder2ForCausalLM|Starcoder2|bigcode/starcoder2-3b, bigcode/starcoder2-7b, bigcode/starcoder2-15b, etc.||✅︎|
|SolarForCausalLM|Solar Pro|upstage/solar-pro-preview-instruct, etc.|✅︎|✅︎|
|XverseForCausalLM|XVERSE|xverse/XVERSE-7B-Chat, xverse/XVERSE-13B-Chat, xverse/XVERSE-65B-Chat, etc.|✅︎|✅︎|



**注意：**

目前，vLLM 的 ROCm 版本仅支持 Mistral 和 Mixtral，上下文长度最多为 4096。


### 文本 Embedding

|架构|模型|HF 模型案例|[LoRA](https://docs.vllm.ai/en/latest/models/lora.html#lora)|[PP](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#distributed-serving)|
|:----|:----|:----|:----|:----|
|Gemma2Model|Gemma2-based|BAAI/bge-multilingual-gemma2, etc.||✅︎|
|MistralModel|Mistral-based|intfloat/e5-mistral-7b-instruct, etc.||✅︎|

**注意：**

有些模型架构同时支持生成和嵌入任务。在这种情况下，你需要传入 `--task embedding` 参数，才能以嵌入模式运行该模型。


### 获奖模型

|架构|模型|HF 模型案例|[LoRA](https://docs.vllm.ai/en/latest/models/lora.html#lora)|[PP](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#distributed-serving)|
|:----|:----|:----|:----|:----|
|Qwen2ForRewardModel|Qwen2-based|Qwen/Qwen2.5-Math-RM-72B, etc.||✅︎|

**注意：**

作为临时措施，这些模型通过 Embeddings API 获得支持。请参阅[该 RFC](https://github.com/vllm-project/vllm/issues/8967) 了解最新变化。


## 多模态语言模型

根据模型的不同，支持以下模态：

* 文本

* 图像

* 视频

* 音频

### 文本生成

|架构|模型|输入|HF 模型案例|[LoRA](https://docs.vllm.ai/en/latest/models/lora.html#lora)|[PP](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#distributed-serving)|
|:----|:----|:----|:----|:----|:----|
|Blip2ForConditionalGeneration|BLIP-2|T + IE|Salesforce/blip2-opt-2.7b, Salesforce/blip2-opt-6.7b, etc.||✅︎|
|ChameleonForConditionalGeneration|Chameleon|T + I|facebook/chameleon-7b etc.||✅︎|
|FuyuForCausalLM|Fuyu|T + I|adept/fuyu-8b etc.||✅︎|
|ChatGLMModel|GLM-4V|T + I|THUDM/glm-4v-9b etc.||✅︎|
|InternVLChatModel|InternVL2|T + IE+|OpenGVLab/InternVL2-4B, OpenGVLab/InternVL2-8B, etc.||✅︎|
|LlavaForConditionalGeneration|LLaVA-1.5|T + IE+|llava-hf/llava-1.5-7b-hf, llava-hf/llava-1.5-13b-hf, etc.||✅︎|
|LlavaNextForConditionalGeneration|LLaVA-NeXT|T + IE+|llava-hf/llava-v1.6-mistral-7b-hf, llava-hf/llava-v1.6-vicuna-7b-hf, etc.||✅︎|
|LlavaNextVideoForConditionalGeneration|LLaVA-NeXT-Video|T + V|llava-hf/LLaVA-NeXT-Video-7B-hf, etc.||✅︎|
|LlavaOnevisionForConditionalGeneration|LLaVA-Onevision|T + I+ + V|llava-hf/llava-onevision-qwen2-7b-ov-hf, llava-hf/llava-onevision-qwen2-0.5b-ov-hf, etc.||✅︎|
|MiniCPMV|MiniCPM-V|T + IE+|openbmb/MiniCPM-V-2 (see note), openbmb/MiniCPM-Llama3-V-2_5, openbmb/MiniCPM-V-2_6, etc.|✅︎|✅︎|
|MllamaForConditionalGeneration|Llama 3.2|T + I|meta-llama/Llama-3.2-90B-Vision-Instruct, meta-llama/Llama-3.2-11B-Vision, etc.|||
|MolmoForCausalLM|Molmo|Image|allenai/Molmo-7B-D-0924, allenai/Molmo-72B-0924, etc.||✅︎|
|NVLM_D_Model|NVLM-D 1.0|T + IE+|nvidia/NVLM-D-72B, etc.||✅︎|
|PaliGemmaForConditionalGeneration|PaliGemma|T + IE|google/paligemma-3b-pt-224, google/paligemma-3b-mix-224, etc.||✅︎|
|Phi3VForCausalLM|Phi-3-Vision, Phi-3.5-Vision|T + IE+|microsoft/Phi-3-vision-128k-instruct, microsoft/Phi-3.5-vision-instruct etc.||✅︎|
|PixtralForConditionalGeneration|Pixtral|T + I+|mistralai/Pixtral-12B-2409||✅︎|
|QWenLMHeadModel|Qwen-VL|T + IE+|Qwen/Qwen-VL, Qwen/Qwen-VL-Chat, etc.||✅︎|
|Qwen2VLForConditionalGeneration|Qwen2-VL|T + IE+ + V+|Qwen/Qwen2-VL-2B-Instruct, Qwen/Qwen2-VL-7B-Instruct, Qwen/Qwen2-VL-72B-Instruct, etc.||✅︎|
|UltravoxModel|Ultravox|T + AE+|fixie-ai/ultravox-v0_3||✅︎|


E 预计算的嵌入可以作为此模态的输入。

+ 对于此模态，每个文本提示可以输入多个项目。


**注意**

对于 `openbmb/MiniCPM-V-2` ，官方仓库还不能工作，所以我们现在需要使用一个分支版本 （`HwwwH/MiniCPM-V-2`）。更多详情请参见: [https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630](https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630)


### 多模态 Embedding

|架构|模型|输入|HF 模型案例|[LoRA](https://docs.vllm.ai/en/latest/models/lora.html#lora)|[PP](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#distributed-serving)|
|:----|:----|:----|:----|:----|:----|
|Phi3VForCausalLM|Phi-3-Vision-based|T + I|TIGER-Lab/VLM2Vec-Full|🚧|✅︎|

**注意：**

有些模型架构同时支持生成和嵌入任务。在这种情况下，你需要传入 `--task embedding` 参数，才能以嵌入模式运行该模型。


如果您的模型使用上述模型架构之一，您可以使用 vLLM 无缝运行您的模型。否则，请参阅 `添加新模型 <adding_a_new_model>` 和 `启用多模式输入 <enabling_multimodal_inputs>` 中的说明了解如何为您的模型提供支持。或者，您可以在我们的 [GitHub](https://github.com/vllm-project/vllm/issues) 项目上提出问题。


**提示：** 

要确认您的模型是否得到支持，最简单的方法是执行以下程序：

```python
from vllm import LLM


llm = LLM(model=...)  # Name or path of your model


llm = LLM(model=...) # 模型的名称或路径


output = llm.generate("Hello, my name is")
print(output)
```
如果 vLLM 成功生成文本，则表明您的模型受支持。

**提示：**

要使用 [ModelScope](https://www.modelscope.cn) 中的模型而不是 HuggingFace Hub 的模型，请设置一个环境变量： 

```plain
export VLLM_USE_MODELSCOPE=True
```
并与 `trust_remote_code=True` 一起使用。
```python
from vllm import LLM


llm = LLM(model=..., revision=..., trust_remote_code=True)  # Name or path of your model


llm = LLM(model=..., revision=..., trust_remote_code=True) # 模型的名称或路径


output = llm.generate("Hello, my name is")
print(output)
```


# 模型支持政策

在 vLLM 中，我们致力于促进第三方模型在我们的生态系统中的集成和支持。我们的方法旨在平衡对鲁棒性的需求和支持各种模型范围的实际限制。以下是我们管理第三方模型支持的方式：


1. **社区驱动的支持****：**我们倡导并鼓励社区成员积极参与，引入新模型。每当用户提出对新模型的支持需求时，我们都非常期待来自社区的拉取请求 (PRs)。在评估这些贡献时，我们主要关注它们产生的输出的合理性，而不是它们与现有实现（例如在 transformers 中的实现）的严格一致性。 贡献号召：非常感谢直接来自于模型供应商的 PR！

2. **尽力保持一致性****：**虽然我们努力保持 vLLM 中实现的模型与 transformers 等其他框架之间的一致性，但并不总是能够完全对齐。加速技术的使用和低精度计算等因素可能会引入差异。我们承诺确保所实现的模型功能正常并产生合理的结果。

3. **问题解决和模型更新****：**我们鼓励用户报告在使用第三方模型时遇到的任何错误或问题。建议通过拉取请求（PRs）提交修复方案，并清楚地说明问题所在以及提出解决方案的理由。如果一个模型的修复影响另一个模型，我们依靠社区来突出和解决这些跨模型依赖关系。注意：对于 bug 修复 PR，通知原作者并获得他们的反馈是一种良好的礼仪。

4. **监控和更新****：**对特定模型感兴趣的用户建议监控这些模型的提交历史记录 （例如，通过跟踪 main/vllm/model_executor/models 目录中的更改）。这种主动方法可以帮助用户随时了解可能影响他们使用的模型的更新和更改。

5. **有选择的关注****：**我们的资源主要集中在那些受到广泛用户关注和具有较大影响力的模型上。对于那些使用频率不高的模型，我们可能无法投入同样多的精力，因此我们依赖社区在这些模型的维护和提升中扮演更加积极的角色。


通过这种方法，vLLM 营造了一个协作环境，核心开发团队和更广泛的社区都为我们生态系统中支持的第三方模型的稳健性和多样性做出了贡献。


请注意，作为推理引擎，vLLM 并没有引入新模型。因此，vLLM 支持的所有模型都是第三方模型。


我们对模型进行以下级别的测试：

1. **严格一致性**：我们将模型的输出与 HuggingFace Transformers 库中模型在贪婪解码下的输出进行比较。这是最严格的测试。请参考[模型测试](https://github.com/vllm-project/vllm/blob/main/tests/models)，了解哪些模型通过了此测试。

2. **输出合理性**：我们通过测量输出的困惑度 (perplexity) 并检查是否存在明显错误，来判断模型输出是否合理和连贯。这是一个较为宽松的测试。

3. **运行时功能**：我们检查模型是否能够无错误地加载并运行。这是最不严格的测试。请参考已通过此测试的模型的[功能测试](https://github.com/vllm-project/vllm/tree/main/tests)和[示例](https://github.com/vllm-project/vllm/tree/main/examples)。

4. **社区反馈**：我们依赖社区对模型提供反馈。如果某个模型出现问题或未按预期工作，我们鼓励用户提出问题报告或提交拉取请求 (PRs) 以修复问题。其余模型属于此类。


