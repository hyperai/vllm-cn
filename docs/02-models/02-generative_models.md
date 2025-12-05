---
title: 生成模型
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM能够很好地支持生成模型，它兼容并能够有效运行大多数的大型语言模型 (LLMs)。

在 vLLM 中，生成模型实现了 VllmModelForTextGeneration 接口。这些模型基于输入的最终隐藏状态，输出生成 token 的对数概率，然后通过 Sampler 处理获取最终文本。

对于生成模型，唯一支持的 --task 选项是 "generate"。通常，这会自动推断，因此您无需手动指定。

## 离线推理

LLM 类提供了多种离线推理的方法。有关初始化模型时的选项列表，请参阅[引擎参数](#engine-args)。

### LLM.generate

generate 方法适用于 vLLM 中的所有生成模型。它与 [HF Transformers 中的对应方法](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate) generate 类似，不同之处在于它还自动执行了 tokenization 和 detokenization。

```python
llm = LLM(model="facebook/opt-125m")
outputs = llm.generate("Hello, my name is")

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

您可以通过传递 vllm.SamplingParams 来选择性地控制语言生成。例如，您可以通过设置 temperature=0 来使用贪婪采样：

```python
llm = LLM(model="facebook/opt-125m")
params = SamplingParams(temperature=0)
outputs = llm.generate("Hello, my name is", params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

代码示例可以在这里找到：[examples/offline_inference/basic/basic.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic/basic.py)

### LLM.beam_search

beam_search 方法在 generate 的基础上实现了[束搜索](https://huggingface.co/docs/transformers/en/generation_strategies#beam-search-decoding)。例如，使用 5 个束进行搜索并最多输出 50 个 token：

```python
llm = LLM(model="facebook/opt-125m")
params = BeamSearchParams(beam_width=5, max_tokens=50)
outputs = llm.generate("Hello, my name is", params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### LLM.chat

chat 方法在 generate 的基础上实现了聊天功能。具体来讲，它接受类似于 [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) 的输入，并自动应用模型的[聊天模板](https://huggingface.co/docs/transformers/en/chat_templating)来格式化提示。

> **注意：**
> 一般情况下，只有经过指令调优的模型才有聊天模板。基础模型没有经过训练来响应聊天对话，可能表现不佳。

```python
llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "Hello"
    },
    {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
    },
    {
        "role": "user",
        "content": "Write an essay about the importance of higher education.",
    },
]
outputs = llm.chat(conversation)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

代码示例可以在这里找到：[examples/offline_inference/basic/chat.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic/chat.py)

如果模型没有聊天模板，或者您想指定另一个模板，您可以显式传递一个聊天模板：

```python
from vllm.entrypoints.chat_utils import load_chat_template

# 您可以在 examples/ 下找到现有聊天模板的列表
custom_template = load_chat_template(chat_template="<path_to_template>")
print("Loaded chat template:", custom_template)

outputs = llm.chat(conversation, chat_template=custom_template)
```

## 在线服务

我们的 [OpenAI 兼容服务器](#openai-compatible-server)提供了与离线 API 对应的端点：

- [Completions API](#completions-api) 类似于 LLM.generate，但只接受文本。
- [Chat API](#chat-api) 类似于 LLM.chat，它接受文本和[多模态输入](#multimodal-inputs)，适用于具有聊天模板的模型。
