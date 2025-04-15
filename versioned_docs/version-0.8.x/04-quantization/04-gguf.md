---
title: GGUF
---


**警告：**

请注意，vLLM 对 GGUF 的支持仍处于高度实验阶段，尚未进行充分优化，因此可能与其他功能不兼容。目前，您可以使用 GGUF 来减少内存占用。如果在使用过程中遇到任何问题，请向 vLLM 团队反馈。  


**警告：**

目前，vLLM 仅支持加载单文件 GGUF 模型。如果您有多文件的 GGUF 模型，可以使用 [gguf-split](https://github.com/ggerganov/llama.cpp/pull/6135) 工具将其合并为一个单文件模型。  

 :::

To run a GGUF model with vLLM, you can download and use the local GGUF model from [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) with the following command:

如需在 vLLM 中运行 GGUF 模型，您可以通过以下命令从 [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) 下载并使用本地 GGUF 模型：

```python
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
# We recommend using the tokenizer from base model to avoid long-time and buggy tokenizer conversion.
# 我们建议使用基础模型的 tokenizer，以避免耗时且存在问题的 tokenizer 转换。
vllm serve ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
```


您还可以添加 `--tensor-parallel-size 2` 启用两块 GPU 进行张量并行推理：

```python
# We recommend using the tokenizer from base model to avoid long-time and buggy tokenizer conversion.
# 我们建议使用基础模型的 tokenizer，以避免耗时且存在问题的 tokenizer 转换。
vllm serve ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tensor-parallel-size 2
```


**警告：**

我们建议您使用基础模型的 tokenizer 而不是 GGUF 模型的 tokenizer。因为从 GGUF 转换 tokenizer 既耗时又不稳定，尤其是对于一些词汇量较大的模型。  


您也可以通过 LLM 入口直接使用 GGUF 模型：

```python
from vllm import LLM, SamplingParams


# In this script, we demonstrate how to pass input to the chat method:
# 在此脚本中，我们演示了如何将输入传递给 chat 方法：


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


# Create a sampling params object.
# 创建 SamplingParams 对象。
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


# 创建一个 LLM。
llm = LLM(model="./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
         tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
# 从提示中生成文本。输出是一个 RequestOutput 列表，包含提示、生成文本和其他信息
outputs = llm.chat(conversation, sampling_params)


# Print the outputs.
# 打印输出
for output in outputs:
   prompt = output.prompt
   generated_text = output.outputs[0].text
   print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```


