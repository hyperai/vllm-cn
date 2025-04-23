---

title: GGUF

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


>**警告**
>请注意，vLLM 中的 GGUF 支持目前处于高度实验性阶段，且未进行充分优化，可能与其他功能不兼容。目前，您可以使用 GGUF 来减少内存占用。如果您遇到任何问题，请向 vLLM 团队报告。  

>**警告**
>目前，vLLM 仅支持加载单文件 GGUF 模型。如果您有多文件的 GGUF 模型，可以使用 [gguf-split](https://github.com/ggerganov/llama.cpp/pull/6135) 工具将其合并为单文件模型。  
To run a GGUF model with vLLM, you can download and use the local GGUF model from [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) with the following command:

如需使用 vLLM 运行 GGUF 模型，您可以从 [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) 下载相应的本地 GGUF 模型，并按照以下命令进行操作：

```plain
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
# We recommend using the tokenizer from base model to avoid long-time and buggy tokenizer conversion.
# 我们建议使用基础模型的 tokenizer，以避免耗时且存在问题的 tokenizer 转换。
vllm serve ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
```
>**警告**
>我们推荐您使用基础模型的 tokenizer，而不是 GGUF 模型的。这是因为将  tokenizer 从GGUF 转换过来不仅耗时，而且在转换过程中可能不稳定，尤其是对于那些词汇量较大的模型来说更是如此。

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


