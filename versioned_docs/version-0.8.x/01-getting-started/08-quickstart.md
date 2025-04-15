---
title: 快速入门
---


本指南将说明如何使用 vLLM 进行以下操作：


* 对数据集进行离线批量推理；

* 为大语言模型构建 API 服务器；

* 启动与 OpenAI 兼容的 API 服务器。


在继续进行本指南之前，请务必完成[安装说明](https://docs.vllm.ai/en/latest/getting_started/installation.html#installation)。


**注意**


默认情况下，vLLM 从 [HuggingFace](https://huggingface.co/) 下载模型。如果您想在以下示例中使用 [ModelScope](https://www.modelscope.cn) 中的模型，请设置环境变量：

```shell
export VLLM_USE_MODELSCOPE=True
```


## 离线批量推理


我们首先演示一个使用 vLLM 对数据集进行离线批处理推理的案例。也就是说，我们使用 vLLM 生成输入提示列表的文本。


从 vLLM 导入 `LLM` 和 `SamplingParams`。`LLM`类是使用 vLLM 引擎运行离线推理的主要类。`SamplingParams`类指定了采样过程的参数。

```python
from vllm import LLM, SamplingParams
```


定义输入提示列表和生成的采样参数。采样温度设置为 0.8，核采样概率 (nucleus sampling probability) 设置为 0.95。有关采样参数的更多信息，请参阅[类定义](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)。

```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
```


使用`LLM`类和 [OPT-125M 模型](https://arxiv.org/abs/2205.01068)初始化 vLLM 引擎以进行离线推理。支持的模型列表可以在[支持的模型](https://docs.vllm.ai/en/latest/models/supported_models.html#supported-models)中找到。

```python
llm = LLM(model="facebook/opt-125m")
```


调用`llm.generate`生成输出。它将输入提示添加到 vLLM 引擎的等待队列中，并执行 vLLM 引擎来生成高吞吐量的输出。输出作为`RequestOutput`对象列表返回，其中包括所有输出的 tokens。

```python
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
# 打印输出

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```


这个代码示例也可以在 [examples/offline_inference.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference.py) 中找到。

## 

## 兼容 OpenAI 服务器


vLLM 可以作为一个实现了 OpenAI API 协议的服务器进行部署。这使得 vLLM 可以直接替代使用 OpenAI API 的应用程序。默认情况下，它在 [http://localhost:8000](http://localhost:8000) 启动服务器。您可以使用 `--host` 和`--port` 参数指定地址。当前，该服务器一次仅托管一个模型（在下面的命令中为 OPT-125M），并实现了[模型列表 (list models)](https://platform.openai.com/docs/api-reference/models/list)、[创建聊天补全 (create chat completion)](https://platform.openai.com/docs/api-reference/chat/completions/create) 和[创建完成 (create completion)](https://platform.openai.com/docs/api-reference/completions/create) 端点。我们正在积极添加对更多端点的支持。


启动服务器：

```plain
vllm serve facebook/opt-125m
```


默认情况下，服务器使用存储在 tokenizer 中的预定义聊天模板。您可以使用 `--chat-template` 参数覆盖此模板：

```plain
vllm serve facebook/opt-125m --chat-template ./examples/template_chatml.jinja
```


该服务器可以按照与 OpenAI API 相同的格式进行查询。例如，列出模型：

```plain
curl http://localhost:8000/v1/models
```


您可以传入参数`--api-key`或设置环境变量`VLLM_API_KEY`，以使服务器能够检查标头中的 API 密钥。


### 在 vLLM 中使用 OpenAI Completions API


使用输入提示查询模型：

```plain
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```


由于该服务器与 OpenAI API 兼容，因此您可以把它作为使用 OpenAI API 的任意应用程序的直接替代品。例如，另一种查询服务器的方法是通过 `openai`的 python 包：

```python
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
# 使用 vLLM 的 API 服务器需要修改 OpenAI 的 API 密钥和 API 库。

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(model="facebook/opt-125m",
                                      prompt="San Francisco is a")
print("Completion result:", completion)
```


有关更详细的客户端示例，请参阅 [examples/openai_completion_client.py](https://github.com/vllm-project/vllm/blob/main/examples/openai_completion_client.py)。


### 在 vLLM 中使用 OpenAI Chat API


vLLM 服务器在设计上支持 OpenAI Chat API，允许您与模型进行动态对话。聊天界面是一种与模型交流更具交互性的方式，可以进行来回交流，并将对话历史存储下来。这对于需要上下文或更详细解释的任务非常有用。


使用 OpenAI Chat API 查询模型：


您可以使用[创建聊天补全 (create chat completion)](https://platform.openai.com/docs/api-reference/chat/completions/create) 端点在类似聊天的界面中与模型进行交流：

```plain
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```


Python 客户端示例：


使用 *openai* 的 python 包，您还可以以类似聊天的方式与模型进行交流:

```python
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.

# 使用 vLLM 的 API 服务器需要设置 OpenAI 的 API 密钥和 API 库。

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
)
print("Chat response:", chat_response)
```


有关 chat API 的更深入示例和高级功能，您可以参考 OpenAI 官方文档。

