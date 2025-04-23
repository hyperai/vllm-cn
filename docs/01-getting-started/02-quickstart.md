---

title: 快速开始

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


本指南将帮助您快速开始使用 vLLM 进行以下操作：

* [离线批量推理](#quickstart-offline)
* [使用 OpenAI 兼容服务器进行在线服务](#quickstart-online)


## 依赖条件

* 系统: Linux
* Python: 3.9 -- 3.12


## 安装

如果您使用的是 NVIDIA GPU，可以直接使用 [pip](https://pypi.org/project/vllm/) 安装 vLLM。


推荐使用 [uv](https://docs.astral.sh/uv/)（一个非常快速的 Python 环境管理器）来创建和管理 Python 环境。请按照[文档](https://docs.astral.sh/uv/#getting-started) 安装 uv。安装完成后，您可以使用以下命令创建一个新的 Python 环境并安装 vLLM：

```plain
uv venv myenv --python 3.12 --seed
source myenv/bin/activate
uv pip install vllm
```


另一种便捷的方式是使用 *uv run* 命令的 *--with [dependency]* 选项，它允许您在不创建环境的情况下运行诸如 vllm serve 的命令：

```plain
uv run --with vllm vllm --help
```


您也可以使用 [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) 来创建和管理 Python 环境。

```plain
conda create -n myenv python=3.12 -y
conda activate myenv
pip install vllm
```


>**注意：**
>对于非 CUDA 平台，请参考[安装](#安装)获取安装 vLLM 的具体说明。

## 离线批量推理

离线批量推理


安装 vLLM 后，您可以开始为输入提示列表生成文本（即离线批量推理）。请参阅示例脚本：[examples/offline_inference/basic/basic.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic/basic.py)


该示例的第一行导入了 LLM 和 SamplingParams 类：

* LLM 是用于运行 vLLM 引擎离线推理的主类。
* SamplingParams 指定了采样过程的参数。

```python
from vllm import LLM, SamplingParams
```


下一部分定义了输入提示列表和文本生成的采样参数。[采样温度](https://arxiv.org/html/2402.05201v1)设置为 0.8，[核心采样概率](https://en.wikipedia.org/wiki/Top-p_sampling)设置为 0.95。您可以在[采样参数](#采样参数)中找到有关采样参数的更多信息。

```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
```


LLM 类初始化了 vLLM 引擎和 [OPT-125M 模型](https://arxiv.org/abs/2205.01068)以进行离线推理。支持的模型列表可以在[支持的模型](#支持的模型)文档中找到。

```python
llm = LLM(model="facebook/opt-125m")
```


>**注意：**
>默认情况下，vLLM 从 HuggingFace 下载模型。如果您想使用 ModelScope 的模型，请在初始化引擎之前设置环境变量 VLLM_USE_MODELSCOPE。

现在，有趣的部分来了！使用 llm.generate 生成输出。它将输入提示添加到 vLLM 引擎的等待队列中，并执行 vLLM 引擎以高吞吐量生成输出。输出以 RequestOutput 对象列表的形式返回，其中包含所有输出 tokens。

```python
outputs = llm.generate(prompts, sampling_params)


for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```


## OpenAI 兼容服务器

vLLM 可以部署为实现 OpenAI API 协议的服务器。这使得 vLLM 可以作为使用 OpenAI API 的应用程序的直接替代品。默认情况下，服务器在 [http://localhost:8000](http://localhost:8000) 启动。您可以使用 --host 和 --port 参数指定地址。服务器目前 1 次托管 1 个模型，并实现了诸如：[列出模型](https://platform.openai.com/docs/api-reference/models/list)、[创建聊天补全](https://platform.openai.com/docs/api-reference/chat/completions/create)和[创建补全](https://platform.openai.com/docs/api-reference/completions/create)等端点。


运行以下命令以启动 vLLM 服务器并使用 [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 模型：

```plain
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```


**注意：**

默认情况下，服务器使用存储在 tokenizer 中的预定义聊天模板。您可以参阅[聊天补全](#聊天补全)了解如何覆盖它。

 :::

This server can be queried in the same format as OpenAI API. For example, to list the models:

此服务器可以采用与 OpenAI API 相同的格式进行查询。例如，列出模型：

```plain
curl http://localhost:8000/v1/models
```


您可以传入参数 --api-key 或环境变量 VLLM_API_KEY，以启用服务器检查请求头中的 API 密钥。


### 使用 vLLM 的 OpenAI 补全 API

启动服务器后，您可以使用输入提示查询模型：

```plain
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```


由于此服务器与 OpenAI API 兼容，因此您可以将其作为使用 OpenAI API 的任何应用程序的直接替代品。例如，另一种查询服务器的方式是通过 openai Python 包：

```python
from openai import OpenAI


# Modify OpenAI's API key and API base to use vLLM's API server.
# 修改 OpenAI 的 API 密钥和 API 基础 URL 以使用 vLLM 的 API 服务器。
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(model="Qwen/Qwen2.5-1.5B-Instruct",
                                      prompt="San Francisco is a")
print("Completion result:", completion)
```
更详细的客户端示例可以在这里找到：[examples/online_serving/openai_completion_client.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_completion_client.py)
###

### 使用 vLLM 的 OpenAI 聊天补全 API

vLLM 设计上还支持 OpenAI 的聊天补全 API。通过聊天界面，用户可以与模型进行更加动态和互动的交流，支持来回对话并将其保存在聊天记录中。这种功能对于需要上下文理解或更详尽解释的任务来说非常有用。


您可以使用[创建聊天补全](https://platform.openai.com/docs/api-reference/chat/completions/create)端点与模型交互：

```plain
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```


或者，您可以使用 openai Python 包：

```python
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
# 设置 OpenAI 的 API 密钥和 API 基础 URL 以使用 vLLM 的 API 服务器。
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
)
print("Chat response:", chat_response)
```


## 注意力后端 (On Attention Backends)

目前，vLLM支持多种后端，以实现在不同平台和架构上高效的注意力计算。它会自动选择与您的系统和模型规格兼容的最优后端。


如果需要，您还可以通过配置环境变量 VLLM_ATTENTION_BACKEND 手动设置您选择的后端，可选值为：FLASH_ATTN、FLASHINFER 或 XFORMERS。


>**注意：**
>目前没有预构建的包含 Flash Infer 的 vllm 轮子，因此您必须先在环境中安装它。请参考 [Flash Infer 官方文档](https://docs.flashinfer.ai/)或查看 [Dockerfile](https://github.com/vllm-project/vllm/blob/main/Dockerfile) 获取安装说明。
