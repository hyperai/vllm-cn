---
title: vLLM 入门教程：零基础分步指南
---

在本教程中，将逐步展示如何配置和运行 vLLM，提供从安装到启动的完整入门指南。

[在线运行此教程](https://openbayes.com/console/public/tutorials/rXxb5fZFr29)

## 目录
- [一、安装 vLLM](#一、安装vLLM)
- [二、开始使用](#二、开始使用)
  - [2.1 模型准备](#2.1模型准备)
  - [2.2 离线推理](#2.2离线推理)
- [三、启动 vLLM 服务器](#三、启动vLLM服务器)
  - [3.1 主要参数设置](#3.1主要参数设置)
  - [3.2 启动命令行](#3.2启动命令行)
- [四、发出请求](#四、发出请求)
  - [4.1 使用 OpenAI 客户端](#4.1使用OpenAI客户端)
  - [4.2 使用 Curl 命令请求](#4.2使用Curl命令请求)

## 一、安装 vLLM

该教程基于 OpenBayes 云平台操作，该平台已完成 vllm==0.5.4 的安装。如果您在平台上操作，请跳过此步骤。如果您在本地部署，请按照以下步骤进行安装。

安装 vLLM 非常简单：

```bash
pip install vllm
```

请注意，vLLM 是使用 CUDA 12.1 编译的，因此您需要确保机器运行的是该版本的 CUDA。

检查 CUDA 版本，运行：

```bash
nvcc --version
```

如果您的 CUDA 版本不是 12.1，您可以安装与您当前 CUDA 版本兼容的 vLLM 版本（更多信息请参考安装说明），或者安装 CUDA 12.1。

## 二、开始使用

### 2.1 模型准备

#### 方法一：使用平台公共模型

首先，我们可以检查平台的公共模型是否已经存在。如果模型已上传到公共资源库，您可以直接使用。如果没有找到，则请参考方法二进行下载。

例如，平台已存放了 `Qwen-1_8B-Chat` 模型。以下是绑定模型的步骤（本教程已将此模型捆绑）。

![图片](/img/docs/02-tutorials/model.png)

![图片](/img/docs/02-tutorials/id.png)

![图片](/img/docs/02-tutorials/bangding.png)

#### 方法二：从 HuggingFace下载 或者 联系客服帮忙上传平台

大多数主流模型都可以在 HuggingFace 上找到，vLLM 支持的模型列表请参见官方文档： [vllm-supported-models](https://vllm.io/docs/supported_models)。

本教程将使用 `Qwen-1_8B-Chat` 进行测试。请按照以下步骤使用 Git LFS 下载模型：

```bash
apt install git-lfs -y
git lfs install
# 下载模型，将模型文件下载到当前文件夹
cd /input0/
git clone https://huggingface.co/Qwen/Qwen-1_8B-Chat
```

### 2.2 离线推理

vLLM 作为一个开源项目，可以通过其 Python API 执行 LLM 推理。以下是一个简单的示例，请将代码保存为 `offline_infer.py` 文件:

```python
from vllm import LLM, SamplingParams

# 输入几个问题
prompts = [
    "你好，你是谁？",
    "法国的首都在哪里？",
]

# 设置初始化采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# 加载模型，确保路径正确
llm = LLM(model="/input0/Qwen-1_8B-Chat/", trust_remote_code=True, max_model_len=4096)

# 展示输出结果
outputs = llm.generate(prompts, sampling_params)

# 打印输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

然后运行脚本:

```bash
python offline_infer.py
```

模型加载后，您将看到以下输出：

```python
Processed prompts: 100%|██████████| 2/2 [00:00<00:00, 10.23it/s, est. speed input: 56.29 toks/s, output: 225.14 toks/s]
Prompt: '你好，你是谁？', Generated text: '我是来自阿里云的大规模语言模型，我叫通义千问。'
Prompt: '法国的首都在哪里？', Generated text: '法国的首都是巴黎。'
```

## 三、启动 vLLM 服务器

要使用 vLLM 提供在线服务，您可以启动与 OpenAI API 兼容的服务器。成功启动后，您可以像使用 GPT 一样使用部署的模型。

### 3.1 主要参数设置

以下是启动 vLLM 服务器时常用的一些参数：

- `--model`：要使用的 HuggingFace 模型名称或路径（默认值：`facebook/opt-125m`）。
- `--host` 和 `--port`：指定服务器地址和端口。
- `--dtype`：模型权重和激活的精度类型。可能的值：`auto`、`half`、`float16`、`bfloat16`、`float`、`float32`。默认值：`auto`。
- `--tokenizer`：要使用的 HuggingFace 标记器名称或路径。如果未指定，默认使用模型名称或路径。
- `--max-num-seqs`：每次迭代的最大序列数。
- `--max-model-len`：模型的上下文长度，默认值自动从模型配置中获取。
- `--tensor-parallel-size`、`-tp`：张量并行副本数量（对于 GPU）。默认值：`1`。
- `--distributed-executor-backend=ray`：指定分布式服务的后端，可能的值：`ray`、`mp`。默认值：`ray`（当使用超过一个 GPU 时，自动设置为 `ray`）。

### 3.2 启动命令行

创建兼容 OpenAI API 接口的服务器。运行以下命令启动服务器：

```bash
python3 -m vllm.entrypoints.openai.api_server --model /input0/Qwen-1_8B-Chat/ --host 0.0.0.0 --port 8080 --dtype auto --max-num-seqs 32 --max-model-len 4096 --tensor-parallel-size 1 --trust-remote-code
```

成功启动后，您将看到类似以下的输出：

![图片](/img/docs/02-tutorials/start.png)

vLLM 现在可以作为实现 OpenAI API 协议的服务器进行部署，默认情况下它将在 `http://localhost:8080` 启动服务器。您可以通过 `--host` 和 `--port` 参数指定其他地址。

## 四、发出请求

在本教程中启动的 API 地址是 `http://localhost:8080`，您可以通过访问该地址来使用 API。`localhost` 指平台本机，`8080` 是 API 服务监听的端口号。

在工作空间右侧，API 地址将转发到本地 8080 服务，可以通过真实主机进行请求，如下图所示：

![图片](/img/docs/02-tutorials/api_path.png)

### 4.1 使用 OpenAI 客户端

在第四步中启动 vLLM 服务后，您可以通过 OpenAI 客户端调用 API。以下是一个简单的示例：

```python
# 注意：请先安装 openai
# pip install openai
from openai import OpenAI

# 设置 OpenAI API 密钥和 API 基础地址
openai_api_key = "EMPTY"  # 请替换为您的 API 密钥
openai_api_base = "http://localhost:8080/v1"  # 本地服务地址

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

models = client.models.list()
model = models.data[0].id
prompt = "描述一下北京的秋天"

# Completion API 调用
completion

 = client.completions.create(model=model, prompt=prompt)
res = completion.choices[0].text.strip()
print(f"Prompt: {prompt}\nResponse: {res}")
```

执行命令：

```bash
python api_infer.py
```

您将看到如下输出结果：

![图片](/img/docs/02-tutorials/res_api.png)

### 4.2 使用 Curl 命令请求

您也可以使用以下命令直接发送请求。在平台上访问时，输入以下命令：

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/input0/Qwen-1_8B-Chat/",
    "prompt": "描述一下北京的秋天",
    "max_tokens": 512
  }'
```

您将得到如下响应：

![图片](/img/docs/02-tutorials/curl_res.png)

如果您使用的是 OpenBayes 平台，输入以下命令：

```bash
curl https://hyperai-tutorials-****.gear-c1.openbayes.net/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/input0/Qwen-1_8B-Chat/",
    "prompt": "描述一下北京的秋天",
    "max_tokens": 512
  }'
```

响应结果如下：

![图片](/img/docs/02-tutorials/curl_res_local.png)


