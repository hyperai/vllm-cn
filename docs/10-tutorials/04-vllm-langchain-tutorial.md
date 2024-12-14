---
title: 将 LangChain 与 vLLM 结合使用：完整教程
---

[在线运行此教程](https://openbayes.com/console/hyperai-tutorials/containers/ODfeIHjjXbW)

LangChain 是提供构建复杂操作链的工具，而 vLLM 专注于高效的模型推理。两者结合应用可以简化并加速智能 LLM 应用程序的开发。

在本教程中，我们将介绍如何将 LangChain 与 vLLM 结合使用，从设置到分布式推理和量化的所有内容。

## 目录
- [1. 安装和设置 vLLM](#1.安装和设置vLLM)
- [2. 配置 vLLM 以与 LangChain 配合使用](#2.配置vLLM以与LangChain配合使用)
- [3. 使用 LangChain 和 vLLM 创建链](#3.使用LangChain和vLLM创建链)
- [4. 利用多 GPU 推理进行扩展](#4.利用多GPU推理进行扩展)
- [5. 利用量化提高效率](#5.利用量化提高效率)
- [6. 结论](#6.结论)

## 1. 安装和设置 vLLM

vLLM 配置要求：

操作系统： Linux

Python 版本： Python >= 3.8

GPU 要求：计算能力 >= 7.0 的 GPU（例如 V100、T4、RTX20xx、A100、L4、H100）。

CUDA 版本： vLLM 使用 CUDA 12.1 编译。请确保您的系统正在运行此版本。

如果您没有运行 CUDA 12.1，您可以安装为您的 CUDA 版本编译的 vLLM 版本或将您的 CUDA 升级到版本 12.1。

在继续之前，建议执行一些基本检查以确保一切都安装正确。您可以通过运行以下命令来验证 PyTorch 是否与 CUDA 一起使用：

```python
# Ensure torch is working with CUDA, this should print: True
python -c 'import torch; print(torch.cuda.is_available())'
```
vLLM 是一个 Python 库，还包含预编译的 C++ 和 CUDA (12.1) 二进制文件。但是，如果您需要 CUDA 11.8，则可以使用以下命令安装兼容版本：

```python
# Install vLLM with CUDA 11.8
export VLLM_VERSION=0.6.1.post1
export PYTHON_VERSION=310
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

### Docker 安装
对于那些在构建 vLLM 或处理 CUDA 兼容性时遇到问题的人，建议使用 NVIDIA PyTorch Docker 映像。它提供了一个预配置的环境，其中包含正确版本的 CUDA 和其他依赖项：

```python 
# Use `--ipc=host` to ensure the shared memory is sufficient
docker run --gpus all -it --rm --ipc=host nvcr.io/nvidia/pytorch:23.10-py3
```
集成过程最终从安装所需的软件包开始。我们建议将 vLLM 升级到最新版本，以避免兼容性问题并受益于最新的改进和功能。

```python
pip install --upgrade --quiet vllm -q
pip install langchain langchain_community -q
```
本教程已经安装 vllm==0.6.4，只需将 langchain 相关包安装完毕。

```
!pip install -U langchain langchain_community -q
```


## 2. 配置 vLLM 以与 LangChain 配合使用
现在依赖项已安装完毕，我们可以设置 vLLM 并将其连接到 LangChain。为此，我们将从 LangChain 社区集成中导入 VLLM。下面的示例演示了如何使用 vLLM 库初始化模型并将其与 LangChain 集成。


```
import gc
import ctypes
import torch
def clean_memory(deep=False):
    gc.collect()
    if deep:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()
```

```
from langchain_community.llms import VLLM

# Initializing the vLLM model
llm = VLLM(
    model="/input0/Qwen2.5-1.5B-Instruct",
    trust_remote_code=True,  # mandatory for Hugging Face models
    max_new_tokens=128,
    top_k=10,
    top_p=0.95,
    temperature=0.8,
)

# Running a simple query
print(llm.invoke("What are the most popular Halloween Costumes?"))
```

以下是使用 vLLM 与 LangChain 时需要考虑的参数列表：

| 参数名称              | 描述                                                                                               |
|---------------------|----------------------------------------------------------------------------------------------------|
| 模型                  | 要使用的 Hugging Face Transformers 模型的名称或路径。                                                  |
| top_k               | 将采样池限制为前 k 个 token，以提高多样性。默认值为 -1。                                               |
| top_p               | 使用累积概率来确定要考虑哪些标记，从而支持更一致的输出。默认值为 1.0。                                 |
| 信任远程代码           | 允许模型执行远程代码，对某些 Hugging Face 模型有用。默认值为 False。                                  |
| 温度                  | 控制采样的随机性，值越高，输出越多样化。默认值为 1.0。                                                |
| 最大新令牌数            | 指定每个输出序列生成的最大标记数。默认值为 512。                                                     |
| 回调                  | 添加到运行跟踪的回调，对于在生成期间添加日志记录或监控功能很有用。                                     |
| 标签                  | 添加到运行跟踪的标签可以方便进行分类和调试。                                                          |
| tensor_parallel_size | 用于分布式张量并行执行的 GPU 数量。默认值为 1。                                                      |
| 使用光束搜索            | 是否使用集束搜索而不是采样来生成更优化的序列。默认值为 False。                                        |
| 复制代码               | 保存对 vLLM LLM 调用有效的未明确指定的附加参数。                                                     |

在此示例中，我们加载 `Qwen2.5-1.5B-Instruct` 模型并配置`max_new_tokens`、`top_k`和 等参数`temperature`。这些设置会影响模型生成文本的方式。

## 3. 使用 LangChain 和 vLLM 创建链

LangChain 的核心功能之一是能够创建操作链，从而实现更复杂的交互。我们可以轻松地将 vLLM 模型集成到 LLMChain 中，从而提供更大的灵活性。


```
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# Defining a prompt template for our LLMChain
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

# Creating an LLMChain with vLLM
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Testing the LLMChain
question = "Who was the US president in the year the first Pokemon game was released?"
print(llm_chain.invoke(question))
```

## 4. 利用多 GPU 推理进行扩展
如果您正在使用本地托管的大型模型，则可能需要利用多个 GPU 进行推理。特别是对于需要同时处理许多请求的高吞吐量系统。vLLM 允许这样做：分布式张量并行推理，以帮助扩展操作。

要运行多 GPU 推理，请 tensor_parallel_size 在初始化 VLLM 类时使用该参数。

```
del llm

clean_memory(deep=True)
```

```
from langchain_community.llms import VLLM

# Running inference on multiple GPUs
llm = VLLM(
    model="/input0/Qwen2.5-1.5B-Instruct",
    tensor_parallel_size=1,  # using 1 GPUs
    trust_remote_code=True,
)

print(llm.invoke("What is the future of AI?"))
```

对于较大的模型，强烈建议使用此方法，因为它的计算量很大，而且在单个 GPU 上运行速度太慢。

## 5. 利用量化提高效率
量化是一种通过减少内存使用和加快计算来提高语言模型性能的有效技术。

vLLM 支持 AWQ 量化格式。要启用它，请通过参数传递量化选项 vllm_kwargs。量化允许在资源受限的环境（例如边缘设备或较旧的 GPU）中部署 LLM，而不会牺牲太多准确性。

```
del llm

clean_memory(deep=True)
```

```
llm_q = VLLM(
    model="/input0/Qwen2.5-3B-Instruct-AWQ",
    trust_remote_code=True,
    max_new_tokens=512,
    vllm_kwargs={"quantization": "awq"},
)
# Running a simple query
print(llm_q.invoke("What are the most popular Halloween Costumes?"))
```

在此示例中，Qwen2.5-3B-Instruct-AWQ模型已量化以实现最佳性能。在将应用程序部署到生产环境（成本和资源效率至关重要）时，此功能尤其有用。

## 6. 结论
通过利用分布式 GPU 支持、先进的量化技术和保持 API 兼容性，您可以创建不仅提供卓越性能而且还能灵活满足不同业务需求的系统。

当您继续使用 LangChain 和 vLLM 进行大型语言模型研究时，请务必记住，持续优化和监控是实现最佳效率的关键。

例如，vLLM 的 CUDA 优化内核和连续批处理策略可以显著减少响应时间。

然而，在生产系统中，特别是面向用户的系统（如聊天机器人）中，监控实时推理延迟至关重要。
