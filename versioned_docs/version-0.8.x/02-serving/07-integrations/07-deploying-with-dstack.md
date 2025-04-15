---
title: 使用 dstack 进行部署
---


![图片](/img/docs/02-07/07-Deploying-with-dstack.png)

vLLM 可以通过 [dstack](https://dstack.ai/) 在基于云的 GPU 计算机上运行，​​dstack 是一个用于在任何云上运行 LLM 的开源框架。本教程假设您已在云环境中配置凭据、网关和 GPU 配额。


请运行一下代码，安装 dstack 客户端：

```plain
pip install "dstack[all]
dstack server
```


接下来，配置 dstack 项目，请运行：

```plain
mkdir -p vllm-dstack
cd vllm-dstack
dstack init
```


接下来，要使用您选择的 LLM 来配置虚拟机实例（本示例为 *NousResearch/Llama-2-7b-chat-hf*），请为 dstack *Service* 创建以下 *serve.dstack.yml* 文件：

```yaml
type: service


python: "3.11"
env:
    - MODEL=NousResearch/Llama-2-7b-chat-hf
port: 8000
resources:
    gpu: 24GB
commands:
    - pip install vllm
    - vllm serve $MODEL --port 8000
model:
    format: openai
    type: chat
    name: NousResearch/Llama-2-7b-chat-hf
```


然后，运行以下 CLI 进行配置：

```plain
dstack run . -f serve.dstack.yml


⠸ Getting run plan...
 Configuration  serve.dstack.yml
 Project        deep-diver-main
 User           deep-diver
 Min resources  2..xCPU, 8GB.., 1xGPU (24GB)
 Max price      -
 Max duration   -
 Spot policy    auto
 Retry policy   no


 #  BACKEND  REGION       INSTANCE       RESOURCES                               SPOT  PRICE
 1  gcp   us-central1  g2-standard-4  4xCPU, 16GB, 1xL4 (24GB), 100GB (disk)  yes   $0.223804
 2  gcp   us-east1     g2-standard-4  4xCPU, 16GB, 1xL4 (24GB), 100GB (disk)  yes   $0.223804
 3  gcp   us-west1     g2-standard-4  4xCPU, 16GB, 1xL4 (24GB), 100GB (disk)  yes   $0.223804
    ...
 Shown 3 of 193 offers, $5.876 max


Continue? [y/n]: y
⠙ Submitting run...
⠏ Launching spicy-treefrog-1 (pulling)
spicy-treefrog-1 provisioning completed (running)
Service is published at ...
```


配置完成后，您可以使用 OpenAI SDK 与模型进行交互：

```python
from openai import OpenAI


client = OpenAI(
    base_url="https://gateway.<gateway domain>",
    api_key="<YOUR-DSTACK-SERVER-ACCESS-TOKEN>"
)


completion = client.chat.completions.create(
    model="NousResearch/Llama-2-7b-chat-hf",
    messages=[
        {
            "role": "user",
            "content": "Compose a poem that explains the concept of recursion in programming.",
        }
    ]
)


print(completion.choices[0].message.content)
```


注意：

dstack 会自动使用 dstack 的 tokens 在网关上处理认证。同时，如果您不想配置网关，您可以配置 dstack *Task* 而不是 *Service*。 *任务*仅用于开发目的。如果您想了解更多有关如何使用 dstack 提供 vLLM 服务的实践材料，请查看[此存储库](https://github.com/dstackai/dstack-examples/tree/main/deployment/vllm)。
