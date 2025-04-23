---

title: dstack

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


![图片](/img/docs/v1-deployment/03-dstack_1.png)

vLLM 可以通过 [dstack](https://dstack.ai/) 在基于云的 GPU 计算机上运行，​​dstack 是一个用于在任何云上运行 LLM 的开源框架。本教程假设您已在云环境中配置凭据、网关和 GPU 配额。


要安装 dstack 客户端，请运行: 

```plain
pip install "dstack[all]
dstack server
```


接下来，要配置 dstack 项目，请运行：

```plain
mkdir -p vllm-dstack
cd vllm-dstack
dstack init
```


接下来，要使用您选择的 LLM 来配置 VM 实例 （本示例为 *NousResearch/Llama-2-7b-chat-hf*），请为 dstack *Service* 创建以下 *serve.dstack.yml* 文件: 

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


然后，运行以下 CLI 进行配置: 

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


**注意**

dstack 使用 dstack 的令牌对网关上的身份验证自动处理。同时，如果您不想配置网关，您可以配置 dstack *Task* 而不是 *Service*。 *任务* 仅用于开发目的。如果您想了解更多有关如何使用 dstack 提供 vLLM 服务的实践材料，请查看[此存储库](https://github.com/dstackai/dstack-examples/tree/main/deployment/vllm)。

