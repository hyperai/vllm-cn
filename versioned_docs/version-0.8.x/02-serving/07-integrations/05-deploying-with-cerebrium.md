---
title: 使用 Cerebrium 进行部署
---


![图片](/img/docs/02-07/05-Deploying-with-Cerebrium.png)

vLLM 可以通过 [Cerebrium](https://www.cerebrium.ai/) 在基于云的 GPU 计算机上运行，​​这是一个无服务器的 AI 基础设施平台，使公司可以更轻松地构建和部署基于 AI 的应用程序。


如需安装 Cerebrium 客户端，请运行：

```plain
pip install cerebrium
cerebrium login
```


接下来，运行以下命令来创建您的 Cerebrium 项目：

```plain
cerebrium init vllm-project
```


接下来，安装所需的软件包，请将以下内容添加到 cerebrium.toml 中：

```plain
[cerebrium.deployment]
docker_base_image_url = "nvidia/cuda:12.1.1-runtime-ubuntu22.04"


[cerebrium.dependencies.pip]
vllm = "latest"
```


接下来，让我们添加代码来处理您选择的 LLM 的推理 （本例为 *mistralai/Mistral-7B-Instruct-v0.1*），将以下代码添加到您的 main.py：

```python
from vllm import LLM, SamplingParams


llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1")


def run(prompts: list[str], temperature: float = 0.8, top_p: float = 0.95):


    sampling_params = SamplingParams(temperature=temperature, top_p=top_p)
    outputs = llm.generate(prompts, sampling_params)


    # Print the outputs.


    # 打印输出。


    results = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        results.append({"prompt": prompt, "generated_text": generated_text})


    return {"results": results}
```


然后，运行以下代码将其部署到云端：

```plain
cerebrium deploy
```


如果部署成功，您应该得到一个 CURL 命令返回，您可以根据该命令进行推理。只需记住以您正在调用的函数名称结束 url（在我们的例子中为 /run）：

```python
curl -X POST https://api.cortex.cerebrium.ai/v4/p-xxxxxx/vllm/run \
 -H 'Content-Type: application/json' \
 -H 'Authorization: <JWT TOKEN>' \
 --data '{
   "prompts": [
     "Hello, my name is",
     "The president of the United States is",
     "The capital of France is",
     "The future of AI is"
   ]
 }'
```


您应该得到如下响应：

```python
{
    "run_id": "52911756-3066-9ae8-bcc9-d9129d1bd262",
    "result": {
        "result": [
            {
                "prompt": "Hello, my name is",
                "generated_text": " Sarah, and I'm a teacher. I teach elementary school students. One of"
            },
            {
                "prompt": "The president of the United States is",
                "generated_text": " elected every four years. This is a democratic system.\n\n5. What"
            },
            {
                "prompt": "The capital of France is",
                "generated_text": " Paris.\n"
            },
            {
                "prompt": "The future of AI is",
                "generated_text": " bright, but it's important to approach it with a balanced and nuanced perspective."
            }
        ]
    },
    "run_time_ms": 152.53663063049316
}
```


您现在拥有了一个自动扩充端点，而您只需为您使用的计算资源付费！
