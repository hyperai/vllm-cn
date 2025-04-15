---
title: OpenAI 补全客户端
---


源代码: [vllm-project/vllm](https://raw.githubusercontent.com/vllm-project/vllm/main/examples/openai_completion_client.py)

```python
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
# 设置 OpenAI 的 API key，和 API base 以使用 vLLM's API 服务
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    # 默认为 os.environ.get("OPENAI_API_KEY")

    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# Completion API

stream = False
completion = client.completions.create(
    model=model,
    prompt="A robot may not injure a human being",
    echo=False,
    n=2,
    stream=stream,
    logprobs=3)

print("Completion results:")
if stream:
    for c in completion:
        print(c)
else:
    print(completion)
```


