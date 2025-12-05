---
title: Openai 嵌入式客户端
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/online_serving/openai_embedding_client.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_embedding_client.py)

```python
# SPDX-License-Identifier: Apache-2.0

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
# 修改 OpenAI 的 API key 和 API base, 使用 vLLM 的 API 服务器。
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

responses = client.embeddings.create(
    input=[
        "Hello my name is",
        "The best thing about vLLM is that it supports many different models"
    ],
    model=model,
)

for data in responses.data:
    print(data.embedding)  # List of float of len 4096 # float 列表，大小 4096

```
