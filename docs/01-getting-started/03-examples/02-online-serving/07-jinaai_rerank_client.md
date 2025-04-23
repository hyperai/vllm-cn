---
title: Jinaai Rerank Client
---

[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/online_serving/jinaai_rerank_client.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/jinaai_rerank_client.py)

```python
# SPDX-License-Identifier: Apache-2.0

"""
使用 OpenAI 入口点的 Rerank API 的示例，
该 API 兼容 Jina 和 Cohere https://jina.ai/reranker

run: vllm serve BAAI/bge-reranker-base
"""
import json

import requests

url = "http://127.0.0.1:8000/rerank"

headers = {"accept": "application/json", "Content-Type": "application/json"}

data = {
    "model":
    "BAAI/bge-reranker-base",
    "query":
    "What is the capital of France?",
    "documents": [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.", "Horses and cows are both animals"
    ]
}
response = requests.post(url, headers=headers, json=data)

# Check the response
# 检查响应
if response.status_code == 200:
    print("Request successful!")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Request failed with status code: {response.status_code}")
    print(response.text)

```
