---
title: Cohere Rerank Client
---

[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/online_serving/cohere_rerank_client.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/cohere_rerank_client.py)

```python
# SPDX-License-Identifier: Apache-2.0

"""
使用 OpenAI 入口点的 Rerank API 的示例，
该 API 兼容Cohere SDK:https://github.com/cohere-ai/cohere-python
run: vllm serve BAAI/bge-reranker-base
"""
import cohere

# cohere v1 client
# cohere v1 客户端
co = cohere.Client(base_url="http://localhost:8000", api_key="sk-fake-key")
rerank_v1_result = co.rerank(
    model="BAAI/bge-reranker-base",
    query="What is the capital of France?",
    documents=[
        "The capital of France is Paris", "Reranking is fun!",
        "vLLM is an open-source framework for fast AI serving"
    ])

print(rerank_v1_result)

# or the v2
# 或 V2
co2 = cohere.ClientV2("sk-fake-key", base_url="http://localhost:8000")

v2_rerank_result = co2.rerank(
    model="BAAI/bge-reranker-base",
    query="What is the capital of France?",
    documents=[
        "The capital of France is Paris", "Reranking is fun!",
        "vLLM is an open-source framework for fast AI serving"
    ])

print(v2_rerank_result)

```
