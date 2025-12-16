---
title: Openai 交叉编码器得分
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/online_serving/openai_cross_encoder_score.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_cross_encoder_score.py)

```python
# SPDX-License-Identifier: Apache-2.0
"""
Example online usage of Score API.
在线使用分数 API 示例。

Run `vllm serve <model> --task score` to start up the server in vLLM.
"""
import argparse
import pprint

import requests


def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="BAAI/bge-reranker-v2-m3")

    args = parser.parse_args()
    api_url = f"http://{args.host}:{args.port}/score"
    model_name = args.model

    text_1 = "What is the capital of Brazil?"
    text_2 = "The capital of Brazil is Brasilia."
    prompt = {"model": model_name, "text_1": text_1, "text_2": text_2}
    score_response = post_http_request(prompt=prompt, api_url=api_url)
    print("Prompt when text_1 and text_2 are both strings:")
    pprint.pprint(prompt)
    print("Score Response:")
    pprint.pprint(score_response.json())

    text_1 = "What is the capital of France?"
    text_2 = [
        "The capital of Brazil is Brasilia.", "The capital of France is Paris."
    ]
    prompt = {"model": model_name, "text_1": text_1, "text_2": text_2}
    score_response = post_http_request(prompt=prompt, api_url=api_url)
    print("Prompt when text_1 is string and text_2 is a list:")
    pprint.pprint(prompt)
    print("Score Response:")
    pprint.pprint(score_response.json())

    text_1 = [
        "What is the capital of Brazil?", "What is the capital of France?"
    ]
    text_2 = [
        "The capital of Brazil is Brasilia.", "The capital of France is Paris."
    ]
    prompt = {"model": model_name, "text_1": text_1, "text_2": text_2}
    score_response = post_http_request(prompt=prompt, api_url=api_url)
    print("Prompt when text_1 and text_2 are both lists:")
    pprint.pprint(prompt)
    print("Score Response:")
    pprint.pprint(score_response.json())

```
