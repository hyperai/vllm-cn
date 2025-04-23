---
title: 嵌入多通道客户端的 OpenAI 聊天工具
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/online_serving/openai_chat_embedding_client_for_multimodal.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_embedding_client_for_multimodal.py)

```python
# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import io

import requests
from PIL import Image

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


def vlm2vec():
    response = requests.post(
        "http://localhost:8000/v1/embeddings",
        json={
            "model":
            "TIGER-Lab/VLM2Vec-Full",
            "messages": [{
                "role":
                "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type": "text",
                        "text": "Represent the given image."
                    },
                ],
            }],
            "encoding_format":
            "float",
        },
    )
    response.raise_for_status()
    response_json = response.json()

    print("Embedding output:", response_json["data"][0]["embedding"])


def dse_qwen2_vl(inp: dict):
    # Embedding an Image
    # 嵌入图像
    if inp["type"] == "image":
        messages = [{
            "role":
            "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": inp["image_url"],
                }
            }, {
                "type": "text",
                "text": "What is shown in this image?"
            }]
        }]
    # Embedding a Text Query
    # 嵌入文本查询
    else:
        # MrLight/dse-qwen2-2b-mrl-v1 requires a placeholder image
        # of the minimum input size
        # MRLIGHT/DSE-QWEN2-2-2B-MRL-V1需要最小输入大小的占位符图像
        buffer = io.BytesIO()
        image_placeholder = Image.new("RGB", (56, 56))
        image_placeholder.save(buffer, "png")
        buffer.seek(0)
        image_placeholder = base64.b64encode(buffer.read()).decode('utf-8')
        messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_placeholder}",
                    }
                },
                {
                    "type": "text",
                    "text": f"Query: {inp['content']}"
                },
            ]
        }]

    response = requests.post(
        "http://localhost:8000/v1/embeddings",
        json={
            "model": "MrLight/dse-qwen2-2b-mrl-v1",
            "messages": messages,
            "encoding_format": "float",
        },
    )
    response.raise_for_status()
    response_json = response.json()

    print("Embedding output:", response_json["data"][0]["embedding"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Script to call a specified VLM through the API. Make sure to serve "
        "the model with --task embed before running this.")
    parser.add_argument("--model",
                        type=str,
                        choices=["vlm2vec", "dse_qwen2_vl"],
                        required=True,
                        help="Which model to call.")
    args = parser.parse_args()

    if args.model == "vlm2vec":
        vlm2vec()
    elif args.model == "dse_qwen2_vl":
        dse_qwen2_vl({
            "type": "image",
            "image_url": image_url,
        })
        dse_qwen2_vl({
            "type": "text",
            "content": "What is the weather like today?",
        })

```
