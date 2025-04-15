---
title: OpenAI 音频 API 客户端
---


源代码: [vllm-project/vllm](https://raw.githubusercontent.com/vllm-project/vllm/main/examples/openai_audio_api_client.py)

```python
"""An example showing how to use vLLM to serve VLMs.

Launch the vLLM server with the following command:
vllm serve fixie-ai/ultravox-v0_3
示例展示如何使用 vLLM 来服务 VLMs 


启动 vLLM 服务器的命令如下：
vllm serve fixie-ai/ultravox-v0_3

"""
import base64

import requests
from openai import OpenAI

from vllm.assets.audio import AudioAsset

# Modify OpenAI's API key and API base to use vLLM's API server.
# 设置 OpenAI 的 API key，和 API base 以使用 vLLM's API 服务

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# Any format supported by librosa is supported
# 支持其他 librosa 支持的格式

audio_url = AudioAsset("winning_call").url

# Use audio url in the payload
# 在装载中使用音频 url

chat_completion_from_url = client.chat.completions.create(
    messages=[{
        "role":
        "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this audio?"
            },
            {
                "type": "audio_url",
                "audio_url": {
                    "url": audio_url
                },
            },
        ],
    }],
    model=model,
    max_tokens=64,
)

result = chat_completion_from_url.choices[0].message.content
print(f"Chat completion output:{result}")


# Use base64 encoded audio in the payload
# 在装载中使用 base64 编码 音频

def encode_audio_base64_from_url(audio_url: str) -> str:
    """Encode an audio retrieved from a remote url to base64 format."""

    with requests.get(audio_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')

    return result


audio_base64 = encode_audio_base64_from_url(audio_url=audio_url)
chat_completion_from_base64 = client.chat.completions.create(
    messages=[{
        "role":
        "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this audio?"
            },
            {
                "type": "audio_url",
                "audio_url": {
                    # Any format supported by librosa is supported
                    "url": f"data:audio/ogg;base64,{audio_base64}"
                },
            },
        ],
    }],
    model=model,
    max_tokens=64,
)

result = chat_completion_from_base64.choices[0].message.content
print(f"Chat completion output:{result}")
```


