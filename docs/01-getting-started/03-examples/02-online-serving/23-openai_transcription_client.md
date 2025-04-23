---
title: Openai Transcription 客户端
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/online_serving/openai_transcription_client.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_transcription_client.py)

```python
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json

import httpx
from openai import OpenAI

from vllm.assets.audio import AudioAsset

mary_had_lamb = AudioAsset('mary_had_lamb').get_local_path()
winning_call = AudioAsset('winning_call').get_local_path()

# Modify OpenAI's API key and API base to use vLLM's API server.
# 修改 OpenAI 的 API key 和 API base, 使用 vLLM 的 API 服务器。
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def sync_openai():
    with open(str(mary_had_lamb), "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="openai/whisper-small",
            language="en",
            response_format="json",
            temperature=0.0)
        print("transcription result:", transcription.text)


sync_openai()


# OpenAI Transcription API client does not support streaming.
# OpenAI Transcription API 客户端不支持流。
async def stream_openai_response():
    data = {
        "language": "en",
        'stream': True,
        "model": "openai/whisper-large-v3",
    }
    url = openai_api_base + "/audio/transcriptions"
    print("transcription result:", end=' ')
    async with httpx.AsyncClient() as client:
        with open(str(winning_call), "rb") as f:
            async with client.stream('POST', url, files={'file': f},
                                     data=data) as response:
                async for line in response.aiter_lines():
                    # Each line is a JSON object prefixed with 'data: '
                    # 每行都是一个带有 'data: ' 的 JSON 对象。
                    if line:
                        if line.startswith('data: '):
                            line = line[len('data: '):]
                        # Last chunk, stream ends
                        # 最后一块，流结束
                        if line.strip() == '[DONE]':
                            break
                        # Parse the JSON response
                        # 解析 JSON 响应
                        chunk = json.loads(line)
                        # Extract and print the content
                        # 提取并打印内容
                        content = chunk['choices'][0].get('delta',
                                                          {}).get('content')
                        print(content, end='')


# Run the asynchronous function
# 运行异步功能
asyncio.run(stream_openai_response())

```
