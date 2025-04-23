---
title: OpenAI Chat Completion Client With Tools Required
---

[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/online_serving/openai_chat_completion_client_with_tools_required.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_client_with_tools_required.py)

```python

# SPDX-License-Identifier: Apache-2.0
"""
要运行此示例，可以启动 vLLM 服务器，但不使用任何特定标志：

```

```bash
VLLM_USE_V1=0 vllm serve unsloth/Llama-3.2-1B-Instruct \
    --guided-decoding-backend outlines
```

```
此示例演示如何使用 OpenAI Python 客户端库生成聊天完成。
"""

from openai import OpenAI

# 修改 OpenAI 的 API 密钥和 API 基以使用 vLLM 的 API 服务器。

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type":
                        "string",
                        "description":
                        "The city to find the weather for"
                        ", e.g. 'San Francisco'",
                    },
                    "state": {
                        "type":
                        "string",
                        "description":
                        "the two-letter abbreviation for the state that the "
                        "city is in, e.g. 'CA' which would mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Get the weather forecast for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type":
                        "string",
                        "description":
                        "The city to get the forecast for, e.g. 'New York'",
                    },
                    "state": {
                        "type":
                        "string",
                        "description":
                        "The two-letter abbreviation for the state, e.g. 'NY'",
                    },
                    "days": {
                        "type":
                        "integer",
                        "description":
                        "Number of days to get the forecast for (1-7)",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "days", "unit"],
            },
        },
    },
]

messages = [
    {
        "role": "user",
        "content": "Hi! How are you doing today?"
    },
    {
        "role": "assistant",
        "content": "I'm doing well! How can I help you?"
    },
    {
        "role":
        "user",
        "content":
        "Can you tell me what the current weather is in Dallas \
            and the forecast for the next 5 days, in fahrenheit?",
    },
]

chat_completion = client.chat.completions.create(
    messages=messages,
    model=model,
    tools=tools,
    tool_choice="required",
    stream=True  # Enable streaming response
)

for chunk in chat_completion:
    if chunk.choices and chunk.choices[0].delta.tool_calls:
        print(chunk.choices[0].delta.tool_calls)

chat_completion = client.chat.completions.create(messages=messages,
                                                 model=model,
                                                 tools=tools,
                                                 tool_choice="required")

print(chat_completion.choices[0].message.tool_calls)

```
