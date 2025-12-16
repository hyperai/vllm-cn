---
title: Chat With Tools
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/chat_with_tools.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/chat_with_tools.py)

````python
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa
import json
import random
import string

from vllm import LLM
from vllm.sampling_params import SamplingParams

# 此脚本是用于函数调用的离线演示
#
# 如果要运行服务器/客户端设置，请按以下代码:
#
# - 服务器:
#
# ```bash
# vllm serve mistralai/Mistral-7B-Instruct-v0.3 --tokenizer-mode mistral --load-format mistral --config-format mistral
# ```
#
# - 客户端:
#
# ```bash
# curl --location 'http://<your-node-url>:8000/v1/chat/completions' \
# --header 'Content-Type: application/json' \
# --header 'Authorization: Bearer token' \
# --data '{
#     "model": "mistralai/Mistral-7B-Instruct-v0.3"
#     "messages": [
#       {
#         "role": "user",
#         "content": [
#             {"type" : "text", "text": "Describe this image in detail please."},
#             {"type": "image_url", "image_url": {"url": "https://s3.amazonaws.com/cms.ipressroom.com/338/files/201808/5b894ee1a138352221103195_A680%7Ejogging-edit/A680%7Ejogging-edit_hero.jpg"}},
#             {"type" : "text", "text": "and this one as well. Answer in French."},
#             {"type": "image_url", "image_url": {"url": "https://www.wolframcloud.com/obj/resourcesystem/images/a0e/a0ee3983-46c6-4c92-b85d-059044639928/6af8cfb971db031b.png"}}
#         ]
#       }
#     ]
#   }'
# ```
#
# 用法:
#     python demo.py simple
#     python demo.py advanced

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# 或切换到 "Mistralai/Mistral-Nemo-Instruct-2407"
# 或 "Mistralai/Mistral-Large-Instruct-2407"
# 或具有功能通话能力的任何其他 Mistral 模型

sampling_params = SamplingParams(max_tokens=8192, temperature=0.0)
llm = LLM(model=model_name,
          tokenizer_mode="mistral",
          config_format="mistral",
          load_format="mistral")


def generate_random_id(length=9):
    characters = string.ascii_letters + string.digits
    random_id = ''.join(random.choice(characters) for _ in range(length))
    return random_id


# 模拟可以调用的 API
def get_current_weather(city: str, state: str, unit: 'str'):
    return (f"The weather in {city}, {state} is 85 degrees {unit}. It is "
            "partly cloudly, with highs in the 90's.")


tool_funtions = {"get_current_weather": get_current_weather}

tools = [{
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
                    "The city to find the weather for, e.g. 'San Francisco'"
                },
                "state": {
                    "type":
                    "string",
                    "description":
                    "the two-letter abbreviation for the state that the city is"
                    " in, e.g. 'CA' which would mean 'California'"
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["city", "state", "unit"]
        }
    }
}]

messages = [{
    "role":
    "user",
    "content":
    "Can you tell me what the temperate will be in Dallas, in fahrenheit?"
}]

outputs = llm.chat(messages, sampling_params=sampling_params, tools=tools)
output = outputs[0].outputs[0].text.strip()

# 附加助手消息
messages.append({
    "role": "assistant",
    "content": output,
})

# 现在让我们实际上解析并执行模型的输出，以模拟 API 调用
# 上面定义的功能
tool_calls = json.loads(output)
tool_answers = [
    tool_funtions[call['name']](**call['arguments']) for call in tool_calls
]

# 附加答案到工具消息中，让 LLM 给您答案
messages.append({
    "role": "tool",
    "content": "\n\n".join(tool_answers),
    "tool_call_id": generate_random_id(),
})

outputs = llm.chat(messages, sampling_params, tools=tools)

print(outputs[0].outputs[0].text.strip())
# yields
# 结果
#   'The weather in Dallas, TX is 85 degrees fahrenheit. '
#   'It is partly cloudly, with highs in the 90's.'

````
