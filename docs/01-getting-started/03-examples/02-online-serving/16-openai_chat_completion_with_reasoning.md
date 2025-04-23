---
title: Openai Chat Completion With Reasoning
---

[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/online_serving/openai_chat_completion_with_reasoning.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_with_reasoning.py)

```python
# SPDX-License-Identifier: Apache-2.0
"""
本示例演示如何从推理模型如 DeepSeekR1 生成聊天完成。

要运行此示例，您需要启动 vLLM 服务器，并启用推理解析器：


vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
--enable-reasoning --reasoning-parser deepseek_r1

# 本示例演示了如何使用 OpenAI Python 客户端库从推理模型生成聊天完成。
"""

from openai import OpenAI

# 修改 OpenAI 的 API key 和 API base, 使用 vLLM 的 API 服务器。
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# 第1轮
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]

# 对于 granite，添加:`extra_body = { "chat_template_kwargs":{ "thinky":true}}'
response = client.chat.completions.create(model=model, messages=messages)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print("reasoning_content for Round 1:", reasoning_content)
print("content for Round 1:", content)

# 第2轮
messages.append({"role": "assistant", "content": content})
messages.append({
    "role": "user",
    "content": "How many Rs are there in the word 'strawberry'?",
})
response = client.chat.completions.create(model=model, messages=messages)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print("reasoning_content for Round 2:", reasoning_content)
print("content for Round 2:", content)

```
