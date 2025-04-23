---
title: OpenAI 聊天完成结构化输出与推理
---

[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/online_serving/openai_chat_completion_structured_outputs_with_reasoning.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_structured_outputs_with_reasoning.py)

```python
# SPDX-License-Identifier: Apache-2.0

"""  
本示例展示如何使用推理模型（如 DeepSeekR1）生成结构化输出。  
思考过程不会受到用户提供的 JSON 模式的引导，只有最终输出会被结构化。  

要运行此示例，需要使用推理解析器启动 vLLM 服务器：  

```bash  
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \  
--enable-reasoning --reasoning-parser deepseek_r1  
```
本示例演示如何使用 OpenAI Python 客户端库，从推理模型生成聊天补全内容。
"""

from enum import Enum

from openai import OpenAI
from pydantic import BaseModel

# 修改 OpenAI 的 API key 和 API base, 使用 vLLM 的 API 服务器。
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# Guided decoding by Regex
# 使用 Regex 的引导式解码
prompt = ("What is the capital of France?")

completion = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={
        "guided_regex": "(Paris|London)",
    },
)
print("reasoning_content: ", completion.choices[0].message.reasoning_content)
print("content: ", completion.choices[0].message.content)


class People(BaseModel):
    name: str
    age: int


json_schema = People.model_json_schema()

prompt = ("Generate a JSON with the name and age of one random person.")
completion = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={"guided_json": json_schema},
)
print("reasoning_content: ", completion.choices[0].message.reasoning_content)
print("content: ", completion.choices[0].message.content)

# 使用 pydantic 模式的 JSON 引导式解码
class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType


json_schema = CarDescription.model_json_schema()

prompt = ("Generate a JSON with the brand, model and car_type of"
          "the most iconic car from the 90's")
completion = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={"guided_json": json_schema},
)
print("reasoning_content: ", completion.choices[0].message.reasoning_content)
print("content: ", completion.choices[0].message.content)

# 使用 Grammar 的引导式解码
simplified_sql_grammar = """
    ?start: select_statement

    ?select_statement: "SELECT " column_list " FROM " table_name

    ?column_list: column_name ("," column_name)*

    ?table_name: identifier

    ?column_name: identifier

    ?identifier: /[a-zA-Z_][a-zA-Z0-9_]*/
"""

# This may be very slow https://github.com/vllm-project/vllm/issues/12122
# 这可能很慢
prompt = ("Generate an SQL query to show the 'username' and 'email'"
"from the 'users' table.")
completion = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={"guided_grammar": simplified_sql_grammar},
)
print("reasoning_content: ", completion.choices[0].message.reasoning_content)
print("content: ", completion.choices[0].message.content)

```
