---
title: Openai Chat Completion Structured Outputs
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/online_serving/openai_chat_completion_structured_outputs.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_structured_outputs.py)

```python
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from openai import BadRequestError, OpenAI
from pydantic import BaseModel

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="-",
)

# Guided decoding by Choice (list of possible options)
# 使用 Choice 的引导式解码 (可能的选项列表)
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{
        "role": "user",
        "content": "Classify this sentiment: vLLM is wonderful!"
    }],
    extra_body={"guided_choice": ["positive", "negative"]},
)
print(completion.choices[0].message.content)

# Guided decoding by Regex
# 使用 Regex 的引导式解码
prompt = ("Generate an email address for Alan Turing, who works in Enigma."
          "End in .com and new line. Example result:"
          "alan.turing@enigma.com\n")

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={
        "guided_regex": "\w+@\w+\.com\n",
        "stop": ["\n"]
    },
)
print(completion.choices[0].message.content)


# Guided decoding by JSON using Pydantic schema
# 使用 Pydantic 模式的 JSON 引导式解码
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
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={"guided_json": json_schema},
)
print(completion.choices[0].message.content)

# Guided decoding by Grammar
# 使用 Grammar 的引导式解码
simplified_sql_grammar = """
    ?start: select_statement

    ?select_statement: "SELECT " column_list " FROM " table_name

    ?column_list: column_name ("," column_name)*

    ?table_name: identifier

    ?column_name: identifier

    ?identifier: /[a-zA-Z_][a-zA-Z0-9_]*/
"""

prompt = ("Generate an SQL query to show the 'username' and 'email'"
"from the 'users' table.")
completion = client.chat.completions.create(
model="Qwen/Qwen2.5-3B-Instruct",
messages=[{
"role": "user",
"content": prompt,
}],
extra_body={"guided_grammar": simplified_sql_grammar},
)
print(completion.choices[0].message.content)

# Extra backend options
# 额外后端选项
prompt = ("Generate an email address for Alan Turing, who works in Enigma."
          "End in .com and new line. Example result:"
          "alan.turing@enigma.com\n")

try:
    # The no-fallback option forces vLLM to use xgrammar, so when it fails
    # you get a 400 with the reason why
    # no-fallback 选项强制 vLLM 使用 xgrammar，因此当解析失败时，
    # 你会收到一个 400 错误，并附带失败原因。
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct",
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        extra_body={
            "guided_regex": "\w+@\w+\.com\n",
            "stop": ["\n"],
            "guided_decoding_backend": "xgrammar:no-fallback"
        },
    )
except BadRequestError as e:
    print("This error is expected:", e)

```
