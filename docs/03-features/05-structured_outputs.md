---
title: 结构化输出
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM 支持使用 [outlines](https://github.com/dottxt-ai/outlines)、[lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer) 或 [xgrammar](https://github.com/mlc-ai/xgrammar) 作为引导解码的后端生成结构化输出。本文档展示了一些可用于生成结构化输出的不同选项的示例。

## 在线服务（OpenAI API）

您可以使用 OpenAI 的 [Completions](https://platform.openai.com/docs/api-reference/completions) 和 [Chat](https://platform.openai.com/docs/api-reference/chat) API 生成结构化输出。

支持以下参数，这些参数必须作为额外参数添加：

- `guided_choice`：输出将是选项之一。
- `guided_regex`：输出将遵循正则表达式模式。
- `guided_json`：输出将遵循 JSON 模式。
- `guided_grammar`：输出将遵循上下文无关语法。
- `guided_whitespace_pattern`：用于覆盖引导 JSON 解码的默认空白模式。
- `guided_decoding_backend`：用于选择要使用的引导解码后端。

您可以在 [OpenAI 兼容服务器](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#openai-compatible-server) 页面上查看支持的完整参数列表。

现在让我们看一个每个案例的示例，从最简单的 `guided_choice` 开始：

```plain
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="-",
)


completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
    ],
    extra_body={"guided_choice": ["positive", "negative"]},
)
print(completion.choices[0].message.content)
```

下一个示例展示了如何使用 `guided_regex`。目标是生成一个电子邮件地址，给定一个简单的正则表达式模板：

```plain
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Generate an example email address for Alan Turing, who works in Enigma. End in .com and new line. Example result: alan.turing@enigma.com\n",
        }
    ],
    extra_body={"guided_regex": "\w+@\w+\.com\n", "stop": ["\n"]},
)
print(completion.choices[0].message.content)
```

结构化文本生成中最相关的功能之一是生成具有预定义字段和格式的有效 JSON。为此，我们可以以两种不同的方式使用 `guided_json` 参数：

- 直接使用 [JSON Schema](https://json-schema.org/)。
- 定义一个 [Pydantic 模型](https://docs.pydantic.dev/latest/)，然后从中提取 JSON Schema（这通常是一种更简单的选择）。

下一个示例展示了如何使用带有 Pydantic 模型的 `guided_json` 参数：

```plain
from pydantic import BaseModel
from enum import Enum


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


completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
        }
    ],
    extra_body={"guided_json": json_schema},
)
print(completion.choices[0].message.content)
```

> 提示
>
> 虽然并非严格必要，但通常在提示中指示需要生成 JSON 以及 LLM 应如何填充字段可以显著改善结果。

最后是 `guided_grammar`，它可能是最难使用的一项功能，但非常强大，因为它允许我们定义完整的语言，如 SQL 查询。它通过使用上下文无关的 EBNF 语法来工作，例如，我们可以使用它来定义简化 SQL 查询的特定格式，如下例所示：

```plain
simplified_sql_grammar = """
    ?start: select_statement


    ?select_statement: "SELECT " column_list " FROM " table_name


    ?column_list: column_name ("," column_name)*


    ?table_name: identifier


    ?column_name: identifier


    ?identifier: /[a-zA-Z_][a-zA-Z0-9_]*/
"""


completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Generate an SQL query to show the 'username' and 'email' from the 'users' table.",
        }
    ],
    extra_body={"guided_grammar": simplified_sql_grammar},
)
print(completion.choices[0].message.content)
```

完整示例：[examples/online_serving/openai_chat_completion_structured_outputs.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_structured_outputs.py)

## 实验性自动解析（OpenAI API）

本节介绍了 OpenAI 对 `client.chat.completions.create()` 方法的 beta 包装器，该包装器提供了与 Python 特定类型的更丰富集成。

在撰写本文时（`openai==1.54.4`），这是 OpenAI 客户端库中的“beta”功能。代码参考可以在[此处](https://github.com/openai/openai-python/blob/52357cff50bee57ef442e94d78a0de38b4173fc2/src/openai/resources/beta/chat/completions.py#L100-L104)查询。

对于以下示例，vLLM 使用 `vllm serve meta-llama/Llama-3.1-8B-Instruct` 进行设置。

以下是一个简单的示例，展示了如何使用 Pydantic 模型获取结构化输出：

```plain
from pydantic import BaseModel
from openai import OpenAI




class Info(BaseModel):
    name: str
    age: int




client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="dummy")
completion = client.beta.chat.completions.parse(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Cameron, I'm 28. What's my name and age?"},
    ],
    response_format=Info,
    extra_body=dict(guided_decoding_backend="outlines"),
)


message = completion.choices[0].message
print(message)
assert message.parsed
print("Name:", message.parsed.name)
print("Age:", message.parsed.age)
```

输出：

```go
ParsedChatCompletionMessage[Testing](content='{"name": "Cameron", "age": 28}', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], parsed=Testing(name='Cameron', age=28))
Name: Cameron
Age: 28
```

以下是一个更复杂的示例，使用嵌套的 Pydantic 模型来处理分步数学解决方案：

```plain
from typing import List
from pydantic import BaseModel
from openai import OpenAI




class Step(BaseModel):
    explanation: str
    output: str




class MathResponse(BaseModel):
    steps: List[Step]
    final_answer: str




client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="dummy")
completion = client.beta.chat.completions.parse(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful expert math tutor."},
        {"role": "user", "content": "Solve 8x + 31 = 2."},
    ],
    response_format=MathResponse,
    extra_body=dict(guided_decoding_backend="outlines"),
)


message = completion.choices[0].message
print(message)
assert message.parsed
for i, step in enumerate(message.parsed.steps):
    print(f"Step #{i}:", step)
print("Answer:", message.parsed.final_answer)
```

输出：

```go
ParsedChatCompletionMessage[MathResponse](content='{ "steps": [{ "explanation": "First, let\'s isolate the term with the variable \'x\'. To do this, we\'ll subtract 31 from both sides of the equation.", "output": "8x + 31 - 31 = 2 - 31"}, { "explanation": "By subtracting 31 from both sides, we simplify the equation to 8x = -29.", "output": "8x = -29"}, { "explanation": "Next, let\'s isolate \'x\' by dividing both sides of the equation by 8.", "output": "8x / 8 = -29 / 8"}], "final_answer": "x = -29/8" }', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], parsed=MathResponse(steps=[Step(explanation="First, let's isolate the term with the variable 'x'. To do this, we'll subtract 31 from both sides of the equation.", output='8x + 31 - 31 = 2 - 31'), Step(explanation='By subtracting 31 from both sides, we simplify the equation to 8x = -29.', output='8x = -29'), Step(explanation="Next, let's isolate 'x' by dividing both sides of the equation by 8.", output='8x / 8 = -29 / 8')], final_answer='x = -29/8'))
Step #0: explanation="First, let's isolate the term with the variable 'x'. To do this, we'll subtract 31 from both sides of the equation." output='8x + 31 - 31 = 2 - 31'
Step #1: explanation='By subtracting 31 from both sides, we simplify the equation to 8x = -29.' output='8x = -29'
Step #2: explanation="Next, let's isolate 'x' by dividing both sides of the equation by 8." output='8x / 8 = -29 / 8'
Answer: x = -29/8
```

## 离线推理

离线推理允许使用相同类型的引导解码。要使用它，我们需要使用 `SamplingParams` 中的 `GuidedDecodingParams` 类配置引导解码。`GuidedDecodingParams` 中的主要可用选项包括：

- `json`
- `regex`
- `choice`
- `grammar`
- `backend`
- `whitespace_pattern`

这些参数可以按照与上述在线服务示例相同的方式使用。以下是使用 `choices` 参数的示例：

```plain
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


llm = LLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")


guided_decoding_params = GuidedDecodingParams(choice=["Positive", "Negative"])
sampling_params = SamplingParams(guided_decoding=guided_decoding_params)
outputs = llm.generate(
    prompts="Classify this sentiment: vLLM is wonderful!",
    sampling_params=sampling_params,
)
print(outputs[0].outputs[0].text)
```

完整示例：[examples/offline_inference/structured_outputs.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/structured_outputs.py)
