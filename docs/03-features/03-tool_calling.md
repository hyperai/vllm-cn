---
title: 工具调用
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM 目前支持命名函数调用，并在聊天补全 API 的 `tool_choice` 字段中支持 `auto` 和 `none` 选项。`required` 选项**尚未支持**，但已在[开发计划](https://github.com/vllm-project/vllm/issues/13002#)中。

## 快速开始

启动服务器时启用工具调用功能。此示例使用 Meta 的 Llama 3.1 8B 模型，因此我们需要使用 vLLM 示例目录中的 llama3 工具调用聊天模板：

```plain
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template examples/tool_chat_template_llama3.1_json.jinja
```

接下来，向模型发起一个请求，该请求应导致模型使用可用的工具：

```plain
from openai import OpenAI
import json


client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")


def get_weather(location: str, unit: str):
    return f"Getting the weather for {location} in {unit}..."
tool_functions = {"get_weather": get_weather}


tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location", "unit"]
        }
    }
}]


response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
    tools=tools,
    tool_choice="auto"
)


tool_call = response.choices[0].message.tool_calls[0].function
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
print(f"Result: {get_weather(**json.loads(tool_call.arguments))}")
```

示例输出:

```plain
Function called: get_weather
Arguments: {"location": "San Francisco, CA", "unit": "fahrenheit"}
Result: Getting the weather for San Francisco, CA in fahrenheit...
```

此示例演示了以下内容：

- 启用工具调用功能并设置服务器
- 定义一个实际函数来处理工具调用
- 使用 `tool_choice="auto"` 发出请求
- 处理结构化响应并执行相应的函数

您还可以通过设置 `tool_choice={"type": "function", "function": {"name": "get_weather"}}` 来使用命名函数调用指定特定函数。请注意，这将使用引导式解码后端——因此第一次使用时会有几秒钟（或更长时间）的延迟，因为 FSM 首次编译后会缓存以供后续请求使用。

请记住，调用方的责任包括：

1. 在请求中定义适当的工具

2. 在聊天消息中包含相关上下文

3. 在应用程序逻辑中处理工具调用

有关更高级的用法，包括并行工具调用和不同模型特定的解析器，请参阅以下部分。

## 命名函数调用

vLLM 默认支持聊天补全 API 中的命名函数调用。它通过 Outlines 使用引导式解码来实现此功能，因此默认启用，并且适用于所有支持的模型。您可以确保获得一个可解析的函数调用——但不一定是高质量的。

vLLM 将使用引导式解码来确保响应与 `tools` 参数中定义的 JSON 模式工具参数对象匹配。为了获得最佳结果，我们建议在提示中指定预期的输出格式/模式，以确保模型的生成意图与引导式解码后端强制生成的模式一致。

要使用命名函数，您需要在聊天补全请求的 `tools` 参数中定义函数，并在聊天补全请求的 `tool_choice` 参数中指定其中一个工具的名称。

## 自动化函数调用

要启用此功能，您应设置以下标志：

- `--enable-auto-tool-choice` – **必填**。自动工具选择。告诉 vLLM 您希望启用模型在认为合适时生成自己的工具调用。
- `--tool-call-parser` – 选择要使用的工具解析器（如下所列）。未来将继续添加更多工具解析器，并且您还可以在 `--tool-parser-plugin` 中注册自己的工具解析器。
- `--tool-parser-plugin` – **可选**。工具解析器插件，用于将用户定义的工具解析器注册到 vLLM 中，注册的工具解析器名称可以在 `--tool-call-parser` 中指定。
- `--chat-template` – 自动工具选择的**可选项**。处理 `tool`-角色消息和包含先前生成工具调用的 `assistant`-角色消息的聊天模板路径。Hermes、Mistral 和 Llama 模型在其 `tokenizer_config.json` 文件中具有工具兼容的聊天模板，但您可以指定自定义模板。如果您的模型在 `tokenizer_config.json` 中配置了特定于工具使用的聊天模板，则可以将此参数设置为 `tool_use`。在这种情况下，将根据 transformers 规范使用它。更多信息请参阅 HuggingFace 的[说明](https://huggingface.co/docs/transformers/en/chat_templating#why-do-some-models-have-multiple-templates)；您可以在此处的 `tokenizer_config.json` 中找到[示例](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/blob/main/tokenizer_config.json)。

如果您喜欢的工具调用模型没有支持，可以随时贡献解析器和工具使用聊天模板！

### Hermes 模型(`hermes`)

所有 Nous Research Hermes 系列的模型（比 Hermes 2 Pro 更新的版本）都应支持。

- `NousResearch/Hermes-2-Pro-*`
- `NousResearch/Hermes-2-Theta-*`
- `NousResearch/Hermes-3-*`

注意：Hermes 2 **Theta**模型由于创建过程中的合并步骤，已知工具调用质量和能力有所下降。

标志：`--tool-call-parser hermes`

### Mistral 模型 (`mistral`)

支持的模型：

- `mistralai/Mistral-7B-Instruct-v0.3`（已确认）
- 其他 Mistral 函数调用模型也兼容。

已知问题：

1. Mistral 7B 难以正确生成并行工具调用。

2. Mistral 的 `tokenizer_config.json` 聊天模板要求工具调用 ID 必须为 9 位数字，而 vLLM 生成的 ID 更长。当不满足此条件时，会抛出异常。因此，提供了以下额外的聊天模板：

   - `examples/tool_chat_template_mistral.jinja`：这是“官方”Mistral 聊天模板，但经过调整以支持 vLLM 的工具调用 ID（工具调用 ID 字段被截断为最后 9 位）。

   - `examples/tool_chat_template_mistral_parallel.jinja`：这是一个“更好”的版本，当提供工具时添加了工具使用系统提示，从而在并行工具调用时提供更高的可靠性。

 推荐标志:`--tool-call-parser mistral --chat-template examples/tool_chat_template_mistral_parallel.jinja`

### Llama 模型 (`llama3_json`)

支持的模型：

- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `meta-llama/Meta-Llama-3.1-70B-Instruct`
- `meta-llama/Meta-Llama-3.1-405B-Instruct`
- `meta-llama/Meta-Llama-3.1-405B-Instruct-FP8`

支持的工具调用是[基于 JSON 的工具调用](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling)。对于 Llama-3.2 模型中的 [Python 风格工具调用](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md#zero-shot-function-calling)，请参阅下面的 Python 风格工具解析器。不支持内置的 Python 工具调用或自定义工具调用格式。

已知问题：

1. 不支持并行工具调用。

2. 模型可能生成格式错误的参数，例如将数组序列化为字符串而不是数组。

`tool_chat_template_llama3_json.jinja` 文件包含「官方」Llama 聊天模板，但经过调整以更好地与 vLLM 配合使用。

推荐标志: `--tool-call-parser llama3_json --chat-template examples/tool_chat_template_llama3_json.jinja`

#### IBM Granite

支持的模型：

- `ibm-granite/granite-3.0-8b-instruct`

推荐标志:  `--tool-call-parser granite --chat-template examples/tool_chat_template_granite.jinja`

`examples/tool_chat_template_granite.jinja`：这是从 Huggingface 原始模板修改而来的聊天模板。支持并行函数调用。

- `ibm-granite/granite-3.1-8b-instruct`

推荐标志: `--tool-call-parser granite`

可以直接使用 Huggingface 的聊天模板。支持并行函数调用。

- `ibm-granite/granite-20b-functioncalling`

Recommended flags: 推荐标志: `--tool-call-parser granite-20b-fc --chat-template examples/tool_chat_template_granite_20b_fc.jinja`

`examples/tool_chat_template_granite_20b_fc.jinja`：这是从 Huggingface 原始模板修改而来的聊天模板，原始模板与 vLLM 不兼容。它融合了 Hermes 模板中的函数描述元素，并遵循[论文](https://arxiv.org/abs/2407.00121)中的“响应生成”模式的系统提示。支持并行函数调用。

### InternLM 模型(`internlm`)

支持的模型：

- `internlm/internlm2_5-7b-chat`（已确认）
- 其他 `internlm2.5` 函数调用模型也兼容。

已知问题：

- 尽管此实现也支持 InternLM2，但在使用 `internlm/internlm2-chat-7b` 模型测试时，工具调用结果不稳定。

推荐标志：`--tool-call-parser internlm --chat-template examples/tool_chat_template_internlm2_tool.jinja`

### Jamba 模型(`jamba`)

支持 AI21 的 Jamba-1.5 模型。

- `ai21labs/AI21-Jamba-1.5-Mini`
- `ai21labs/AI21-Jamba-1.5-Large`

标志：`--tool-call-parser jamba`

### 支持 Python 风格工具调用的模型 (`pythonic`)

越来越多的模型使用 Python 列表来表示工具调用，而不是 JSON。这样做的好处是天生支持并行工具调用，并消除了工具调用所需的 JSON 模式的歧义。`Python 风格`工具解析器可以支持此类模型。

举一个具体示例，这些模型可以通过生成以下内容来查询旧金山和西雅图的天气：

```plain
[get_weather(city='San Francisco', metric='celsius'), get_weather(city='Seattle', metric='celsius')]
```

局限性：

- 模型不得在同一生成中同时生成文本和工具调用。对于特定模型来说，这可能不难更改，但社区目前对生成工具调用时发出的 token 缺乏共识。（特别是 Llama 3.2 模型不会发出此类 token。）
- Llama 的小型模型难以有效使用工具。

支持的示例模型:

- `meta-llama/Llama-3.2-1B-Instruct*`（与 `examples/tool_chat_template_llama3.2_pythonic.jinja` 一起使用）
- `meta-llama/Llama-3.2-3B-Instruct*`（与 `examples/tool_chat_template_llama3.2_pythonic.jinja` 一起使用）
- `Team-ACE/ToolACE-8B`（与 `examples/tool_chat_template_toolace.jinja` 一起使用）
- `fixie-ai/ultravox-v0_4-ToolACE-8B`（与 `examples/tool_chat_template_toolace.jinja` 一起使用）

标志: `--tool-call-parser pythonic --chat-template {see_above}`

---

**警告**：Llama 的小型模型经常无法以正确格式生成工具调用。效果可能因情况而异。

---

## 如何编写工具解析器插件

工具解析器插件是一个包含一个或多个 ToolParser 实现的 Python 文件。您可以编写类似于 vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py 中的 `Hermes2ProToolParser` 的 ToolParser。

以下是插件文件的摘要：

```plain


# 导入所需的包


# 定义一个工具解析器并将其注册到 vLLM
# register_module 中的名称列表
# 可以在 --tool-call-parser 中使用。
# 您可以在此处定义任意数量的工具解析器。


@ToolParserManager.register_module(["example"])
class ExampleToolParser(ToolParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)


    # 调整请求。例如：将 skip_special_tokens 设置为 False 以支持工具调用输出。
    def adjust_request(
            self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        return request


    # 实现流式调用的工具解析
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        return delta


    # 实现非流式调用的工具解析
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        return ExtractedToolCallInformation(tools_called=False,
                                            tool_calls=[],
                                            content=text)




```

接下来你可以再命令行中按下面的方式使用该补丁，如下所示：

```go
    --enable-auto-tool-choice \
    --tool-parser-plugin <absolute path of the plugin file>
    --tool-call-parser example \
    --chat-template <your chat template> \
```
