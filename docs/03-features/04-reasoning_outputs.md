---

title: 推理输出

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


vLLM 支持推理模型，例如 [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)，这些模型旨在生成包含推理步骤和最终结论的输出。


推理模型在其输出中返回一个额外的 `reasoning_content` 字段，该字段包含导致最终结论的推理步骤。其他模型的输出中不存在此字段。


## 支持的模型

vLLM 目前支持以下推理模型：

|型号系列|解析器名称|结构化输出支持|工具调用|
|:----|:----|:----|:----|
|[DeepSeek R1 series  DeepSeek R1 系列](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d)|deepseek_r1|guided_json, guided_regex|guided_json、guided_regex|❌|
|[QwQ-32B](https://huggingface.co/Qwen/QwQ-32B)|deepseek_r1|guided_json, guided_regex|guided_json、guided_regex|✅|


## 快速入门

要使用推理模型，您需要在向聊天补全端点发出请求时指定 `--enable-reasoning` 和 `--reasoning-parser` 标志。`--reasoning-parser` 标志指定用于从模型输出中提取推理内容的推理解析器。

```plain
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --enable-reasoning --reasoning-parser deepseek_r1
```


接下来，向模型发出请求，该请求应在响应中返回推理内容。

```plain
from openai import OpenAI


# Modify OpenAI's API key and API base to use vLLM's API server.
# 修改 OpenAI 的 API 密钥和 API 基础 URL 以使用 vLLM 的 API 服务器。
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


models = client.models.list()
model = models.data[0].id


# Round 1
# 第一轮
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(model=model, messages=messages)


reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content


print("reasoning_content:", reasoning_content)
print("content:", content)
```


`reasoning_content` 字段包含导致最终结论的推理步骤，而 `content` 字段包含最终结论。


## 流式聊天补全

推理模型也支持流式聊天补全。`reasoning_content` 字段在 [聊天补全响应块](https://platform.openai.com/docs/api-reference/chat/streaming) 的 `delta` 字段中可用。

```plain
{
    "id": "chatcmpl-123",
    "object": "chat.completion.chunk",
    "created": 1694268190,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "system_fingerprint": "fp_44709d6fcb",
    "choices": [
        {
            "index": 0,
            "delta": {
                "role": "assistant",
                "reasoning_content": "is",
            },
            "logprobs": null,
            "finish_reason": null
        }
    ]
}
```


OpenAI 的 Python 客户端库官方不支持流式输出中的 `reasoning_content` 属性。但客户端支持在响应中添加额外的属性。你可以使用 `hasattr` 来检查响应中是否存在 `reasoning_content` 属性。例如：

```python
from openai import OpenAI


# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


models = client.models.list()
model = models.data[0].id


messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
stream = client.chat.completions.create(model=model,
                                        messages=messages,
                                        stream=True)


print("client: Start streaming chat completions...")
printed_reasoning_content = False
printed_content = False


for chunk in stream:
    reasoning_content = None
    content = None
    # Check the content is reasoning_content or content
    if hasattr(chunk.choices[0].delta, "reasoning_content"):
        reasoning_content = chunk.choices[0].delta.reasoning_content
    elif hasattr(chunk.choices[0].delta, "content"):
        content = chunk.choices[0].delta.content


    if reasoning_content is not None:
        if not printed_reasoning_content:
            printed_reasoning_content = True
            print("reasoning_content:", end="", flush=True)
        print(reasoning_content, end="", flush=True)
    elif content is not None:
        if not printed_content:
            printed_content = True
            print("\ncontent:", end="", flush=True)
        # Extract and print the content
        print(content, end="", flush=True)


```
请记住在访问响应之前检查响应中是否存在 `reasoning_content`。您可以查看[示例 ](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_with_reasoning_streaming.py)。

## 结构化输出

推理内容也可在结构化输出中找到。像 `xgrammar` 这样的结构化输出引擎将使用推理内容来生成结构化输出。

```python
from openai import OpenAI
from pydantic import BaseModel


# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


models = client.models.list()
model = models.data[0].id




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
```


## 工具调用

当工具调用和推理解析器都处于启用状态时，推理内容也可用。此外，工具调用仅分析 `content` 字段中的函数，而不分析 `reasoning_content` 中的函数。

```python
from openai import OpenAI


client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")


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


print(response)
tool_call = response.choices[0].message.tool_calls[0].function


print(f"reasoning_content: {response.choices[0].message.reasoning_content}")
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
```

## 

## 如何支持新的推理模型

您可以添加一个新的 `ReasoningParser`，类似于 `vllm/entrypoints/openai/reasoning_parsers/deepseek_r1_reasoning_parser.py`。

```plain
# import the required packages
# 导入所需的包


from vllm.entrypoints.openai.reasoning_parsers.abs_reasoning_parsers import (
    ReasoningParser, ReasoningParserManager)
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)


# define a reasoning parser and register it to vllm
# the name list in register_module can be used
# in --reasoning-parser.
# 定义一个推理解析器并将其注册到 vLLM
# register_module 中的名称列表可以在
# --reasoning-parser 中使用。
@ReasoningParserManager.register_module(["example"])
class ExampleParser(ReasoningParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)


    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """
        Instance method that should be implemented for extracting reasoning
        from an incomplete response; for use when handling reasoning calls and
        streaming. Has to be an instance method because  it requires state -
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        实例方法，用于从未完成的响应中提取推理内容；
        适用于处理推理调用和流式传输时。
        必须是一个实例方法，因为它需要状态 -
        当前的 token/差异，以及之前解析和提取的信息（参见构造函数）。
        """


    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from a complete model-generated string.
        从完整的模型生成字符串中提取推理内容。


        Used for non-streaming responses where we have the entire model response
        available before sending to the client.
        用于非流式响应，其中我们在发送给客户端之前拥有完整的模型响应。


        Parameters:
        参数：
        model_output: str
            The model-generated string to extract reasoning content from.
        model_output: str
            要从中提取推理内容的模型生成字符串。


        request: ChatCompletionRequest
            The request object that was used to generate the model_output.
        request: ChatCompletionRequest
            用于生成 model_output 的请求对象。


        Returns:
        Tuple[Optional[str], Optional[str]]
            A tuple containing the reasoning content and the content.
        返回：
        Tuple[Optional[str], Optional[str]]
            包含推理内容和内容的元组。
        """
```


此外，要启用结构化输出，您需要创建一个类似于 中的新 `Reasoner` `vllm/model_executor/guided_decoding/reasoner/deepseek_reasoner.py` 。

```python
@dataclass
class DeepSeekReasoner(Reasoner):
    """
    Reasoner for DeepSeek R series models.
    """
    start_token_id: int
    end_token_id: int


    start_token: str = "<think>"
    end_token: str = "</think>"


    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizer) -> Reasoner:
        return cls(start_token_id=tokenizer.encode(
            "<think>", add_special_tokens=False)[0],
                   end_token_id=tokenizer.encode("</think>",
                                                 add_special_tokens=False)[0])


    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self.end_token_id in input_ids
    ...
```
像 `xgrammar` 这样的结构化输出引擎将使用 `end_token_id` 来检查模型输出中是否存在推理内容，如果是，则跳过结构化输出。

最后，您可以使用 `--enable-reasoning` 和 `--reasoning-parser` 标志为模型启用推理。


```plain
vllm serve <model_tag> \
    --enable-reasoning --reasoning-parser example
```


## 局限性

* 推理内容仅适用于在线服务的聊天补全端点（`/v1/chat/completions`）。



