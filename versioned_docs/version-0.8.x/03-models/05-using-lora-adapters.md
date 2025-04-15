---
title: 使用 LoRA 适配器
---


本文档向您展示如何在基本模型之上将 [LoRA 适配器](https://arxiv.org/abs/2106.09685) 与 vLLM 结合使用。


LoRA 适配器可与任何实现了 `SupportsLoRA` 的 vLLM 模型一起使用。


适配器能以最小的开销根据每个请求有效地服务。首先，我们下载适配器并将其保存在本地：

```python
from huggingface_hub import snapshot_download


sql_lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
```


然后我们实例化基础模型并传入 `enable_lora=True` 标志: 

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_lora=True)
```


我们现在可以提交提示并使用 `lora_request` 参数调用 `llm.generate` 。`LoRARequest` 的第一个参数是人为可识别的名称，第二个参数是适配器的全局唯一 ID，第三个参数是 LoRA 适配器的路径。

```python
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
    stop=["[/assistant]"]
)


prompts = [
     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
     "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
]


outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("sql_adapter", 1, sql_lora_path)
)
```
查看 [examples/multilora_inference.py](https://github.com/vllm-project/vllm/blob/main/examples/multilora_inference.py)，了解如何将 LoRA 适配器与异步引擎结合使用以及如何使用更高级的配置选项。

## LoRA 适配器服务

LoRA 适配模型也可以通过 Open-AI 兼容的 vLLM 服务器提供服务。为此，我们在启动服务器时使用 `--lora-modules {name}={path} {name}={path}` 来指定每个 LoRA 模块：

```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --enable-lora \
    --lora-modules sql-lora=$HOME/.cache/huggingface/hub/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c/
```
**注意：**
提交 ID *0dfa347e8877a4d4ed19ee56c140fa518470028c* 可能会随着时间的推移而改变。请检查您环境中的最新提交 ID，以确保您使用的是正确的提交 ID。


服务器入口点接受所有其他 LoRA 配置参数（`max_loras` 、 `max_lora_rank` 、 `max_cpu_loras` 等），这些参数将应用于所有即将到来的请求。查询 `/models` 端点后，我们应该看到 LoRA 及其基本模型：

```bash
curl localhost:8000/v1/models | jq .
{
    "object": "list",
    "data": [
        {
            "id": "meta-llama/Llama-2-7b-hf",
            "object": "model",
            ...
        },
        {
            "id": "sql-lora",
            "object": "model",
            ...
        }
    ]
}
```


请求可以通过 `model` 请求参数指定 LoRA 适配器，就像指定任何其他模型一样。这些请求将根据服务器端的 LoRA 配置进行处理（即与基础模型请求并行处理，如果提供了其他 LoRA 适配器请求且 `max_loras` 设置得足够高，也可能与其他 LoRA 适配器请求并行处理）。


以下是请求示例：

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "sql-lora",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }' | jq
```

## 

## 动态提供 LoRA 适配器

除了在服务器启动时提供 LoRA 适配器，vLLM 服务器现在还支持通过专门的 API 端点在运行时动态加载和卸载 LoRA 适配器。当需要灵活地在运行中更改模型时，这个功能特别有用。


请注意：在生产环境中启用这一特性可能带来风险，因为用户可能会参与到模型适配器的管理工作中。


如需启用 LoRA 适配器的动态加载和卸载，请确保环境变量 *VLLM_ALLOW_RUNTIME_LORA_UPDATING* 已被设置为 True。启用此选项后，API 服务器会在日志中记录一条警告，以示动态加载功能已经开启。


```python
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```


加载 LoRA 适配器：


如需动态加载一个 LoRA 适配器，需要向 */v1/load_lora_adapter* 端点发送一个 POST 请求，并提供要加载的适配器的详细信息。请求的数据负载应包括 LoRA 适配器的名称和路径。


加载 LoRA 适配器的请求示例：


```python
curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "sql_adapter",
    "lora_path": "/path/to/sql-lora-adapter"
}'
```
如果请求成功，API 将返回一个 200 OK 的状态码。如果发生错误，比如适配器无法找到或加载，将返回一个适当的错误信息。

卸载 LoRA 适配器：


如需卸载之前已加载的 LoRA 适配器，需要向 */v1/unload_lora_adapter* 端点发送一个 POST 请求，并提供要卸载的适配器的名称或 ID。


卸载 LoRA 适配器的请求示例：

```plain
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "sql_adapter"
}'
```


## –lora-modules 的新格式

在以前的版本中，用户会通过以下格式提供 LoRA 模块，要么以键值对的形式，要么以 JSON 格式。例如：

```plain
--lora-modules sql-lora=$HOME/.cache/huggingface/hub/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c/
```
此格式只包含每个 LoRA 模块的名称和路径，但未提供指定 *base_model_name* 的方法。现在，你可以通过 JSON 格式同时指定 base_model_name 以及名称和路径。例如：
```plain
--lora-modules '{"name": "sql-lora", "path": "/path/to/lora", "base_model_name": "meta-llama/Llama-2-7b"}'
```
为了确保向后兼容性，你仍然可以使用旧的键值对格式 (name=path)，但在这种情况下 *base_model_name* 将保持未指定状态。

## 模型卡中的 LoRA 模型谱系

新的 –lora-modules 格式主要用于支持在模型卡中显示父模型信息。以下是你当前响应如何支持此功能的说明：


* LoRA 模型 sql-lora 的父字段现在链接到其基础模型 *meta-llama/Llama-2-7b-hf*。这正确地反映了基础模型与 LoRA 适配器之间的层次关系。

* *root* 字段指向 lora 适配器的工件位置。

```plain
curl http://localhost:8000/v1/models


{
    "object": "list",
    "data": [
        {
        "id": "meta-llama/Llama-2-7b-hf",
        "object": "model",
        "created": 1715644056,
        "owned_by": "vllm",
        "root": "~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/",
        "parent": null,
        "permission": [
            {
            .....
            }
        ]
        },
        {
        "id": "sql-lora",
        "object": "model",
        "created": 1715644056,
        "owned_by": "vllm",
        "root": "~/.cache/huggingface/hub/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c/",
        "parent": meta-llama/Llama-2-7b-hf,
        "permission": [
            {
            ....
            }
        ]
        }
    ]
}
```


