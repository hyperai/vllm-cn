---
title: 使用 OpenAI 批处理文件格式进行离线推理
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

源码 [examples/offline_inference/openai](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/openai)

> **重要信息**
>
> 本指南介绍如何使用 OpenAI 批处理文件格式执行批量推理，**而非**完整的 Batch (REST) API。  

## 文件格式

OpenAI 批处理文件格式由多行 JSON 对象组成。

[点击此处查看示例文件。](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/openai/openai_example_batch.jsonl)

每行代表一个独立请求。详情请参阅 [OpenAI 包参考文档](https://platform.openai.com/docs/api-reference/batch/requestInput)。

```plain
We currently support `/v1/chat/completions`, `/v1/embeddings`, and `/v1/score` endpoints (completions coming soon).
```

## 准备工作

- 本文示例使用 `meta-llama/Meta-Llama-3-8B-Instruct` 模型。

  - 创建 [用户访问令牌](https://huggingface.co/docs/hub/en/security-tokens)

  - 在本地安装令牌（运行 `huggingface-cli login`）。

  - 访问 [模型卡片](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 并同意条款以获取受限模型访问权限。

## 示例 1：使用本地文件运行

### 步骤 1：创建批处理文件

您可以下载示例批处理文件，或在工作目录中创建自己的批处理文件。

```plain
wget https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference/openai/openai_example_batch.jsonl
```

创建完成后，文件内容应如下所示：

```plain
$ cat offline_inference/openai/openai_example_batch.jsonl
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
```

### 步骤 2：运行批处理

批处理工具设计为通过命令行使用。

运行以下命令执行批处理，结果将写入 `results.jsonl` 文件：

```plain
python -m vllm.entrypoints.openai.run_batch -i offline_inference/openai/openai_example_batch.jsonl -o results.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct
```

### 步骤 3：查看结果

结果现已保存在 `results.jsonl` 中。可通过 `cat results.jsonl` 查看：

```plain
$ cat results.jsonl
{"id":"vllm-383d1c59835645aeb2e07d004d62a826","custom_id":"request-1","response":{"id":"cmpl-61c020e54b964d5a98fa7527bfcdd378","object":"chat.completion","created":1715633336,"model":"meta-llama/Meta-Llama-3-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"Hello! It's great to meet you! I'm here to help with any questions or tasks you may have. What's on your mind today?"},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":25,"total_tokens":56,"completion_tokens":31}},"error":null}
{"id":"vllm-42e3d09b14b04568afa3f1797751a267","custom_id":"request-2","response":{"id":"cmpl-f44d049f6b3a42d4b2d7850bb1e31bcc","object":"chat.completion","created":1715633336,"model":"meta-llama/Meta-Llama-3-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"*silence*"},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":27,"total_tokens":32,"completion_tokens":5}},"error":null}
```

## 示例 2：使用远程文件

批处理运行器支持通过 http/https 访问的远程输入输出 URL。

例如，要运行位于 `https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference/openai/openai_example_batch.jsonl` 的示例输入文件，可执行：

```plain
python -m vllm.entrypoints.openai.run_batch -i https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference/openai/openai_example_batch.jsonl -o results.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct
```

## 示例 3：与 AWS S3 集成

要与云存储集成，我们推荐使用预签名 URL。

[了解更多关于 S3 预签名 URL 的信息]

### 额外准备工作

- [创建 S3 存储桶](https://docs.aws.amazon.com/AmazonS3/latest/userguide/creating-bucket.html)。
- 安装 `awscli` 包（运行 `pip install awscli`）以配置凭证并交互式使用 S3。

  - [配置凭证](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html)。

- 安装 `boto3` Python 包（运行 `pip install boto3`）以生成预签名 URL。

### 步骤 1：上传输入脚本

您可以下载示例批处理文件，或在工作目录中创建自己的批处理文件。

```plain
wget https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference/openai/openai_example_batch.jsonl
```

创建完成后，文件内容应如下所示：

```plain
$ cat offline_inference/openai/openai_example_batch.jsonl
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
```

现在将批处理文件上传至 S3 存储桶：

```plain
aws s3 cp offline_inference/openai/openai_example_batch.jsonl s3://MY_BUCKET/MY_INPUT_FILE.jsonl
```

### 步骤 2：生成预签名 URL

预签名 URL 只能通过 SDK 生成。运行以下 Python 脚本生成预签名 URL。请将 `MY_BUCKET`、`MY_INPUT_FILE.jsonl` 和 `MY_OUTPUT_FILE.jsonl` 替换为您的存储桶和文件名。

（脚本改编自 [https://github.com/awsdocs/aws-doc-sdk-examples/blob/main/python/example_code/s3/s3_basics/presigned_url.py](https://github.com/awsdocs/aws-doc-sdk-examples/blob/main/python/example_code/s3/s3_basics/presigned_url.py)）

```python
import boto3
from botocore.exceptions import ClientError


def generate_presigned_url(s3_client, client_method, method_parameters, expires_in):

    """
    生成可用于执行操作的预签名 Amazon S3 URL。


    :param s3_client: Boto3 Amazon S3 客户端。
    :param client_method: URL 执行的客户端方法名称。
    :param method_parameters: 指定客户端方法的参数。
    :param expires_in: 预签名 URL 的有效秒数。
    :return: 预签名 URL。
    """


    try:
        url = s3_client.generate_presigned_url(
            ClientMethod=client_method, Params=method_parameters, ExpiresIn=expires_in
        )
    except ClientError:
        raise
    return url




s3_client = boto3.client("s3")
input_url = generate_presigned_url(
    s3_client, "get_object", {"Bucket": "MY_BUCKET", "Key": "MY_INPUT_FILE.jsonl"}, 3600
)
output_url = generate_presigned_url(
    s3_client, "put_object", {"Bucket": "MY_BUCKET", "Key": "MY_OUTPUT_FILE.jsonl"}, 3600
)
print(f"{input_url=}")
print(f"{output_url=}")
```

脚本输出应类似：

```plain
input_url='https://s3.us-west-2.amazonaws.com/MY_BUCKET/MY_INPUT_FILE.jsonl?AWSAccessKeyId=ABCDEFGHIJKLMNOPQRST&Signature=abcdefghijklmnopqrstuvwxyz12345&Expires=1715800091'
output_url='https://s3.us-west-2.amazonaws.com/MY_BUCKET/MY_OUTPUT_FILE.jsonl?AWSAccessKeyId=ABCDEFGHIJKLMNOPQRST&Signature=abcdefghijklmnopqrstuvwxyz12345&Expires=1715800091'
```

### 步骤 3：使用预签名 URL 运行批处理

现在可以使用上一步生成的 URL 运行批处理：

```plain
python -m vllm.entrypoints.openai.run_batch \
    -i "https://s3.us-west-2.amazonaws.com/MY_BUCKET/MY_INPUT_FILE.jsonl?AWSAccessKeyId=ABCDEFGHIJKLMNOPQRST&Signature=abcdefghijklmnopqrstuvwxyz12345&Expires=1715800091" \
    -o "https://s3.us-west-2.amazonaws.com/MY_BUCKET/MY_OUTPUT_FILE.jsonl?AWSAccessKeyId=ABCDEFGHIJKLMNOPQRST&Signature=abcdefghijklmnopqrstuvwxyz12345&Expires=1715800091" \
    --model --model meta-llama/Meta-Llama-3-8B-Instruct
```

### 步骤 4：查看结果

结果现已保存在 S3 上。可通过以下命令在终端查看：

```plain
aws s3 cp s3://MY_BUCKET/MY_OUTPUT_FILE.jsonl -
```

## 示例 4：使用 embeddings 端点

### 额外准备工作

- 确保使用 `vllm >= 0.5.5` 版本。

### 步骤 1：创建批处理文件

在批处理文件中添加 embedding 请求。示例如下：

```plain
{"custom_id": "request-1", "method": "POST", "url": "/v1/embeddings", "body": {"model": "intfloat/e5-mistral-7b-instruct", "input": "You are a helpful assistant."}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/embeddings", "body": {"model": "intfloat/e5-mistral-7b-instruct", "input": "You are an unhelpful assistant."}}
```

您甚至可以在批处理文件中混合 chat completion 和 embedding 请求，只要您使用的模型同时支持这两种功能（注意所有请求必须使用同一模型）。

### 步骤 2：运行批处理

使用与之前示例相同的命令运行批处理。

### 步骤 3：查看结果

通过 `cat results.jsonl` 查看结果：

```plain
$ cat results.jsonl
{"id":"vllm-db0f71f7dec244e6bce530e0b4ef908b","custom_id":"request-1","response":{"status_code":200,"request_id":"vllm-batch-3580bf4d4ae54d52b67eee266a6eab20","body":{"id":"embd-33ac2efa7996430184461f2e38529746","object":"list","created":444647,"model":"intfloat/e5-mistral-7b-instruct","data":[{"index":0,"object":"embedding","embedding":[0.016204833984375,0.0092010498046875,0.0018358230590820312,-0.0028228759765625,0.001422882080078125,-0.0031147003173828125,...]}],"usage":{"prompt_tokens":8,"total_tokens":8,"completion_tokens":0}}},"error":null}
...
```

## 示例 5：使用 score 端点

### 额外准备工作

- 确保使用 `vllm >= 0.7.0` 版本。

### 步骤 1：创建批处理文件

在批处理文件中添加 score 请求。示例如下：

```plain
{"custom_id": "request-1", "method": "POST", "url": "/v1/score", "body": {"model": "BAAI/bge-reranker-v2-m3", "text_1": "What is the capital of France?", "text_2": ["The capital of Brazil is Brasilia.", "The capital of France is Paris."]}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/score", "body": {"model": "BAAI/bge-reranker-v2-m3", "text_1": "What is the capital of France?", "text_2": ["The capital of Brazil is Brasilia.", "The capital of France is Paris."]}}
```

您可以在批处理文件中混合 chat completion、embedding 和 score 请求，只要您使用的模型支持所有这些功能（注意所有请求必须使用同一模型）。

### 步骤 2：运行批处理

使用与之前示例相同的命令运行批处理。

### 步骤 3：查看结果

通过 `cat results.jsonl` 查看结果：

```plain
$ cat results.jsonl
{"id":"vllm-f87c5c4539184f618e555744a2965987","custom_id":"request-1","response":{"status_code":200,"request_id":"vllm-batch-806ab64512e44071b37d3f7ccd291413","body":{"id":"score-4ee45236897b4d29907d49b01298cdb1","object":"list","created":1737847944,"model":"BAAI/bge-reranker-v2-m3","data":[{"index":0,"object":"score","score":0.0010900497436523438},{"index":1,"object":"score","score":1.0}],"usage":{"prompt_tokens":37,"total_tokens":37,"completion_tokens":0,"prompt_tokens_details":null}}},"error":null}
{"id":"vllm-41990c51a26d4fac8419077f12871099","custom_id":"request-2","response":{"status_code":200,"request_id":"vllm-batch-73ce66379026482699f81974e14e1e99","body":{"id":"score-13f2ffe6ba40460fbf9f7f00ad667d75","object":"list","created":1737847944,"model":"BAAI/bge-reranker-v2-m3","data":[{"index":0,"object":"score","score":0.001094818115234375},{"index":1,"object":"score","score":1.0}],"usage":{"prompt_tokens":37,"total_tokens":37,"completion_tokens":0,"prompt_tokens_details":null}}},"error":null}
```

# 示例材料

## openai_example_batch.jsonl

```json
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3
```
