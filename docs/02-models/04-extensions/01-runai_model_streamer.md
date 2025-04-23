---

title: 使用 Run:ai Model Streamer 加载模型

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


Run:ai Model Streamer 是一个用于并发读取张量并将其流式传输到 GPU 内存的库。更多信息可以在 [Run:ai Model Streamer 文档](https://github.com/run-ai/runai-model-streamer/blob/master/docs/README.md)中找到。


vLLM 支持使用 Run:ai Model Streamer 加载 Safetensors 格式的权重。首先，您需要安装 vLLM 的 RunAI 可选依赖项：

```plain
pip3 install vllm[runai]
```


如需将其作为 OpenAI 兼容服务器运行，请添加 --load-format runai_streamer 标志：

```plain
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct --load-format runai_streamer
```


如需从 AWS S3 对象存储运行模型，请运行：

```plain
vllm serve s3://core-llm/Llama-3-8b --load-format runai_streamer
```


如需从 S3 兼容的对象存储运行模型，请运行：

```plain
RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 AWS_EC2_METADATA_DISABLED=true AWS_ENDPOINT_URL=https://storage.googleapis.com vllm serve s3://core-llm/Llama-3-8b --load-format runai_streamer
```

## 

## 可调参数


您可以使用 --model-loader-extra-config 调整参数：


您可以调节 concurrency，这个设置决定了同时进行的并发任务数量，以及从文件中读取数据到CPU缓冲区的操作系统线程数。如果是从S3服务器读取数据，该设置将影响主机同时打开的客户端实例数量。

```plain
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct --load-format runai_streamer --model-loader-extra-config '{"concurrency":16}'
```


您可以控制从文件读取张量的 CPU 内存缓冲区的大小，并限制此大小。如果您想了解更多关于如何限制CPU缓冲内存的信息，可以在[此处](https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/env-vars.md#runai_streamer_memory_limit)找到更多资料。。

```plain
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct --load-format runai_streamer --model-loader-extra-config '{"memory_limit":5368709120}'
```
**注意：**
若想了解如何通过环境变量调整可配置的参数和额外参数，请参阅[环境变量文档](https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/env-vars.md)。

