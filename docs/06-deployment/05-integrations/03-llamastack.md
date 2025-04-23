---
title: Llama Stack
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM 也可通过 [Llama Stack](https://github.com/meta-llama/llama-stack) 获取。

运行下面命令安装 Llama Stack：

```go
pip install llama-stack -q
```

## 使用 OpenAI 兼容 API 进行推理

接下来，使用以下配置启动 Llama Stack 服务器，并将其指向您的 vLLM 服务器：

```plain
inference:
  - provider_id: vllm0
    provider_type: remote::vllm
    config:
      url: http://127.0.0.1:8000
```

请参考[该引导](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/remote-vllm.html)获取更多关于远程 vLLM 提供程序的细节。

## 通过嵌入式 vLLM 进行推理

这里还提供了一个[内联 vLLM 提供程序](https://github.com/meta-llama/llama-stack/tree/main/llama_stack/providers/inline/inference/vllm)。以下是使用该方法的进行配置的示例：

```plain
inference
  - provider_type: vllm
    config:
      model: Llama3.1-8B-Instruct
      tensor_parallel_size: 4
```
