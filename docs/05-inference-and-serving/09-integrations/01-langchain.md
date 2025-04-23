---
title: LangChain
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM 也可通过 [LangChain](https://github.com/langchain-ai/langchain) 获取。

运行下面命令安装 LangChain：

```go
pip install langchain langchain_community -q
```

从 `langchain` 使用 `VLLM` 类，在一个或多个 GPU 上运行推理

```plain
from langchain_community.llms import VLLM


llm = VLLM(model="mosaicml/mpt-7b",
           trust_remote_code=True,  # mandatory for hf models
           max_new_tokens=128,
           top_k=10,
           top_p=0.95,
           temperature=0.8,
           # tensor_parallel_size=... # for distributed inference
)


print(llm("What is the capital of France ?"))
```

请参阅该[教程](https://python.langchain.com/docs/integrations/llms/vllm)获得更多细节。
