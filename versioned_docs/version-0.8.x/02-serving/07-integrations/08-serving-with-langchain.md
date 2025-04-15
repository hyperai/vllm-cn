---
title: 使用 Langchain 提供服务
---


vLLM 也可通过 [Langchain](https://github.com/langchain-ai/langchain) 获得。


要安装 langchain，请运行：

```plain
pip install langchain langchain_community -q
```


要在单个或多个 GPU 上运行推理，请使用 `langchain` 中的 `VLLM` 类。

```python
from langchain_community.llms import VLLM


llm = VLLM(model="mosaicml/mpt-7b",
           trust_remote_code=True,  # mandatory for hf models
           trust_remote_code=True, # 对于 hf 型号是强制的


           max_new_tokens=128,
           top_k=10,
           top_p=0.95,
           temperature=0.8,
           # tensor_parallel_size=... # for distributed inference
           # tensor_parallel_size=... # 用于分布式推理


)


print(llm("What is the capital of France ?"))
```


请参阅此 [教程](https://python.langchain.com/docs/integrations/llms/vllm) 了解更多详细信息。

