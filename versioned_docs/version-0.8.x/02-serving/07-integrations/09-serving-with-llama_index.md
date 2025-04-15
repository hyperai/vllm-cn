---
title: 使用 llama_index 服务
---


vLLM 也可通过 [llama_index](https://github.com/run-llama/llama_index) 获得。


要安装 llamaindex，请运行

```plain
pip install llama-index-llms-vllm -q
```


如需在单个或多个 GPU 上运行推理，请使用 `llamaindex` 中的 `Vllm` 类：

```python
from llama_index.llms.vllm import Vllm


llm = Vllm(
    model="microsoft/Orca-2-7b",
    tensor_parallel_size=4,
    max_new_tokens=100,
    vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.5},
)
```


请参阅此[教程](https://docs.llamaindex.ai/en/latest/examples/llm/vllm/)了解更多详细信息。

