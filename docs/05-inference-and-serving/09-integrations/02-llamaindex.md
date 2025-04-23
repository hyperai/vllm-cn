---

title: LlamaIndex

---

[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM 也可通过 [LlamaIndex](https://github.com/run-llama/llama_index) 获取。


运行下面命令安装 LlamaIndex：

```go
pip install llama-index-llms-vllm -q
```


如需使用单个或多个 GPU 进行推理，请使用 `llamaindex` 中的 `VLLM` 类。

```plain
from llama_index.llms.vllm import Vllm


llm = Vllm(
    model="microsoft/Orca-2-7b",
    tensor_parallel_size=4,
    max_new_tokens=100,
    vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.5},
)
```


请参阅该[教程](https://docs.llamaindex.ai/en/latest/examples/llm/vllm/)获取更多详细信息。

