---
title: BitsAndBytes
---


vLLM 现在支持 [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) 以实现更高效的模型推理。 BitsAndBytes 量化模型可以减少内存使用并增强性能，且不会明显降低准确性。与其他量化方法相比，BitsAndBytes 无需使用输入数据来校准量化模型。


以下是在 vLLM 中 使用 BitsAndBytes 的步骤。

```plain
pip install bitsandbytes>=0.42.0
```
vLLM 会读取模型的配置文件，并支持动态量化和预量化的 checkpoint。

您可以在 [https://huggingface.co/models?other=bitsandbytes](https://huggingface.co/models?other=bitsandbytes) 上找到经过 bitsandbytes 量化模型。通常，这些存储库都有一个包含  quantization_config 部分的 config.json 文件。


## 读取量化 checkpoint

```python
from vllm import LLM
import torch
# unsloth/tinyllama-bnb-4bit is a pre-quantized checkpoint.
# unsloth/tinyllama-bnb-4bit 是一个预量化的 checkpoint。


model_id = "unsloth/tinyllama-bnb-4bit"
llm = LLM(model=model_id, dtype=torch.bfloat16, trust_remote_code=True, \
quantization="bitsandbytes", load_format="bitsandbytes")
```


## 过程中量化：加载为 4 位量化

```python
from vllm import LLM
import torch
model_id = "huggyllama/llama-7b"
llm = LLM(model=model_id, dtype=torch.bfloat16, trust_remote_code=True, \
quantization="bitsandbytes", load_format="bitsandbytes")
```


