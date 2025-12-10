---
title: TorchAO
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

TorchAO 是 PyTorch 的架构优化库，它为推理和训练提供高性能的 dtype、优化技术和内核，具有与原生 PyTorch 功能（如 torch.compile、FSDP 等）的可组合性。可以[在此处](https://github.com/pytorch/ao/tree/main/torchao/quantization#benchmarks)找到一些基准数字。

我们建议安装最新的 torchao nightly。

## 量化 HuggingFace 模型

你可以使用 torchao 量化自己的 huggingface 模型，例如 [transformers](https://huggingface.co/docs/transformers/main/en/quantization/torchao) 和 [diffusers](https://huggingface.co/docs/diffusers/en/quantization/torchao)，并使用以下示例代码将检查点保存到 huggingface hub，如下所示：

```
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int8WeightOnlyConfig

model_name = "meta-llama/Meta-Llama-3-8B"
quantization_config = TorchAoConfig(Int8WeightOnlyConfig())
quantized_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

hub_repo = # YOUR HUB REPO ID
tokenizer.push_to_hub(hub_repo)
quantized_model.push_to_hub(hub_repo, safe_serialization=False)

```

或者，您可以使用 TorchAO 量化空间通过简单的 UI 量化模型。另请参阅： https://huggingface.co/spaces/medmekk/。TorchAO_Quantization
