---
title: GPTQModel
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

要创建新的 4 位或 8 位 GPTQ 量化模型，您可以利用 ModelCloud.AI 的 [GPTQModel](https://github.com/ModelCloud/GPTQModel)。

量化将模型的精度从 BF16/FP16（16 位）降低到 INT4（4 位）或 INT8（8 位），从而显著减少了模型的总内存占用，同时提高了推理性能。

兼容的 GPTQModel 量化模型可以利用 Marlin 和 Machete vLLM 自定义内核来最大限度地提高 Ampere (A100+) 和 Hopper (H100+) Nvidia GPU 的每秒批处理事务 tps 和令牌延迟性能。这两个内核由 vLLM 和 NeuralMagic（现在是 Redhat 的一部分）进行高度优化，以实现量化 GPTQ 模型的世界级推理性能。

GPTQModel 是世界上为数不多的允许动态每模块量化的量化工具包之一，其中 llm 模型中的不同层和/或模块可以使用自定义量化参数进一步优化。 动态量化完全集成到 vLLM 中，并由 ModelCloud.AI 团队提供支持。请参阅 [GPTQModel 自述文件](https://github.com/ModelCloud/GPTQModel?tab=readme-ov-file#dynamic-quantization-per-module-quantizeconfig-override)了解有关此功能和其他高级功能的更多详细信息。

您可以通过安装 [GPTQModel](https://github.com/ModelCloud/GPTQModel) 或从 [Huggingface 上的 5000+ 模型中](https://huggingface.co/models?sort=trending&search=gptq)选择一个来量化自己的模型。

```
pip install -U gptqmodel --no-build-isolation -v
```

安装 GPTQModel 后，您就可以量化模型了。有关更多详细信息，请参阅 [GPTQModel 自述文件](https://github.com/ModelCloud/GPTQModel/?tab=readme-ov-file#quantization)。

以下是如何量化的示例 meta-llama/Llama-3.2-1B-Instruct：

```
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
  ).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)

# 增加`batch_size` 以匹配 GPU/VRAM 的规格，从而加快量化速度。
model.quantize(calibration_dataset, batch_size=2)

model.save(quant_path)
```

要使用 vLLM 运行 GPTQModel 量化模型，您可以通过以下命令使用 [DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2](https://huggingface.co/ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2)：

```
python examples/offline_inference/llm_engine_example.py --model DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2
```

GPTQModel 量化模型也直接通过 LLM 入口点支持：

````
from vllm import LLM, SamplingParams

# 采样输入提示
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# 创建一个采样参数对象
sampling_params = SamplingParams(temperature=0.6, top_p=0.9)

# 创建一个 LLM.
llm = LLM(model="DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2")
# 从提示生成文本。输出是一个包含提示、生成的文本以及其他信息的 RequestOutput 对象列表。
outputs = llm.generate(prompts, sampling_params)
# 打印输出
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```
````
