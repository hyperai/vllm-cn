---
title: AutoAWQ
---


**警告：**

注意，目前 vLLM 中的 AWQ 支持尚未进行优化。我们建议使用模型的非量化版本以获得更好的精度和更高的吞吐量。当前，您可以使用 AWQ 作为减少内存占用的一种方法。截至目前，它更适合并发请求数量较少的低延迟推理。 vLLM 的 AWQ 实现的吞吐量低于未量化版本。


如需创建新的 4-bit 量化模型，您可以利用 [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)。量化操作会将模型的精度从 FP16 降低到 INT4，从而有效地将文件大小减少约 70%。其主要优势在于更低的延迟和更少的内存占用。


您可以通过安装 AutoAWQ 或从 [Huggingface 上的 400 多个模型](https://huggingface.co/models?sort=trending&search=awq)中选择一个来量化您自己的模型。

```plain
pip install autoawq
```


安装 AutoAWQ 后，您就可以对模型进行量化了。以下是一个如何量化 *mistralai/Mistral-7B-Instruct-v0.2* 的示例：

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
quant_path = 'mistral-instruct-v0.2-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }


# Load model
# 加载模型


model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


# Quantize
# 量化


model.quantize(tokenizer, quant_config=quant_config)


# Save quantized model
# 保存量化模型


model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)


print(f'Model is quantized and saved at "{quant_path}"')
```


如需使用 vLLM 运行一个 AWQ 模型，您可以使用 [TheBloke/Llama-2-7b-Chat-AWQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-AWQ)，并配合以下命令：

```plain
python examples/llm_engine_example.py --model TheBloke/Llama-2-7b-Chat-AWQ --quantization awq
```


AWQ 模型也可以直接由 LLM 入口点获得支持：

```python
from vllm import LLM, SamplingParams


# Sample prompts.
# 提示示例。


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
# 创建一个 SamplingParams 对象。


sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


# Create an LLM.
# 创建一个 LLM。


llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="AWQ")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
# 从提示中生成文本。输出是一个 RequestOutput 列表，包含提示、生成文本和其他信息


outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
# 打印输出。


for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```


