---
title: AutoAWQ
---

您可以使用 [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) 创建新的 4-bit 量化模型。量化将模型的精度从 FP16 降低到 INT4，从而有效地将文件大小减少约 70%，这样做的主要优势在于较低的延迟和内存使用量。

您可以通过安装 AutoAWQ 或选择 [Huggingface 上的 400+ 模型](https://huggingface.co/models?sort=trending&search=awq)之一来量化自己的模型。

```plain
pip install autoawq
```

安装 AutoAWQ 后，您就可以量化模型了。以下是如何量化 _mistralai/Mistral-7B-Instruct-v0.2_ 的示例：

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

To run an AWQ model with vLLM, you can use [TheBloke/Llama-2-7b-Chat-AWQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-AWQ) with the following command:
如需使用 vLLM 运行 AWQ 模型，您可以通过以下命令使用 [TheBloke/Llama-2-7b-Chat-AWQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-AWQ)：

```plain
python examples/llm_engine_example.py --model TheBloke/Llama-2-7b-Chat-AWQ --quantization awq
```

AWQ 模型也可以直接由 LLM 入口点支持：

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
