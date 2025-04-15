---
title: FP8 E4M3 KV 缓存
---


将 KV 缓存量化为 FP8 可以减少其内存占用。这增加缓存中可以存储的 token 数量，从而提高了吞吐量。 OCP（Open Compute Project，开放计算项目 [www.opencompute.org](http://www.opencompute.org）)）指定了两种常见的 8 位浮点数据格式：E5M2（5 个指数位和 2 个尾数位）和 E4M3FN（4 个指数位和 3 个尾数位），通常缩写为 E4M3。E4M3 格式相较于 E5M2 的优势在于，它能够以更高的精度来表示浮点数。然而，FP8 E4M3 的小动态范围（通常只能表示 ±240.0） 通常需要在每个量化张量配合使用 1 个更高精度的缩放因子（通常为 FP32）。目前，仅支持每个张量（标量）缩放因子。团队正在开发，以支持更细粒度的缩放因子（例如每个通道的缩放因子）。


这些缩放因子可以通过在加载时将可选的量化参数 JSON 传递给 LLM 引擎来进行指定。如果未指定此 JSON，则缩放因子默认为 1.0。这些缩放因子通常在通过量化器工具 （例如 AMD 量化器或 NVIDIA AMMO）运行 1 个未量化模型的过程中获得的。


安装 AMMO（AlgorithMic 模型优化）：

```plain
pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com nvidia-ammo
```


研究表明，FP8 E4M3 量化通常只会最小程度地降低推理精度。最新的硅芯产品，例如 AMD MI300、NVIDIA Hopper 及更高版本支持原生硬件在 fp32、fp16、bf16 等格式之间进行转换。因此 LLM 的推理速度得到了大大的提升，同时精度损失达到最小。


以下是如何启用此功能的示例：

```python
# two float8_e4m3fn kv cache scaling factor files are provided under tests/fp8_kv, please refer to 
# 在tests/fp8_kv下提供了两个float8_e4m3fn kv缓存缩放因子文件，请参考


# https://github.com/vllm-project/vllm/blob/main/examples/fp8/README.md to generate kv_cache_scales.json of your own.
# https://github.com/vllm-project/vllm/blob/main/examples/fp8/README.md 生成您自己的 kv_cache_scales.json 。


from vllm import LLM, SamplingParams
sampling_params = SamplingParams(temperature=1.3, top_p=0.8)
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf",
          kv_cache_dtype="fp8",
          quantization_param_path="./tests/fp8_kv/llama2-7b-fp8-kv/kv_cache_scales.json")
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)


# output w/ scaling factors:  England, the United Kingdom, and one of the world's leading financial,
# 带缩放因子的输出：England, the United Kingdom, and one of the world's leading financial,


# output w/o scaling factors:  England, located in the southeastern part of the country. It is known 
# 不带缩放因子的输出：England, located in the southeastern part of the country. It is known 
```


