---
title: 量化内核支持的硬件
---


下表展示了 vLLM 中各种量化实现与不同硬件平台的兼容性情况：

|Implementation|Volta|Turing|Ampere|Ada|Hopper|AMD GPU|Intel GPU|x86 CPU|AWS Inferentia|Google TPU|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|AWQ|✗|✅︎|✅︎|✅︎|✅︎|✗|✗|✅︎|✗|✗|
|GPTQ|✅︎|✅︎|✅︎|✅︎|✅︎|✗|✗|✗|✗|✗|
|Marlin (GPTQ/AWQ/FP8)|✗|✗|✅︎|✅︎|✅︎|✗|✗|✗|✗|✗|
|INT8 (W8A8)|✗|✅︎|✅︎|✅︎|✅︎|✗|✗|✅︎|✗|✗|
|FP8 (W8A8)|✗|✗|✗|✅︎|✅︎|✅︎|✗|✗|✗|✗|
|AQLM|✅︎|✅︎|✅︎|✅︎|✅︎|✗|✗|✗|✗|✗|
|bitsandbytes|✅︎|✅︎|✅︎|✅︎|✅︎|✗|✗|✗|✗|✗|
|DeepSpeedFP|✅︎|✅︎|✅︎|✅︎|✅︎|✗|✗|✗|✗|✗|
|GGUF|✅︎|✅︎|✅︎|✅︎|✅︎|✗|✗|✗|✗|✗|

## 注意:

* Volta 对应 SM 7.0，Turing 对应 SM 7.5，Ampere 对应 SM 8.0/8.6，Ada 对应 SM 8.9，Hopper 对应 SM 9.0。

* 「✅︎」表示在指定硬件上支持该量化方法。

* 「✗」表示在指定硬件上不支持该量化方法。


请注意，随着 vLLM 不断发展并扩展其对不同硬件平台和量化方法的支持，此兼容性图表可能会更新变化。


获取有关硬件支持和量化方法的最新信息，请查阅[量化目录](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization)或咨询 vLLM 开发团队。

