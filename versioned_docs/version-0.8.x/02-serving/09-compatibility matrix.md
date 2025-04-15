---
title: 兼容性矩阵
---


下表显示了互斥的特性以及在某些硬件上的支持情况。


**注意：**

对于标记为「✗」的条目，您可以通过提供的链接查看不支持的特性或硬件组合的详细跟踪问题。


## Feature x Feature

|Feature|[CP](https://docs.vllm.ai/en/latest/models/performance.html#chunked-prefill)|[APC](https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html#apc)|[LoRA](https://docs.vllm.ai/en/latest/models/lora.html#lora)|prmpt adptr|[SD](https://docs.vllm.ai/en/latest/models/spec_decode.html#spec-decode)|CUDA graph|enc-dec|logP|prmpt logP|async output|multi-step|MM|best-of|beam-search|guided dec|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|[CP](https://docs.vllm.ai/en/latest/models/performance.html#chunked-prefill)||||||||||||||||
|[APC](https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html#apc)|✅|||||||||||||||
|[LoRA](https://docs.vllm.ai/en/latest/models/lora.html#lora)|[✗](https://github.com/vllm-project/vllm/pull/9057)|✅||||||||||||||
|prmpt adptr|✅|✅|✅|||||||||||||
|[SD](https://docs.vllm.ai/en/latest/models/spec_decode.html#spec-decode)|✗|✅|✗|✅||||||||||||
|CUDA graph|✅|✅|✅|✅|✅|||||||||||
|enc-dec|✗|[✗](https://github.com/vllm-project/vllm/issues/7366)|✗|✗|[✗](https://github.com/vllm-project/vllm/issues/7366)|✅||||||||||
|logP|✅|✅|✅|✅|✅|✅|✅|||||||||
|prmpt logP|✅|✅|✅|✅|[✗](https://github.com/vllm-project/vllm/pull/8199)|✅|✅|✅||||||||
|async output|✅|✅|✅|✅|✗|✅|✗|✅|✅|||||||
|multi-step|✗|✅|✗|✅|✗|✅|✗|✅|[✗](https://github.com/vllm-project/vllm/issues/8198)|✅||||||
|MM|[✗](https://github.com/vllm-project/vllm/pull/8346)|[✗](https://github.com/vllm-project/vllm/pull/8348)|[✗](https://github.com/vllm-project/vllm/pull/7199)|?|?|✅|✗|✅|✅|✅|?|||||
|best-of|✅|✅|✅|✅|[✗](https://github.com/vllm-project/vllm/issues/6137)|✅|✅|✅|✅|?|[✗](https://github.com/vllm-project/vllm/issues/7968)|✅||||
|beam-search|✅|✅|✅|✅|[✗](https://github.com/vllm-project/vllm/issues/6137)|✅|✅|✅|✅|?|[✗](https://github.com/vllm-project/vllm/issues/7968)|?|✅|||
|guided dec|✅|✅|?|?|✅|✅|?|✅|✅|✅|[✗](https://github.com/vllm-project/vllm/issues/8985)|?|✅|✅||


## Feature x Hardware

|Feature|Volta|Turing|Ampere|Ada|Hopper|CPU|AMD|
|:----|:----|:----|:----|:----|:----|:----|:----|
|[CP](https://docs.vllm.ai/en/latest/models/performance.html#chunked-prefill)|[✗](https://github.com/vllm-project/vllm/issues/2729)|✅|✅|✅|✅|✗|✅|
|[APC](https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html#apc)|[✗](https://github.com/vllm-project/vllm/issues/3687)|✅|✅|✅|✅|✗|✅|
|[LoRA](https://docs.vllm.ai/en/latest/models/lora.html#lora)|✅|✅|✅|✅|✅|[✗](https://github.com/vllm-project/vllm/pull/4830)|✅|
|prmpt adptr|✅|✅|✅|✅|✅|[✗](https://github.com/vllm-project/vllm/issues/8475)|✅|
|[SD](https://docs.vllm.ai/en/latest/models/spec_decode.html#spec-decode)|✅|✅|✅|✅|✅|✅|✅|
|CUDA graph|✅|✅|✅|✅|✅|✗|✅|
|enc-dec|✅|✅|✅|✅|✅|[✗](https://github.com/vllm-project/vllm/blob/a84e598e2125960d3b4f716b78863f24ac562947/vllm/worker/cpu_model_runner.py#L125)|✗|
|logP|✅|✅|✅|✅|✅|✅|✅|
|prmpt logP|✅|✅|✅|✅|✅|✅|✅|
|async output|✅|✅|✅|✅|✅|✗|✗|
|multi-step|✅|✅|✅|✅|✅|[✗](https://github.com/vllm-project/vllm/issues/8477)|✅|
|MM|✅|✅|✅|✅|✅|✅|✅|
|best-of|✅|✅|✅|✅|✅|✅|✅|
|beam-search|✅|✅|✅|✅|✅|✅|✅|
|guided dec|✅|✅|✅|✅|✅|✅|✅|





