---

title: 兼容矩阵

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


下表展示了互斥特性和对某些硬件的支持


以下为所使用符号的含义：


* ✅ = 完全兼容

* 🟠 = 部分兼容

* ❌ = 不兼容


>**注意**
>检查 ❌ 或 🟠 的连接，查看不支持功能、硬件组合的问题的进展。
## 

## 功能 x 功能

|功能|[CP](https://docs.vllm.ai/en/latest/performance/optimization.html#chunked-prefill)|[APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html#automatic-prefix-caching)|[LoRA](https://docs.vllm.ai/en/latest/features/lora.html#lora-adapter)|prmpt adptr|[SD](https://docs.vllm.ai/en/latest/features/spec_decode.html)|CUDA graph|pooling|enc-dec|logP|prmpt logP|async output|multi-step|mm|best-of|beam-search|guided dec|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|[CP](https://docs.vllm.ai/en/latest/performance/optimization.html#chunked-prefill)|✅||||||||||||||||
|[APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html#automatic-prefix-caching)|✅|✅|||||||||||||||
|[LoRA](https://docs.vllm.ai/en/latest/features/lora.html#lora-adapter)|✅|✅|✅||||||||||||||
|prmpt adptr|✅|✅|✅|✅|||||||||||||
|[SD](https://docs.vllm.ai/en/latest/features/spec_decode.html)|✅|✅|❌|✅|✅||||||||||||
|CUDA graph|✅|✅|✅|✅|✅|✅|||||||||||
|pooling|❌|❌|❌|❌|❌|❌|✅||||||||||
|enc-dec|❌|[❌](https://github.com/vllm-project/vllm/issues/7366#)|❌|❌|[❌](https://github.com/vllm-project/vllm/issues/7366#)|✅|✅|✅|||||||||
|logP|✅|✅|✅|✅|✅|✅|❌|✅|✅||||||||
|prmpt logP|✅|✅|✅|✅|✅|✅|❌|✅|✅|✅|||||||
|async output|✅|✅|✅|✅|❌|✅|❌|❌|✅|✅|✅||||||
|multi-step|❌|✅|❌|✅|❌|✅|❌|❌|✅|✅|✅|✅|||||
|mm|✅|[🟠](https://github.com/vllm-project/vllm/pull/8348#)|[🟠](https://github.com/vllm-project/vllm/pull/4194#)|❔|❔|✅|✅|✅|✅|✅|✅|❔|✅||||
|best-of|✅|✅|✅|✅|[❌](https://github.com/vllm-project/vllm/issues/6137#)|✅|❌|✅|✅|✅|❔|[❌](https://github.com/vllm-project/vllm/issues/7968#)|✅|✅|||
|beam-search|✅|✅|✅|✅|[❌](https://github.com/vllm-project/vllm/issues/6137#)|✅|❌|✅|✅|✅|❔|[❌](https://github.com/vllm-project/vllm/issues/7968#)|❔|✅|✅||
|guided dec|✅|✅|❔|❔|[❌](https://github.com/vllm-project/vllm/issues/11484#)|✅|❌|❔|✅|✅|✅|[❌](https://github.com/vllm-project/vllm/issues/9893#)|❔|✅|✅|✅|


## 功能 x 硬件

|Feature|Volta|Turing|Ampere|Ada|Hopper|CPU|AMD|
|:----|:----|:----|:----|:----|:----|:----|:----|
|[CP](https://docs.vllm.ai/en/latest/performance/optimization.html#chunked-prefill)|[❌](https://github.com/vllm-project/vllm/issues/2729#)|✅|✅|✅|✅|✅|✅|
|[APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html#automatic-prefix-caching)|[❌](https://github.com/vllm-project/vllm/issues/3687#)|✅|✅|✅|✅|✅|✅|
|[LoRA](https://docs.vllm.ai/en/latest/features/lora.html#lora-adapter)|✅|✅|✅|✅|✅|✅|✅|
|prmpt adptr|✅|✅|✅|✅|✅|[❌](https://github.com/vllm-project/vllm/issues/8475#)|✅|
|[SD](https://docs.vllm.ai/en/latest/features/spec_decode.html)|✅|✅|✅|✅|✅|✅|✅|
|CUDA graph|✅|✅|✅|✅|✅|❌|✅|
|pooling|✅|✅|✅|✅|✅|✅|❔|
|enc-dec|✅|✅|✅|✅|✅|✅|❌|
|mm|✅|✅|✅|✅|✅|✅|✅|
|logP|✅|✅|✅|✅|✅|✅|✅|
|prmpt logP|✅|✅|✅|✅|✅|✅|✅|
|async output|✅|✅|✅|✅|✅|❌|❌|
|multi-step|✅|✅|✅|✅|✅|[❌](https://github.com/vllm-project/vllm/issues/8477#)|✅|
|best-of|✅|✅|✅|✅|✅|✅|✅|
|beam-search|✅|✅|✅|✅|✅|✅|✅|
|guided dec|✅|✅|✅|✅|✅|✅|✅|



