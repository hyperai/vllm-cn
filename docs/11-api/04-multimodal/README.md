---

title: 多模态支持

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM 通过 `vllm.multimodal` 包提供对多模态模型的实验性支持。


多模态输入可以与文本和 token 提示一起传递给[支持的模型](https://docs.vllm.ai/en/latest/models/supported_models.html#supported-mm-models)，通过 `vllm.inputs.PromptType` 中的 `multi_modal_data` 字段传递。


想要添加自己的多模态模型？请按照[此处](https://docs.vllm.ai/en/latest/contributing/model/multimodal.html#supports-multimodal)列出的说明操作。


## 模块内容

**vllm.multimodal.****MULTIMODAL_REGISTRY*****= <vllm.multimodal.registry.MultiModalRegistry object>***

[[source]](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py#L101)


全局的 `MultiModalRegistry` 被模型运行器用于根据目标模型分派数据处理。


>**另请参阅**
>[多模态数据处理](https://docs.vllm.ai/en/latest/design/mm_processing.html#mm-processing)

## 子模块

* [输入定义](https://docs.vllm.ai/en/latest/api/multimodal/inputs.html)
* [数据解析](https://docs.vllm.ai/en/latest/api/multimodal/parse.html)
* [数据处理](https://docs.vllm.ai/en/latest/api/multimodal/processing.html)
* [内存分析](https://docs.vllm.ai/en/latest/api/multimodal/profiling.html)
* [注册表](https://docs.vllm.ai/en/latest/api/multimodal/registry.html)


