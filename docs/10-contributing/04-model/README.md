---

title: 添加新模型

---

[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


本节提供了更多关于如何将 [PyTorch](https://pytorch.org/) 模型集成到 vLLM 中的信息。


#### 目录

* [实现基础模型](https://docs.vllm.ai/en/latest/contributing/model/basic.html)

* [将模型注册到 vLLM](https://docs.vllm.ai/en/latest/contributing/model/registration.html)

* [编写单元测试](https://docs.vllm.ai/en/latest/contributing/model/tests.html)

* [多模态支持](https://docs.vllm.ai/en/latest/contributing/model/multimodal.html)


>**注意**
>添加新模型的复杂性很大程度上取决于模型的架构。如果模型与 vLLM 中现有模型的架构相似，那么这个过程会相对简单。然而，对于包含新操作符（例如新的注意力机制）的模型，这个过程可能会稍微复杂一些。

>**提示**
>如果您在将模型集成到 vLLM 时遇到问题，请随时在 [GitHub issue](https://github.com/vllm-project/vllm/issues) 中提出，或在我们的[开发者 Slack](https://slack.vllm.ai/) 上提问。我们将很乐意为您提供帮助！

