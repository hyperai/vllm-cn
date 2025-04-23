---

title: Transformers 强化学习

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

Transformers 强化学习 (TRL) 是一个全栈库，提供了一套工具，用于通过监督微调 (SFT)、组相对策略优化 (GRPO)、直接偏好优化 (DPO)、奖励建模等方法训练 Transformer 语言模型。该库与 🤗 Transformers 集成。


在线方法（如 GRPO 或在线 DPO）需要模型生成补全内容。vLLM 可用于生成这些补全内容！


有关更多信息，请参阅 TRL 文档中的指南：[vLLM 用于在线方法中的快速生成](https://huggingface.co/docs/trl/main/en/speeding_up_training#vllm-for-fast-generation-in-online-methods)。


>**另请参阅**
>
>有关这些在线方法配置中可提供的 `use_vllm` 标志的更多信息，请参阅：
>`trl.GRPOConfig.use_vllm`
>`trl.OnlineDPOConfig.use_vllm`

