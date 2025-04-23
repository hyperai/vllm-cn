---

title: RLHF 基于人类反馈的强化学习

---


[*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)


基于人类反馈的强化学习 (Reinforcement Learning from Human Feedback, RLHF) 是一种利用人类生成的偏好数据微调语言模型的技术，以使模型输出与期望行为保持一致。


vLLM 可用于生成 RLHF 的补全内容。最佳实践是使用诸如 [TRL](https://github.com/huggingface/trl)、[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 和 [verl](https://github.com/volcengine/verl) 等库。


如果您不想使用现有库，可以参考以下基础示例入门：


* [训练和推理过程位于不同的 GPU 上（灵感来自 OpenRLHF）](https://docs.vllm.ai/en/latest/getting_started/examples/rlhf.html)

* [使用 Ray 将训练和推理过程共置于同一 GPU 上](https://docs.vllm.ai/en/latest/getting_started/examples/rlhf_colocate.html)

* [使用 vLLM 执行 RLHF 的实用工具](https://docs.vllm.ai/en/latest/getting_started/examples/rlhf_utils.html)


