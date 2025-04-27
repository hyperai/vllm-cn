---
title: 分离式预填充（实验性功能）
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

> **警告**
> 
> 请注意，vLLM 中的推测解码尚未优化，通常不会减少所有提示数据集或采样参数的 token 间延迟。优化工作正在进行中，可以在 [这个 issue](https://github.com/vllm-project/vllm/issues/4630) 中进行跟进。

> **警告**
> 
> 目前 vLLM 中的推测编码并不兼容流水线多线程。

本文档介绍如何将[推测解码](https://x.com/karpathy/status/1697318534555336961)与 vLLM 结合使用。推测性解码是一种可改善内存绑定 LLM 推理中 token 间延迟的技术。

## 用草稿模型进行推测

以下代码在离线模式下配置 vLLM，以对草稿模型使用推测解码，一次推测 5 个 token 。

```python
from vllm import LLM, SamplingParams


prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


llm = LLM(
    model="facebook/opt-6.7b",
    tensor_parallel_size=1,
    speculative_model="facebook/opt-125m",
    num_speculative_tokens=5,
    use_v2_block_manager=True,
)
outputs = llm.generate(prompts, sampling_params)


for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

如需使用在线模式执行相同的操作，请启动服务器：

```bash
python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --model facebook/opt-6.7b \
--seed 42 -tp 1 --speculative_model facebook/opt-125m --use-v2-block-manager \
--num_speculative_tokens 5 --gpu_memory_utilization 0.8
```

> **警告**
> 
> 注意：请使用 `--speculative_config` 设置所有与推测解码相关的配置。之前通过 `--speculative_model` 指定模型并单独添加相关参数（例如 `--num_speculative_tokens`）的方法将在下一个版本中弃用。

然后使用客户端：

```python
from openai import OpenAI


# 修改 OpenAI 的 API 密钥和 API 库以使用 vLLM 的 API 服务器。


openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


client = OpenAI(
# 默认为 os.environ.get("OPENAI_API_KEY")


    api_key=openai_api_key,
    base_url=openai_api_base,
)


models = client.models.list()
model = models.data[0].id


# 补全 API


stream = False
completion = client.completions.create(
    model=model,
    prompt="The future of AI is",
    echo=False,
    n=1,
    stream=stream,
)


print("Completion results:")
if stream:
    for c in completion:
        print(c)
else:
    print(completion)
```

## 通过匹配提示中的 n-grams 进行推测

以下代码配置 vLLM 使用推测解码，其中通过匹配提示中的 n-grams 生成建议。有关更多信息，请阅读[此线程](https://x.com/joao_gante/status/1747322413006643259)。

```python
from vllm import LLM, SamplingParams


prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


llm = LLM(
    model="facebook/opt-6.7b",
    tensor_parallel_size=1,
    speculative_model="[ngram]",
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,
    use_v2_block_manager=True,
)
outputs = llm.generate(prompts, sampling_params)


for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## 使用 MLP 推测器进行推测

以下代码配置了 vLLM 以使用推测性解码，其中提案由草稿模型生成，该草稿模型根据上下文向量和采样 token 调节草稿预测。有关更多信息，请参阅[此博客](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)或[此技术报告](https://arxiv.org/abs/2404.19124)。

```python
from vllm import LLM, SamplingParams


prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_model="ibm-fms/llama3-70b-accelerator",
    speculative_draft_tensor_parallel_size=1,
    use_v2_block_manager=True,
)
outputs = llm.generate(prompts, sampling_params)


for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

请注意，这些推测模型当前需要在没有张量并行性的情况下运行，尽管可以使用张量并行性运行主模型 （参见上面的示例）。由于推测模型相对较小，我们仍然看到显着的加速。不过此限制将在未来版本中修复。

HF hub 上提供了多种此类推测模型：

- [llama-13b-accelerator](https://huggingface.co/ibm-fms/llama-13b-accelerator)
- [llama3-8b-accelerator](https://huggingface.co/ibm-fms/llama3-8b-accelerator)
- [codellama-34b-accelerator](https://huggingface.co/ibm-fms/codellama-34b-accelerator)
- [llama2-70b-accelerator](https://huggingface.co/ibm-fms/llama2-70b-accelerator)
- [llama3-70b-accelerator](https://huggingface.co/ibm-fms/llama3-70b-accelerator)
- [granite-3b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-3b-code-instruct-accelerator)
- [granite-8b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-8b-code-instruct-accelerator)
- [granite-7b-instruct-accelerator](https://huggingface.co/ibm-granite/granite-7b-instruct-accelerator)
- [granite-20b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-20b-code-instruct-accelerator)

### 使用基于 EAGLE 的草稿模型进行推测解码

以下代码配置了 vLLM 以使用推测解码，其中提案由基于 [EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency)](https://arxiv.org/pdf/2401.15077)）的草稿模型生成。

```plain
from vllm import LLM, SamplingParams


prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=4,
    speculative_model="yuhuili/EAGLE-LLaMA3-Instruct-8B",
    speculative_draft_tensor_parallel_size=1,
)


outputs = llm.generate(prompts, sampling_params)


for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


```

使用基于 EAGLE 的草稿模型时需要注意的事项：

1. 在 [PR 12304](https://github.com/vllm-project/vllm/pull/12304) 之后，[EAGLE 模型的 HF 仓库](https://huggingface.co/yuhuili)中提供的 EAGLE 草稿模型应能够直接被 vLLM 加载和使用。如果您使用的是 [PR 12304](https://github.com/vllm-project/vllm/pull/12304) 之前的 vLLM 版本，请使用[脚本](https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d)转换推测模型，并指定 `speculative_model="path/to/modified/eagle/model"`。如果使用最新版本的 vLLM 时仍然出现权重加载问题，请留言或提交 issue。

2. 基于 EAGLE 的草稿模型需要在没有张量并行的情况下运行（即 speculative_draft_tensor_parallel_size 设置为 1），尽管主模型可以使用张量并行运行（参见上面的示例）。

3. 当使用基于 EAGLE 的推测器与 vLLM 时，观察到的加速效果低于参考实现中报告的效果。此问题正在调查中，跟踪链接：[vllm-project/vllm#9565](https://github.com/vllm-project/vllm/issues/9565)。

Hugging Face Hub 上提供了多种 EAGLE 草稿模型：

| 基础模型                   | Hugging Face 上的 EAGLE             | # EAGLE 参数 |
| :------------------------- | :---------------------------------- | :----------- |
| Vicuna-7B-v1.3             | yuhuili/EAGLE-Vicuna-7B-v1.3        | 0.24B        |
| Vicuna-13B-v1.3            | yuhuili/EAGLE-Vicuna-13B-v1.3       | 0.37B        |
| Vicuna-33B-v1.3            | yuhuili/EAGLE-Vicuna-33B-v1.3       | 0.56B        |
| LLaMA2-Chat 7B             | yuhuili/EAGLE-llama2-chat-7B        | 0.24B        |
| LLaMA2-Chat 13B            | yuhuili/EAGLE-llama2-chat-13B       | 0.37B        |
| LLaMA2-Chat 70B            | yuhuili/EAGLE-llama2-chat-70B       | 0.99B        |
| Mixtral-8x7B-Instruct-v0.1 | yuhuili/EAGLE-mixtral-instruct-8x7B | 0.28B        |
| LLaMA3-Instruct 8B         | yuhuili/EAGLE-LLaMA3-Instruct-8B    | 0.25B        |
| LLaMA3-Instruct 70B        | yuhuili/EAGLE-LLaMA3-Instruct-70B   | 0.99B        |
| Qwen2-7B-Instruct          | yuhuili/EAGLE-Qwen2-7B-Instruct     | 0.26B        |
| Qwen2-72B-Instruct         | yuhuili/EAGLE-Qwen2-72B-Instruct    | 1.05B        |

### 推测解码的无损保证

在 vLLM 中，推测解码旨在提高推理效率的同时保持准确性。本节讨论推测解码的无损保证，并将其分解为 3 个关键领域：

1. **理论无损性** - 推测解码采样在理论上是无损的，直到硬件数值精度的极限。浮点误差可能会导致输出分布的轻微变化，正如  [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/pdf/2302.01318) 中所讨论的那样。

2. **算法无损性** - vLLM 的推测解码实现经过算法验证是无损的。关键的验证测试包括：

   1. **拒绝采样器收敛性**：确保 vLLM 的拒绝采样器生成的样本与目标分布一致。[查看测试代码](https://github.com/vllm-project/vllm/blob/main/tests/spec_decode/test_rejection_sampler.py)

   2. **贪婪采样一致性**：确认使用推测解码的贪婪采样与不使用推测解码的贪婪采样结果一致。这验证了 vLLM 的推测解码框架在与 vLLM 前向传递和 vLLM 拒绝采样器集成时提供了无损保证。几乎所有的测试都在 [tests/spec_decode/e2e](https://github.com/vllm-project/vllm/tree/main/tests/spec_decode/e2e). 中通过 [此断言实现](https://github.com/vllm-project/vllm/blob/main/tests/spec_decode/e2e/test_greedy_equality.py) 验证了这一特性。

3. **vLLM 对数概率稳定性** - vLLM 目前不保证 token 对数概率（logprobs）的稳定性。这可能导致同一请求在不同运行中生成不同的输出。更多详情请参阅 FAQ 部分中的「[Can the output of a prompt vary across runs in vLLM?](https://docs.vllm.ai/en/stable/faq.html#can-the-output-of-a-prompt-vary-across-runs-in-vllm)」。

尽管 vLLM 努力确保推测解码的无损性，但由于以下因素，使用和不使用推测解码生成的输出可能会有所不同：

- **浮点精度**：硬件数值精度的差异可能导致输出分布的轻微不一致。
- **批量大小和数值稳定性**：批量大小的变化可能会导致 logprobs 和输出概率的变化，这可能是由于批量操作中的非确定性行为或数值不稳定性。

有关缓解策略，请参阅 [FAQ](https://docs.vllm.ai/en/latest/getting_started/faq.html#faq) 中的「Can the output of a prompt vary across runs in vLLM?」。

## 面向 vLLM 贡献者的资源

- [黑客指南：vLLM 中的推测解码](https://www.youtube.com/watch?v=9wNAgpX6z_4)
- [什么是 vLLM 中的预取调度？](https://docs.google.com/document/d/1Z9TvqzzBPnh5WHcRwjvK2UEeFeq5zMZb5mFE8jR0HCs/edit#heading=h.1fjfb0donq5a)
- [关于批量扩展的信息](https://docs.google.com/document/d/1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORppiwaj5asxA/edit#heading=h.kk7dq05lc6q8)
- [动态推测解码](https://github.com/vllm-project/vllm/issues/4565)
