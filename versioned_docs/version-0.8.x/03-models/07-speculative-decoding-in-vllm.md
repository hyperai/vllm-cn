---
title: vLLM 中的推测解码
---

**警告：**

请注意，vLLM 中的推测解码尚未优化，通常不会为所有的提示数据集或采样参数带来 token 间延迟的减少。我们正在进行优化工作可以在[这个 issue](https://github.com/vllm-project/vllm/issues/4630) 中跟踪进展。


本文档展示了如何在使用 vLLM 时应用[推测解码](https://x.com/karpathy/status/1697318534555336961)。这种技术能够降低在内存密集型的 LLM 推理过程中，各个 token 之间的延迟。


## 用草稿模型进行推测

以下代码展示了在离线模式下配置 vLLM，并使用草稿模型进行推测解码，每次推测 5 个 token。

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


若要在线模式下执行相同的操作，请启动服务器：

```bash
python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --model facebook/opt-6.7b \
--seed 42 -tp 1 --speculative_model facebook/opt-125m --use-v2-block-manager \
--num_speculative_tokens 5 --gpu_memory_utilization 0.8
```
然后使用一个客户端：
```bash
from openai import OpenAI


# Modify OpenAI's API key and API base to use vLLM's API server.
# 修改 OpenAI 的 API 密钥和 API 库以使用 vLLM 的 API 服务器。


openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    # 默认为 os.environ.get("OPENAI_API_KEY")


    api_key=openai_api_key,
    base_url=openai_api_base,
)


models = client.models.list()
model = models.data[0].id


# Completion API


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


## 通过在提示符中匹配 n-grams 进行推测

以下代码配置了 vLLM 使用推测解码，其中通过匹配提示中的 n-grams 生成建议。更多信息请阅读[此线程](https://x.com/joao_gante/status/1747322413006643259)。

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

以下代码配置 vLLM 使用推测性解码，其中提案由草稿模型生成，该草稿模型根据上下文向量和采样 token 调节草稿预测。有关更多信息，请参阅[此博客](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)或[此技术报告](https://arxiv.org/abs/2404.19124)。

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


请注意，这些推测模型当前需要在没有张量并行性的情况下运行，尽管可以使用张量并行性运行主模型 （参见上面的示例）。由于推测模型相对较小，我们仍然能够观察到显著的速度提升。不过此限制将在未来版本中修复。


HF hub 上提供了多种此类推测模型：

* [llama-13b-accelerator](https://huggingface.co/ibm-fms/llama-13b-accelerator)

* [llama3-8b-accelerator](https://huggingface.co/ibm-fms/llama3-8b-accelerator)

* [codellama-34b-accelerator](https://huggingface.co/ibm-fms/codellama-34b-accelerator)

* [llama2-70b-accelerator](https://huggingface.co/ibm-fms/llama2-70b-accelerator)

* [llama3-70b-accelerator](https://huggingface.co/ibm-fms/llama3-70b-accelerator)

* [granite-3b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-3b-code-instruct-accelerator)

* [granite-8b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-8b-code-instruct-accelerator)

* [granite-7b-instruct-accelerator](https://huggingface.co/ibm-granite/granite-7b-instruct-accelerator)

* [granite-20b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-20b-code-instruct-accelerator)


## 相关 vLLM 贡献者的资源

* [黑客指南：vLLM 中的推测解码](https://www.youtube.com/watch?v=9wNAgpX6z_4)

* [什么是 vLLM 中的预取调度？](https://docs.google.com/document/d/1Z9TvqzzBPnh5WHcRwjvK2UEeFeq5zMZb5mFE8jR0HCs/edit#heading=h.1fjfb0donq5a)

* [关于批量扩展的信息](https://docs.google.com/document/d/1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORppiwaj5asxA/edit#heading=h.kk7dq05lc6q8)

* [动态推测解码](https://github.com/vllm-project/vllm/issues/4565)


