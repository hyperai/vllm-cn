---
title: 池化模型
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

vLLM 还支持池化模型，包括嵌入模型、重排序模型和奖励模型。

在 vLLM 中，池化模型实现了 VllmModelForPooling 接口。这些模型在返回之前，会使用 1 个池化器 (Pooler) 提取输入的最终隐藏状态。

> **注意：**
> 我们目前支持池化模型主要是为了方便。如[兼容性矩阵](#compatibility-matrix)所示，大多数 vLLM 功能不适用于池化模型，因为它们仅适用于生成或解码阶段，因此性能提升可能并不明显。

对于池化模型，我们支持以下 --task 选项。所选选项设置了用于提取最终隐藏状态的默认池化器：

| 任务             | 池化类型 | 归一化 | Softmax |
| :--------------- | :------- | :----- | :------ |
| 嵌入 (embed)     | LAST     | ✅︎    | ❌      |
| 分类(classify)   | LAST     | ❌     | ✅︎     |
| 句子评分(score)  | \*       | \*     | \*      |
| 奖励模型(reward) | ALL      | ❌     | ❌      |

\*默认池化器始终由模型定义。

> **注意：**
> 如果模型在 vLLM 中的实现定义了自己的池化器，则默认池化器将设置为该池化器，而不是此表中指定的池化器。

当加载 [Sentence Transformers](https://huggingface.co/sentence-transformers) 模型时，我们会尝试根据其 Sentence Transformers 配置文件 (modules.json) 覆盖默认池化器。

**提示：**

您可以通过 --override-pooler-config 选项自定义模型的池化方法，该选项优先于模型和 Sentence Transformers 的默认设置。Offline Inference

## 离线推理

LLM 类提供了多种用于离线推理的方法。获取初始化模型时的选项列表，请参阅[引擎参数](#engine-args)。

### LLM.encode

encode 方法适用于 vLLM 中的所有池化模型。这个方法能够直接提供提取出的隐藏状态信息，这对于构建奖励模型来说非常有用。

```python
llm = LLM(model="Qwen/Qwen2.5-Math-RM-72B", task="reward")
(output,) = llm.encode("Hello, my name is")
data = output.outputs.data
print(f"Data: {data!r}")
```

### LLM.embed

embed 方法为每个提示输出一个嵌入向量，它主要为嵌入模型设计。

```python
llm = LLM(model="intfloat/e5-mistral-7b-instruct", task="embed")
(output,) = llm.embed("Hello, my name is")
embeds = output.outputs.embedding
print(f"Embeddings: {embeds!r} (size={len(embeds)})")
```

代码示例可以在这里找到：[examples/offline_inference/basic/embed.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic/embed.py)

### LLM.classify

classify 方法为每个提示输出一个概率向量，它主要为分类模型设计。

```python
llm = LLM(model="jason9693/Qwen2.5-1.5B-apeach", task="classify")
(output,) = llm.classify("Hello, my name is")
probs = output.outputs.probs
print(f"Class Probabilities: {probs!r} (size={len(probs)})")
```

代码示例可以在这里找到：[examples/offline_inference/basic/classify.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic/classify.py)

### LLM.score

score 方法输出句子对之间的相似度分数。它主要为交叉编码器模型设计，这类模型在 RAG 系统中用作候选查询-文档对之间的重排序器。

> **注意：**
> vLLM 只能执行 RAG 的模型推理组件（例如嵌入、重排序）。如需更高级地处理 RAG，建议使用集成框架，例如 [LangChain](https://github.com/langchain-ai/langchain)。

```python
llm = LLM(model="BAAI/bge-reranker-v2-m3", task="score")
(output,) = llm.score("What is the capital of France?",
                   "The capital of Brazil is Brasilia.")
score = output.outputs.score
print(f"Score: {score}")
```

代码示例可以在这里找到：[examples/offline_inference/basic/score.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic/score.py)

## 在线服务

我们的 OpenAI 兼容服务器提供了与离线 API 对应的端点：

- 池化 API 类似于 LLM.encode，适用于所有类型的池化模型。
- 嵌入 API 类似于 LLM.embed，接受嵌入模型的文本和多模态输入。
- 评分 API 类似于 LLM.score，适用于交叉编码器模型。
