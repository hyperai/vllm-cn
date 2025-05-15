---
title: 自动前缀缓存
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://openbayes.com/console/public/tutorials/rXxb5fZFr29?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

[PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) 的核心思想是将每个请求的 KV 缓存划分为多个 KV 块。每个块包含固定数量的注意力键和值。PagedAttention 算法允许这些块存储在不连续的物理内存中，从而通过按需分配内存来消除内存碎片。

为了自动缓存 KV 缓存，我们利用以下关键观察：每个 KV 块可以通过块中的 token 以及块之前的前缀 token 唯一标识。

```plain
                    Block 1                  Block 2                  Block 3
         [A gentle breeze stirred] [the leaves as children] [laughed in the distance]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
Block 3: |<------------------ prefix -------------------->| |<--- block tokens ---->|
```

在上面的例子中，第一个块中的 KV 缓存可以通过 token「A gentle breeze stirred」唯一标识。第三个块可以通过块中的 token「laughed in the distance」，以及前缀 token「A gentle breeze stirred the leaves as children」唯一标识。因此，我们可以构建以下一对一映射：

```plain
hash(prefix tokens + block tokens) <--> KV Block
```

通过这种映射，我们可以在 vLLM 的 KV 缓存管理中增加另一层间接性。此前，vLLM 中的每个序列维护从其逻辑 KV 块到物理块的映射。为了实现 KV 块的自动缓存，我们将逻辑 KV 块映射到其哈希值，并维护所有物理块的全局哈希表。这样，所有共享相同哈希值的 KV 块（例如，跨两个请求的共享前缀块）可以映射到相同的物理块并共享内存空间。

该设计实现了自动前缀缓存，无需在 KV 块之间维护树结构。更具体地说，所有块彼此独立，可以独立分配和释放，这使我们能够像操作系统中的普通缓存一样管理 KV 缓存。

# 通用缓存策略

将所有 KV 块存储在哈希表中，使得 vLLM 可以缓存来自早期请求的 KV 块，以节省内存并加速未来请求的计算。例如，如果新请求与之前的请求共享系统提示词，则可以直接使用共享提示词的 KV 缓存，而无需重新计算。然而，总的 KV 缓存空间是有限的，当缓存满时，我们必须决定保留哪些 KV 块，淘汰哪些块。

使用哈希表管理 KV 缓存允许我们实现灵活的缓存策略。例如，在当前的 vLLM 中，我们实现了以下淘汰策略：

- 当没有剩余的空闲块时，我们将淘汰引用计数（即当前使用该块的请求数）为 0 的 KV 块。
- 如果有多个引用计数为 0 的块，我们优先淘汰最近最少使用 (LRU) 的块。
- 如果多个块的最后访问时间相同，我们优先淘汰前缀最长的块（即，在它之前的块数量最多的块）。

请注意，当应用于具有全注意力的模型时，此淘汰策略实际上实现了与 [RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/) 中完全相同的策略，优先淘汰引用计数为 0 且最近最少使用的前缀树叶节点。

然而，基于哈希的 KV 缓存管理使我们能够灵活应对更复杂的服务场景，并实现超越上述策略的更复杂的淘汰策略：

- 多 LoRA 服务。在为多个 LoRA 适配器服务时，我们可以让每个 KV 块的哈希还包括请求查询的 LoRA ID，以启用所有适配器的缓存。通过这种方式，我们可以共同管理不同适配器的 KV 块，简化系统实现并提高全局缓存命中率和效率。
- 多模态模型。当用户输入不仅仅包括离散的 token 时，我们可以使用不同的哈希方法来处理不同模态输入的缓存。例如，对图像进行感知哈希以缓存相似的输入图像。
