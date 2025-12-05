---
title: 优化与调优
---

[\*在线运行 vLLM 入门教程：零基础分步指南](https://app.hyper.ai/console/public/tutorials/rUwYsyhAIt3?utm_source=vLLM-CNdoc&utm_medium=vLLM-CNdoc-V1&utm_campaign=vLLM-CNdoc-V1-25ap)

## 抢占 (Preemption)

由于 Transformer 架构的自回归特性，有时 KV 缓存空间不足以处理所有批量请求。 vLLM 可以抢占请求，为其他请求释放 KV 缓存空间。当有足够的 KV 缓存空间再次可用时，被抢占的请求将重新计算。发生这种情况时，会打印以下警告：

`WARNING 05-09 00:49:33 scheduler.py:1057] Sequence group 0 is preempted by PreemptionMode.SWAP mode because there is not enough KV cache space. This can affect the end-to-end performance. Increase gpu_memory_utilization or tensor_parallel_size to provide more KV cache memory. total_cumulative_preemption_cnt=1`

虽然这种机制确保了系统的稳健性，但抢占和重新计算可能会对端到端延迟有不利影响。如果您经常遇到 vLLM 引擎的抢占，请考虑以下操作：

- 增加 `gpu_memory_utilization`。 vLLM 使用内存的 gpu_memory_utilization% 预分配 GPU 缓存。提高此利用率，您可以提供更多的 KV 缓存空间。
- 减少 `max_num_seqs` 或 `max_num_batched_tokens`。这可以减少批量中并发请求的数量，从而需要更少的 KV 缓存空间。
- 增加 `tensor_parallel_size`。该方法对模型权重进行了分片，因此每个 GPU 都有更多的内存用于 KV 缓存。
- 增加 `pipeline_parallel_size`。该方法将模型层分布到多个 GPU 上，减少每个 GPU 上模型权重所需的内存，从而间接为 KV 缓存释放更多内存。

您还可以通过 vLLM 公开的 Prometheus 指标来监控抢占请求的数量。此外，您可以通过设置 disable_log_stats=False 来记录抢占请求的累积数量。

## 分块预填充

vLLM 支持实验性功能——分块预填充。分块预填充允许将大预填充分块成更小的块，并将它们与解码请求一起批处理。

您可以通过在命令行中指定 `--enable-chunked-prefill` 或在 LLM 构造函数中设置 `enable_chunked_prefill=True` 来启用该功能。

```python
llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_chunked_prefill=True)
# 设置 max_num_batched_tokens 来调整性能。
# 注意：2048 是分块预填充的 max_num_batched_tokens 默认值。
# llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_chunked_prefill=True, max_num_batched_tokens=2048)
```

默认情况下，vLLM 调度程序会优先考虑预填充，并且不会将预填充和解码分批处理到同一批次。此策略优化了 TTFT（time to the first token，第一个 token 的时间），但会导致 ITL（inter token latency，token 间延迟）变慢和 GPU 利用率低下。

一旦启用分块预填充，策略就会更改为优先考虑解码请求。在安排任何预填充之前，它会将所有待处理的解码请求分批处理。当有可用的 token_budget (`max_num_batched_tokens`) 时，它会安排待处理的预填充。如果最后一个待处理的预填充请求无法放入 `max_num_batched_tokens`，则会将其分块。

该策略有两个好处：

- 它改进了 ITL 和生成解码，因为解码请求更优先。
- 它通过将计算绑定 （预填充） 和内存绑定 （解码） 请求定位到同一批次，可以实现更好的 GPU 利用率。

您可以通过更改 `max_num_batched_tokens` 调整性能。默认情况下，它设置为 512，在初始基准测试中，它在 A100 上具有最佳 ITL（llama 70B 和 mixtral 8x22B）。较小的 max_num_batched_tokens 可以实现更好的 ITL，因为中断解码的预填充较少。较高的 max_num_batched_tokens 可以实现更好的 TTFT，因为您可以在批次中添加更多预填充。

- 如果 `max_num_batched_tokens` 与 `max_model_len` 相同，则几乎等同于默认调度策略 （但是它仍然优先考虑解码）。
- 请注意， `max_num_batched_tokens` 的默认值 （512） 针对 ITL 进行了优化，并且它的吞吐量可能低于默认调度程序。

我们建议您设置 `max_num_batched_tokens > 2048` 提高吞吐量。

有关更多详细信息，请参阅相关论文（[https://arxiv.org/pdf/2401.08671](https://arxiv.org/pdf/2401.08671) 或 [https://arxiv.org/pdf/2308.16369](https://arxiv.org/pdf/2308.16369)）。

请尝试此功能并通过 GitHub issues 让我们知道您的反馈！
